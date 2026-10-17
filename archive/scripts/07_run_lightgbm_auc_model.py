#!/usr/bin/env python3
"""Train a nonlinear class-balanced LightGBM model optimized for ROC-AUC."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


LABELS = [f"label_ec{index}" for index in range(1, 8)]
CLASS_NAMES = [
    "Oxidoreductases",
    "Transferases",
    "Hydrolases",
    "Lyases",
    "Isomerases",
    "Ligases",
    "Translocases",
]
RESIDUE_GROUPS = {
    "hydrophobic": set("AVILMFWY"),
    "charged": set("DEKR"),
    "polar": set("STNQ"),
    "aromatic": set("FWY"),
    "small": set("AGST"),
    "glycine": {"G"},
    "proline": {"P"},
    "cysteine": {"C"},
    "histidine": {"H"},
}


def physicochemical_features(sequences: pd.Series) -> np.ndarray:
    rows = []
    canonical = set("ACDEFGHIKLMNPQRSTVWY")
    for sequence in sequences:
        residues = [aa for aa in sequence if aa in canonical]
        length = max(len(residues), 1)
        counts = np.array([residues.count(aa) for aa in sorted(canonical)], dtype=float)
        probabilities = counts[counts > 0] / length
        entropy = float(-(probabilities * np.log2(probabilities)).sum())
        longest_hydrophobic = 0
        current = 0
        for aa in sequence:
            if aa in RESIDUE_GROUPS["hydrophobic"]:
                current += 1
                longest_hydrophobic = max(longest_hydrophobic, current)
            else:
                current = 0
        row = [np.log1p(len(sequence)), entropy, longest_hydrophobic / length]
        row.extend(sum(aa in group for aa in residues) / length for group in RESIDUE_GROUPS.values())
        rows.append(row)
    return np.asarray(rows, dtype=np.float32)


def load_features(data: pd.DataFrame, cache: Path) -> np.ndarray:
    accessions = cache.with_suffix(".accessions.txt").read_text(encoding="utf-8").splitlines()
    if accessions != data["accession"].tolist():
        raise ValueError("Feature cache order does not match the master dataset")
    composition = np.load(cache)
    physical = physicochemical_features(data["sequence"])
    return np.hstack([composition, physical])


def build_model(seed: int) -> lgb.LGBMClassifier:
    return lgb.LGBMClassifier(
        objective="binary",
        n_estimators=2500,
        learning_rate=0.025,
        num_leaves=31,
        max_depth=-1,
        min_child_samples=20,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.05,
        reg_lambda=1.0,
        class_weight="balanced",
        random_state=seed,
        n_jobs=4,
        verbosity=-1,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--master", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--split-column", required=True)
    parser.add_argument("--feature-cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data = pd.read_csv(args.master, dtype={"taxonomy_id": str})
    manifest = pd.read_csv(args.split_manifest, dtype={"taxonomy_id": str})[["accession", args.split_column]]
    data = data.merge(manifest, on="accession", how="inner", validate="one_to_one")
    x = load_features(data, args.feature_cache.resolve())
    y = data[LABELS].to_numpy(dtype=int)
    multi = data["is_multifunctional_top_level"].astype(int).to_numpy()
    split = data[args.split_column].to_numpy()
    train, validation, test = split == "train", split == "validation", split == "test"

    probabilities = np.zeros((test.sum(), 7), dtype=float)
    models = []
    per_class = []
    for index, class_name in enumerate(CLASS_NAMES):
        model = build_model(args.seed + index)
        model.fit(
            x[train],
            y[train, index],
            eval_set=[(x[validation], y[validation, index])],
            eval_metric="auc",
            callbacks=[lgb.early_stopping(120, verbose=False), lgb.log_evaluation(0)],
        )
        probability = model.predict_proba(x[test], num_iteration=model.best_iteration_)[:, 1]
        probabilities[:, index] = probability
        models.append(model)
        per_class.append(
            {
                "model": "AAC_DPC_Physchem_LightGBM",
                "split_design": args.split_column,
                "ec_class": index + 1,
                "class_name": class_name,
                "roc_auc": roc_auc_score(y[test, index], probability),
                "pr_auc": average_precision_score(y[test, index], probability),
                "support": int(y[test, index].sum()),
                "best_iteration": int(model.best_iteration_),
            }
        )

    auxiliary = build_model(args.seed + 100)
    auxiliary.fit(
        x[train],
        multi[train],
        eval_set=[(x[validation], multi[validation])],
        eval_metric="auc",
        callbacks=[lgb.early_stopping(120, verbose=False), lgb.log_evaluation(0)],
    )
    multi_probability = auxiliary.predict_proba(x[test], num_iteration=auxiliary.best_iteration_)[:, 1]

    metrics = {
        "model": "AAC_DPC_Physchem_LightGBM",
        "split_design": args.split_column,
        "seed": args.seed,
        "macro_roc_auc": roc_auc_score(y[test], probabilities, average="macro"),
        "micro_roc_auc": roc_auc_score(y[test], probabilities, average="micro"),
        "weighted_roc_auc": roc_auc_score(y[test], probabilities, average="weighted"),
        "macro_pr_auc": average_precision_score(y[test], probabilities, average="macro"),
        "micro_pr_auc": average_precision_score(y[test], probabilities, average="micro"),
        "auxiliary_roc_auc": roc_auc_score(multi[test], multi_probability),
        "auxiliary_pr_auc": average_precision_score(multi[test], multi_probability),
        "min_class_roc_auc": min(row["roc_auc"] for row in per_class),
    }

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([metrics]).to_csv(output / "optimized_auc_metrics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(per_class).to_csv(output / "optimized_per_class_auc.csv", index=False, encoding="utf-8-sig")
    predictions = data.loc[test, ["accession", "top_classes", "is_multifunctional_top_level"]].copy()
    for index in range(7):
        predictions[f"prob_ec{index + 1}"] = probabilities[:, index]
    predictions["multifunction_probability"] = multi_probability
    predictions.to_csv(output / "optimized_test_probabilities.csv", index=False, encoding="utf-8-sig")
    for index, model in enumerate(models, start=1):
        joblib.dump(model, output / f"lightgbm_ec{index}.joblib")
    joblib.dump(auxiliary, output / "lightgbm_multifunction.joblib")
    report = {
        **metrics,
        "feature_dimension": int(x.shape[1]),
        "train_n": int(train.sum()),
        "validation_n": int(validation.sum()),
        "test_n": int(test.sum()),
        "per_class_roc_auc": {row["class_name"]: row["roc_auc"] for row in per_class},
    }
    (output / "optimized_auc_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

