#!/usr/bin/env python3
"""Run a reproducible AAC+DPC logistic-regression multi-task baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    hamming_loss,
    jaccard_score,
    label_ranking_average_precision_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
AA_INDEX = {aa: index for index, aa in enumerate(AMINO_ACIDS)}
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


def features_for_sequence(sequence: str) -> np.ndarray:
    aac = np.zeros(20, dtype=np.float32)
    dpc = np.zeros((20, 20), dtype=np.float32)
    valid = [AA_INDEX[aa] for aa in sequence if aa in AA_INDEX]
    if valid:
        for index in valid:
            aac[index] += 1
        aac /= len(valid)
    valid_pairs = 0
    for left, right in zip(sequence, sequence[1:]):
        if left in AA_INDEX and right in AA_INDEX:
            dpc[AA_INDEX[left], AA_INDEX[right]] += 1
            valid_pairs += 1
    if valid_pairs:
        dpc /= valid_pairs
    return np.concatenate([aac, dpc.ravel()])


def load_or_create_features(data: pd.DataFrame, cache: Path) -> np.ndarray:
    accession_cache = cache.with_suffix(".accessions.txt")
    if cache.exists() and accession_cache.exists():
        accessions = accession_cache.read_text(encoding="utf-8").splitlines()
        if accessions == data["accession"].tolist():
            return np.load(cache)
    matrix = np.vstack([features_for_sequence(sequence) for sequence in data["sequence"]])
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache, matrix)
    accession_cache.write_text("\n".join(data["accession"]) + "\n", encoding="utf-8")
    return matrix


def best_binary_threshold(y_true: np.ndarray, probability: np.ndarray) -> float:
    candidates = np.linspace(0.05, 0.95, 91)
    scores = [f1_score(y_true, probability >= threshold, zero_division=0) for threshold in candidates]
    return float(candidates[int(np.argmax(scores))])


def adaptive_decode(probabilities: np.ndarray, multi_probability: np.ndarray, class_thresholds: np.ndarray, multi_threshold: float) -> np.ndarray:
    predictions = np.zeros_like(probabilities, dtype=int)
    for row_index, row in enumerate(probabilities):
        if multi_probability[row_index] < multi_threshold:
            predictions[row_index, int(np.argmax(row))] = 1
        else:
            selected = np.flatnonzero(row >= class_thresholds)
            if len(selected) < 2:
                selected = np.argsort(row)[-2:]
            predictions[row_index, selected] = 1
    return predictions


def evaluate(y_true: np.ndarray, y_pred: np.ndarray, probabilities: np.ndarray, multi_true: np.ndarray, multi_pred: np.ndarray, multi_probability: np.ndarray) -> dict[str, float]:
    mono = y_true.sum(axis=1) == 1
    true_single = y_true[mono].argmax(axis=1)
    pred_single = probabilities[mono].argmax(axis=1)
    return {
        "multilabel_macro_roc_auc": roc_auc_score(y_true, probabilities, average="macro"),
        "multilabel_micro_roc_auc": roc_auc_score(y_true, probabilities, average="micro"),
        "multilabel_macro_pr_auc": average_precision_score(y_true, probabilities, average="macro"),
        "multilabel_macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "multilabel_micro_f1": f1_score(y_true, y_pred, average="micro", zero_division=0),
        "multilabel_jaccard_samples": jaccard_score(y_true, y_pred, average="samples", zero_division=0),
        "multilabel_hamming_loss": hamming_loss(y_true, y_pred),
        "multilabel_exact_match": accuracy_score(y_true, y_pred),
        "multilabel_lrap": label_ranking_average_precision_score(y_true, probabilities),
        "mono_accuracy": accuracy_score(true_single, pred_single),
        "mono_balanced_accuracy": balanced_accuracy_score(true_single, pred_single),
        "mono_macro_f1": f1_score(true_single, pred_single, average="macro", zero_division=0),
        "mono_mcc": matthews_corrcoef(true_single, pred_single),
        "auxiliary_auroc": roc_auc_score(multi_true, multi_probability),
        "auxiliary_auprc": average_precision_score(multi_true, multi_probability),
        "auxiliary_f1": f1_score(multi_true, multi_pred, zero_division=0),
        "auxiliary_mcc": matthews_corrcoef(multi_true, multi_pred),
    }


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
    if len(data) == 0:
        raise ValueError("No accessions matched the split manifest")
    x = load_or_create_features(data, args.feature_cache.resolve())
    y = data[LABELS].to_numpy(dtype=int)
    multi = data["is_multifunctional_top_level"].astype(int).to_numpy()
    split = data[args.split_column].to_numpy()
    train, validation, test = split == "train", split == "validation", split == "test"

    classifier = make_pipeline(
        StandardScaler(),
        OneVsRestClassifier(
            LogisticRegression(class_weight="balanced", max_iter=2000, random_state=args.seed),
            # Single-process execution avoids Windows joblib failures when the
            # temporary path contains non-ASCII characters.
            n_jobs=1,
        ),
    )
    auxiliary = make_pipeline(
        StandardScaler(),
        LogisticRegression(class_weight="balanced", max_iter=2000, random_state=args.seed),
    )
    classifier.fit(x[train], y[train])
    auxiliary.fit(x[train], multi[train])

    validation_prob = classifier.predict_proba(x[validation])
    class_thresholds = np.array(
        [best_binary_threshold(y[validation, index], validation_prob[:, index]) for index in range(7)]
    )
    validation_multi_prob = auxiliary.predict_proba(x[validation])[:, 1]
    multi_threshold = best_binary_threshold(multi[validation], validation_multi_prob)

    test_prob = classifier.predict_proba(x[test])
    test_multi_prob = auxiliary.predict_proba(x[test])[:, 1]
    test_pred = adaptive_decode(test_prob, test_multi_prob, class_thresholds, multi_threshold)
    test_multi_pred = (test_multi_prob >= multi_threshold).astype(int)
    metrics = evaluate(y[test], test_pred, test_prob, multi[test], test_multi_pred, test_multi_prob)

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    model_name = "AAC_DPC_LogReg_MTL"
    metric_row = {"model": model_name, "split_design": args.split_column, "seed": args.seed, **metrics}
    pd.DataFrame([metric_row]).to_csv(output / "benchmark_metrics.csv", index=False, encoding="utf-8-sig")

    per_class = []
    for index, class_name in enumerate(CLASS_NAMES):
        per_class.append(
            {
                "model": model_name,
                "split_design": args.split_column,
                "ec_class": index + 1,
                "class_name": class_name,
                "precision": precision_score(y[test, index], test_pred[:, index], zero_division=0),
                "recall": recall_score(y[test, index], test_pred[:, index], zero_division=0),
                "f1": f1_score(y[test, index], test_pred[:, index], zero_division=0),
                "pr_auc": average_precision_score(y[test, index], test_prob[:, index]),
                "roc_auc": roc_auc_score(y[test, index], test_prob[:, index]),
                "threshold": class_thresholds[index],
                "support": int(y[test, index].sum()),
            }
        )
    pd.DataFrame(per_class).to_csv(output / "per_class_metrics.csv", index=False, encoding="utf-8-sig")

    predictions = data.loc[test, ["accession", "top_classes", "is_multifunctional_top_level"]].copy()
    predictions["predicted_classes"] = [
        ";".join(str(index + 1) for index in np.flatnonzero(row)) for row in test_pred
    ]
    predictions["predicted_multifunctional"] = test_multi_pred
    predictions["multifunction_probability"] = test_multi_prob
    for index in range(7):
        predictions[f"prob_ec{index + 1}"] = test_prob[:, index]
    predictions.to_csv(output / "baseline_test_predictions.csv", index=False, encoding="utf-8-sig")

    report = {
        "model": model_name,
        "feature_dimension": int(x.shape[1]),
        "train_n": int(train.sum()),
        "validation_n": int(validation.sum()),
        "test_n": int(test.sum()),
        "class_thresholds": class_thresholds.tolist(),
        "multi_threshold": multi_threshold,
        "metrics": metrics,
    }
    (output / "baseline_run_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
