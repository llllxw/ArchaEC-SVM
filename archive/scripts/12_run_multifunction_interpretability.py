#!/usr/bin/env python3
"""Historical multifunctionality and feature-ablation experiment source.

For the active feature-only Figure 5, use scripts 20 (training) and 21 (plotting).
Running this legacy main recreates historical figures and should not be used to
regenerate the current manuscript figure set.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import lightgbm as lgb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    hamming_loss,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from plot_style import (
    CLASS_PALETTE,
    MODEL_COLOR,
    SECONDARY_COLOR,
    TARGET_COLOR,
    apply_arial_style,
)


apply_arial_style()
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
LABELS = [f"label_ec{i}" for i in range(1, 8)]
CLASS_NAMES = [
    "Oxidoreductases", "Transferases", "Hydrolases", "Lyases",
    "Isomerases", "Ligases", "Translocases",
]
PHYS_NAMES = [
    "log_length", "sequence_entropy", "longest_hydrophobic_run",
    "hydrophobic_fraction", "charged_fraction", "polar_fraction",
    "aromatic_fraction", "small_fraction", "glycine_fraction",
    "proline_fraction", "cysteine_fraction", "histidine_fraction",
]
RESIDUE_GROUPS = {
    "hydrophobic": set("AVILMFWY"), "charged": set("DEKR"),
    "polar": set("STNQ"), "aromatic": set("FWY"), "small": set("AGST"),
    "glycine": {"G"}, "proline": {"P"}, "cysteine": {"C"}, "histidine": {"H"},
}


def physicochemical_features(sequences: pd.Series) -> np.ndarray:
    rows = []
    canonical = set(AMINO_ACIDS)
    for sequence in sequences.fillna(""):
        residues = [aa for aa in sequence if aa in canonical]
        length = max(len(residues), 1)
        counts = np.array([residues.count(aa) for aa in AMINO_ACIDS], dtype=float)
        probabilities = counts[counts > 0] / length
        entropy = float(-(probabilities * np.log2(probabilities)).sum())
        longest, current = 0, 0
        for aa in sequence:
            if aa in RESIDUE_GROUPS["hydrophobic"]:
                current += 1
                longest = max(longest, current)
            else:
                current = 0
        row = [np.log1p(len(sequence)), entropy, longest / length]
        row.extend(sum(aa in group for aa in residues) / length for group in RESIDUE_GROUPS.values())
        rows.append(row)
    return np.asarray(rows, dtype=np.float32)


def build_model(seed: int) -> lgb.LGBMClassifier:
    return lgb.LGBMClassifier(
        objective="binary", n_estimators=2500, learning_rate=0.025,
        num_leaves=31, min_child_samples=20, subsample=0.85,
        colsample_bytree=0.85, reg_alpha=0.05, reg_lambda=1.0,
        class_weight="balanced", random_state=seed, n_jobs=4, verbosity=-1,
    )


def force_nonempty(binary: np.ndarray, probabilities: np.ndarray) -> np.ndarray:
    result = binary.copy()
    empty = result.sum(axis=1) == 0
    result[empty, probabilities[empty].argmax(axis=1)] = 1
    return result


def decoding_metrics(name: str, truth: np.ndarray, predicted: np.ndarray) -> dict[str, float | str]:
    return {
        "strategy": name,
        "macro_f1": f1_score(truth, predicted, average="macro", zero_division=0),
        "micro_f1": f1_score(truth, predicted, average="micro", zero_division=0),
        "sample_f1": f1_score(truth, predicted, average="samples", zero_division=0),
        "exact_match": accuracy_score(truth, predicted),
        "hamming_loss": hamming_loss(truth, predicted),
    }


def bootstrap_auxiliary(y: np.ndarray, p: np.ndarray, seed: int, repetitions: int = 1000) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    roc_values, pr_values = [], []
    for _ in range(repetitions):
        indices = rng.integers(0, len(y), len(y))
        if np.unique(y[indices]).size < 2:
            continue
        roc_values.append(roc_auc_score(y[indices], p[indices]))
        pr_values.append(average_precision_score(y[indices], p[indices]))
    return {
        "roc_auc": roc_auc_score(y, p),
        "roc_auc_ci_low": np.quantile(roc_values, 0.025),
        "roc_auc_ci_high": np.quantile(roc_values, 0.975),
        "pr_auc": average_precision_score(y, p),
        "pr_auc_ci_low": np.quantile(pr_values, 0.025),
        "pr_auc_ci_high": np.quantile(pr_values, 0.975),
        "positive_prevalence": y.mean(),
        "bootstrap_repetitions": len(roc_values),
    }


def plot_auxiliary(y: np.ndarray, p: np.ndarray, metrics: dict, output: Path) -> None:
    fpr, tpr, _ = roc_curve(y, p)
    precision, recall, _ = precision_recall_curve(y, p)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.5))
    axes[0].plot(fpr, tpr, color=MODEL_COLOR, linewidth=2.2,
                 label=f"ROC-AUC = {metrics['roc_auc']:.3f}")
    axes[0].plot([0, 1], [0, 1], "--", color=TARGET_COLOR, linewidth=1)
    axes[0].set(xlabel="False-positive rate", ylabel="True-positive rate", title="Auxiliary multifunctionality ROC")
    axes[1].plot(recall, precision, color=SECONDARY_COLOR, linewidth=2.2,
                 label=f"PR-AUC = {metrics['pr_auc']:.3f}")
    axes[1].axhline(metrics["positive_prevalence"], linestyle="--", color=TARGET_COLOR, linewidth=1,
                    label=f"Prevalence = {metrics['positive_prevalence']:.3f}")
    axes[1].set(xlabel="Recall", ylabel="Precision", title="Auxiliary multifunctionality precision-recall")
    for ax in axes:
        ax.legend(frameon=False)
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(output / "figure_5A_multifunction_auc.png", dpi=300)
    fig.savefig(output / "figure_5A_multifunction_auc.pdf")
    plt.close(fig)


def plot_auc_bars(table: pd.DataFrame, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    x = np.arange(len(table))
    bars = ax.bar(x, table["macro_roc_auc"], color=CLASS_PALETTE[:len(table)], edgecolor="#666666", linewidth=0.6)
    ax.axhline(0.8, linestyle="--", color=TARGET_COLOR, linewidth=1.2, label="Target AUC = 0.80")
    ax.set_xticks(x, table["feature_set"], rotation=20, ha="right")
    ax.set_ylabel("Macro ROC-AUC")
    ax.set_ylim(0.5, 1.0)
    ax.set_title("Feature-group ablation on the random test set")
    for bar, value in zip(bars, table["macro_roc_auc"]):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.012, f"{value:.3f}", ha="center")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(output / "figure_5B_feature_ablation_auc.png", dpi=300)
    fig.savefig(output / "figure_5B_feature_ablation_auc.pdf")
    plt.close(fig)


def annotation_auc_table(test_data: pd.DataFrame, truth: np.ndarray, probabilities: np.ndarray) -> pd.DataFrame:
    rows = []
    for field, label in [("active_site", "Active-site annotation"), ("binding_site", "Binding-site annotation"), ("transmembrane", "Transmembrane annotation")]:
        present = test_data[field].fillna("").astype(str).str.strip().ne("").to_numpy()
        for state, mask in [("Present", present), ("Absent", ~present)]:
            scores = []
            for index in range(7):
                if mask.sum() and np.unique(truth[mask, index]).size == 2:
                    scores.append(roc_auc_score(truth[mask, index], probabilities[mask, index]))
            rows.append({"annotation": label, "status": state, "n": int(mask.sum()),
                         "evaluable_classes": len(scores), "macro_roc_auc": np.mean(scores) if scores else np.nan})
    return pd.DataFrame(rows)


def plot_annotation_auc(table: pd.DataFrame, output: Path) -> None:
    labels = table["annotation"].drop_duplicates().tolist()
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    width = 0.34
    for offset, (status, color) in zip([-width / 2, width / 2], [("Present", MODEL_COLOR), ("Absent", SECONDARY_COLOR)]):
        subset = table.set_index(["annotation", "status"]).loc[[(label, status) for label in labels]].reset_index()
        bars = ax.bar(x + offset, subset["macro_roc_auc"], width, label=status, color=color)
        for bar, value in zip(bars, subset["macro_roc_auc"]):
            ax.text(bar.get_x() + bar.get_width() / 2, value + 0.01, f"{value:.3f}", ha="center", fontsize=8)
    ax.axhline(0.8, linestyle="--", color=TARGET_COLOR, linewidth=1.1)
    ax.set_xticks(x, labels, rotation=15, ha="right")
    ax.set_ylabel("Macro ROC-AUC")
    ax.set_ylim(0.5, 1.0)
    ax.set_title("ROC-AUC stratified by curated UniProt annotation")
    ax.legend(frameon=False, loc="center left", bbox_to_anchor=(1.01, 0.5))
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(output / "figure_5C_annotation_stratified_auc.png", dpi=300)
    fig.savefig(output / "figure_5C_annotation_stratified_auc.pdf")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--master", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--feature-cache", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    master = pd.read_csv(args.master, dtype={"taxonomy_id": str})
    cached_accessions = args.feature_cache.with_suffix(".accessions.txt").read_text(encoding="utf-8").splitlines()
    if cached_accessions != master["accession"].tolist():
        raise ValueError("Feature cache order does not match master dataset")
    composition = np.load(args.feature_cache)
    features = np.hstack([composition, physicochemical_features(master["sequence"])])
    manifest = pd.read_csv(args.split_manifest)[["accession", "split_random"]]
    split_map = manifest.set_index("accession")["split_random"]
    split = master["accession"].map(split_map).to_numpy()
    if pd.isna(split).any():
        raise ValueError("Split manifest does not cover the master dataset")
    train, validation, test = split == "train", split == "validation", split == "test"
    truth = master[LABELS].to_numpy(dtype=int)
    multi = master["is_multifunctional_top_level"].astype(int).to_numpy()

    models = [joblib.load(args.model_dir / f"lightgbm_ec{i}.joblib") for i in range(1, 8)]
    auxiliary = joblib.load(args.model_dir / "lightgbm_multifunction.joblib")
    val_p = np.column_stack([model.predict_proba(features[validation], num_iteration=model.best_iteration_)[:, 1] for model in models])
    test_p = np.column_stack([model.predict_proba(features[test], num_iteration=model.best_iteration_)[:, 1] for model in models])
    val_aux = auxiliary.predict_proba(features[validation], num_iteration=auxiliary.best_iteration_)[:, 1]
    test_aux = auxiliary.predict_proba(features[test], num_iteration=auxiliary.best_iteration_)[:, 1]

    thresholds = []
    grid = np.linspace(0.05, 0.95, 181)
    for index in range(7):
        scores = [f1_score(truth[validation, index], val_p[:, index] >= threshold, zero_division=0) for threshold in grid]
        thresholds.append(float(grid[int(np.argmax(scores))]))
    aux_scores = [f1_score(multi[validation], val_aux >= threshold, zero_division=0) for threshold in grid]
    aux_threshold = float(grid[int(np.argmax(aux_scores))])
    pd.DataFrame({"ec_class": range(1, 8), "class_name": CLASS_NAMES, "validation_f1_threshold": thresholds}).to_csv(
        output / "adaptive_thresholds.csv", index=False, encoding="utf-8-sig")

    fixed = force_nonempty(test_p >= 0.5, test_p)
    per_class = force_nonempty(test_p >= np.asarray(thresholds), test_p)
    guided = np.zeros_like(per_class)
    predicted_multi = test_aux >= aux_threshold
    for row in range(len(guided)):
        if predicted_multi[row]:
            guided[row] = per_class[row]
            if guided[row].sum() < 2:
                guided[row, np.argsort(test_p[row])[-2:]] = 1
        else:
            guided[row, np.argmax(test_p[row])] = 1
    decoding = pd.DataFrame([
        decoding_metrics("Fixed 0.50", truth[test], fixed),
        decoding_metrics("Validation class-specific", truth[test], per_class),
        decoding_metrics("Auxiliary-guided adaptive", truth[test], guided),
    ])
    decoding.to_csv(output / "decoding_comparison.csv", index=False, encoding="utf-8-sig")

    auxiliary_metrics = bootstrap_auxiliary(multi[test], test_aux, args.seed)
    auxiliary_metrics.update({"validation_threshold": aux_threshold, "test_n": int(test.sum()), "test_positive_n": int(multi[test].sum())})
    pd.DataFrame([auxiliary_metrics]).to_csv(output / "multifunction_auxiliary_metrics.csv", index=False, encoding="utf-8-sig")
    plot_auxiliary(multi[test], test_aux, auxiliary_metrics, output)

    feature_sets = {
        "AAC only": np.arange(0, 20),
        "DPC only": np.arange(20, 420),
        "AAC + DPC": np.arange(0, 420),
        "Physicochemical only": np.arange(420, 432),
    }
    ablation_rows = []
    for set_index, (name, columns) in enumerate(feature_sets.items()):
        probabilities = np.zeros((test.sum(), 7))
        for class_index in range(7):
            model = build_model(args.seed + 1000 + set_index * 20 + class_index)
            model.fit(features[train][:, columns], truth[train, class_index],
                      eval_set=[(features[validation][:, columns], truth[validation, class_index])],
                      eval_metric="auc", callbacks=[lgb.early_stopping(120, verbose=False), lgb.log_evaluation(0)])
            probabilities[:, class_index] = model.predict_proba(features[test][:, columns], num_iteration=model.best_iteration_)[:, 1]
        ablation_rows.append({"feature_set": name, "n_features": len(columns),
                              "macro_roc_auc": roc_auc_score(truth[test], probabilities, average="macro"),
                              "micro_roc_auc": roc_auc_score(truth[test], probabilities, average="micro")})
    ablation_rows.append({"feature_set": "Full model", "n_features": 432,
                          "macro_roc_auc": roc_auc_score(truth[test], test_p, average="macro"),
                          "micro_roc_auc": roc_auc_score(truth[test], test_p, average="micro")})
    ablation = pd.DataFrame(ablation_rows)
    ablation.to_csv(output / "feature_ablation_auc.csv", index=False, encoding="utf-8-sig")
    plot_auc_bars(ablation, output)

    feature_names = [f"AAC_{aa}" for aa in AMINO_ACIDS]
    feature_names += [f"DPC_{left}{right}" for left in AMINO_ACIDS for right in AMINO_ACIDS]
    feature_names += PHYS_NAMES
    gain = np.zeros((7, len(feature_names)))
    for index, model in enumerate(models):
        values = model.booster_.feature_importance(importance_type="gain")
        gain[index] = values / max(values.sum(), 1)
    importance = pd.DataFrame({"feature": feature_names, "mean_normalized_gain": gain.mean(axis=0),
                               "sd_normalized_gain": gain.std(axis=0)})
    importance["feature_group"] = np.where(importance.index < 20, "AAC", np.where(importance.index < 420, "DPC", "Physicochemical"))
    importance.sort_values("mean_normalized_gain", ascending=False).to_csv(output / "feature_gain_importance.csv", index=False, encoding="utf-8-sig")

    test_data = master.loc[test].reset_index(drop=True)
    annotation = annotation_auc_table(test_data, truth[test], test_p)
    annotation.to_csv(output / "annotation_stratified_auc.csv", index=False, encoding="utf-8-sig")
    plot_annotation_auc(annotation, output)

    true_labels = [";".join(str(i + 1) for i, value in enumerate(row) if value) for row in truth[test]]
    predicted_labels = [";".join(str(i + 1) for i, value in enumerate(row) if value) for row in guided]
    exact = np.all(truth[test] == guided, axis=1)
    loss = -(truth[test] * np.log(np.clip(test_p, 1e-7, 1)) + (1 - truth[test]) * np.log(np.clip(1 - test_p, 1e-7, 1))).mean(axis=1)
    candidates = pd.DataFrame({"row": np.arange(len(test_data)), "exact": exact, "loss": loss,
                               "true_multifunctional": multi[test].astype(bool)})
    chosen = []
    for category, mask, ascending in [
        ("High-confidence correct", candidates["exact"], True),
        ("High-loss incorrect", ~candidates["exact"], False),
        ("Correct multifunctional", candidates["exact"] & candidates["true_multifunctional"], True),
    ]:
        subset = candidates.loc[mask].sort_values("loss", ascending=ascending).head(5)
        for row in subset["row"]:
            chosen.append((category, int(row)))
    representative_rows = []
    for category, row in chosen:
        record = test_data.iloc[row]
        representative_rows.append({
            "category": category, "accession": record["accession"], "protein_names": record["protein_names"],
            "organism": record["organism"], "true_classes": true_labels[row], "predicted_classes": predicted_labels[row],
            "multifunction_probability": test_aux[row], "mean_binary_cross_entropy": loss[row],
            "active_site": record["active_site"], "binding_site": record["binding_site"], "transmembrane": record["transmembrane"],
        })
    pd.DataFrame(representative_rows).drop_duplicates(subset=["category", "accession"]).to_csv(
        output / "representative_proteins.csv", index=False, encoding="utf-8-sig")

    scope = pd.DataFrame([
        {"component": "Auxiliary multifunctionality classifier", "present_in_current_model": True, "analysis": "ROC-AUC, PR-AUC, adaptive decoding"},
        {"component": "Class balancing", "present_in_current_model": True, "analysis": "Retained in all controlled feature ablations"},
        {"component": "Contrastive loss", "present_in_current_model": False, "analysis": "Not ablated; no such component exists"},
        {"component": "Local-motif/residue encoder", "present_in_current_model": False, "analysis": "No residue attribution claimed"},
    ])
    scope.to_csv(output / "component_scope.csv", index=False, encoding="utf-8-sig")

    report = {
        "status": "completed", "seed": args.seed,
        "auxiliary": auxiliary_metrics,
        "best_decoding_by_macro_f1": decoding.sort_values("macro_f1", ascending=False).iloc[0].to_dict(),
        "full_macro_roc_auc": float(ablation.loc[ablation.feature_set == "Full model", "macro_roc_auc"].iloc[0]),
        "best_reduced_feature_set": ablation.loc[ablation.feature_set != "Full model"].sort_values("macro_roc_auc", ascending=False).iloc[0].to_dict(),
        "residue_attribution_available": False,
    }
    (output / "multifunction_interpretability_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    best_decode = decoding.sort_values("macro_f1", ascending=False).iloc[0]
    best_reduced = ablation.loc[ablation.feature_set != "Full model"].sort_values("macro_roc_auc", ascending=False).iloc[0]
    draft = f"""# Results 3.4 - Multifunctionality and model interpretation

The auxiliary classifier discriminated top-level multifunctional enzymes with a
ROC-AUC of {auxiliary_metrics['roc_auc']:.3f} (95% bootstrap CI,
{auxiliary_metrics['roc_auc_ci_low']:.3f}-{auxiliary_metrics['roc_auc_ci_high']:.3f}) and a PR-AUC of
{auxiliary_metrics['pr_auc']:.3f}. Its validation-selected threshold was then used to guide
label cardinality during decoding. Among the evaluated decoding rules,
{best_decode['strategy']} achieved the highest macro-F1 ({best_decode['macro_f1']:.3f})
and an exact-match ratio of {best_decode['exact_match']:.3f}. These threshold-dependent
metrics complement, but do not alter, the threshold-independent ROC-AUC results.

Feature-group ablation showed that the full 432-dimensional representation achieved
a macro ROC-AUC of {report['full_macro_roc_auc']:.3f}. The strongest reduced representation was
{best_reduced['feature_set']} (macro ROC-AUC, {best_reduced['macro_roc_auc']:.3f}). Global
LightGBM gain scores were reported as model-level feature attribution. Stratification
by curated UniProt active-site, binding-site, and transmembrane annotations assessed
whether discrimination was consistent across annotated protein subsets.

The current composition-based LightGBM architecture has no residue encoder,
attention mechanism, contrastive objective, or local-motif component. Consequently,
residue-level catalytic-site attribution and ablation of absent neural components were
not performed. Feature importance is interpreted as predictive association rather
than mechanistic or catalytic evidence.
"""
    (output / "results_3.4_draft.md").write_text(draft, encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
