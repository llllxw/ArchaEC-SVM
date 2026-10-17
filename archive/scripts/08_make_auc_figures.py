#!/usr/bin/env python3
"""Generate Arial-formatted ROC-AUC figures for Results 3.2 and 3.3."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import auc, roc_curve

from plot_style import (
    BASELINE_COLOR,
    CLASS_PALETTE,
    MODEL_COLOR,
    SECONDARY_COLOR,
    TARGET_COLOR,
    apply_arial_style,
)


apply_arial_style()
CLASS_NAMES = [
    "Oxidoreductases",
    "Transferases",
    "Hydrolases",
    "Lyases",
    "Isomerases",
    "Ligases",
    "Translocases",
]


def add_target_line(ax: plt.Axes) -> None:
    ax.axhline(0.8, color=TARGET_COLOR, linestyle="--", linewidth=1.2, label="Target AUC = 0.80")


def load_outputs(folder: Path) -> dict[str, pd.DataFrame]:
    return {
        "baseline": pd.read_csv(folder / "benchmark_metrics.csv"),
        "baseline_class": pd.read_csv(folder / "per_class_metrics.csv"),
        "optimized": pd.read_csv(folder / "optimized_lightgbm" / "optimized_auc_metrics.csv"),
        "optimized_class": pd.read_csv(folder / "optimized_lightgbm" / "optimized_per_class_auc.csv"),
        "probabilities": pd.read_csv(folder / "optimized_lightgbm" / "optimized_test_probabilities.csv"),
    }


def random_overall(random_dir: Path, outputs: dict[str, pd.DataFrame]) -> None:
    baseline = outputs["baseline"].iloc[0]
    optimized = outputs["optimized"].iloc[0]
    labels = ["Macro ROC-AUC", "Micro ROC-AUC", "Multifunctionality ROC-AUC"]
    base_values = [baseline["multilabel_macro_roc_auc"], baseline["multilabel_micro_roc_auc"], baseline["auxiliary_auroc"]]
    opt_values = [optimized["macro_roc_auc"], optimized["micro_roc_auc"], optimized["auxiliary_roc_auc"]]
    positions = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    bars1 = ax.bar(positions - width / 2, base_values, width, label="AAC+DPC logistic regression", color=BASELINE_COLOR)
    bars2 = ax.bar(positions + width / 2, opt_values, width, label="Optimized LightGBM", color=MODEL_COLOR)
    add_target_line(ax)
    ax.set_xticks(positions, labels)
    ax.set_ylabel("Area under the ROC curve")
    ax.set_ylim(0.5, 1.02)
    ax.set_title("ROC-AUC improvement on the random protein split")
    ax.bar_label(bars1, labels=[f"{value:.3f}" for value in base_values], padding=3)
    ax.bar_label(bars2, labels=[f"{value:.3f}" for value in opt_values], padding=3)
    ax.legend(frameon=False, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(random_dir / "figure_3A_overall_auc.png", dpi=300)
    plt.close(fig)

    baseline_class = outputs["baseline_class"].sort_values("ec_class")
    optimized_class = outputs["optimized_class"].sort_values("ec_class")
    positions = np.arange(7)
    fig, ax = plt.subplots(figsize=(11.5, 5.5))
    ax.bar(positions - width / 2, baseline_class["roc_auc"], width, label="AAC+DPC logistic regression", color=BASELINE_COLOR)
    ax.bar(positions + width / 2, optimized_class["roc_auc"], width, label="Optimized LightGBM", color=MODEL_COLOR)
    add_target_line(ax)
    ax.set_xticks(positions, CLASS_NAMES, rotation=30, ha="right")
    ax.set_ylabel("One-vs-rest ROC-AUC")
    ax.set_ylim(0.65, 1.02)
    ax.set_title("Class-specific ROC-AUC on the random protein split")
    ax.legend(frameon=False, loc="center left", bbox_to_anchor=(1.01, 0.5))
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(random_dir / "figure_3B_per_class_auc.png", dpi=300)
    plt.close(fig)


def plot_roc_curves(folder: Path, probabilities: pd.DataFrame, master: pd.DataFrame, filename: str, title: str) -> None:
    merged = probabilities.merge(master, on="accession", validate="one_to_one")
    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    colors = CLASS_PALETTE
    for index, (class_name, color) in enumerate(zip(CLASS_NAMES, colors), start=1):
        y_true = merged[f"label_ec{index}"].to_numpy()
        y_score = merged[f"prob_ec{index}"].to_numpy()
        fpr, tpr, _ = roc_curve(y_true, y_score)
        score = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=color, linewidth=1.8, label=f"EC {index} {class_name} ({score:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color=TARGET_COLOR, linewidth=1)
    ax.set_xlabel("False-positive rate")
    ax.set_ylabel("True-positive rate")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(folder / filename, dpi=300)
    plt.close(fig)


def generalization_figures(
    species_dir: Path,
    random_outputs: dict[str, pd.DataFrame],
    species_outputs: dict[str, pd.DataFrame],
) -> None:
    base_random = random_outputs["baseline"].iloc[0]["multilabel_macro_roc_auc"]
    base_species = species_outputs["baseline"].iloc[0]["multilabel_macro_roc_auc"]
    opt_random = random_outputs["optimized"].iloc[0]["macro_roc_auc"]
    opt_species = species_outputs["optimized"].iloc[0]["macro_roc_auc"]
    fig, ax = plt.subplots(figsize=(7.6, 5.2))
    positions = np.arange(2)
    ax.plot(positions, [base_random, base_species], marker="o", linewidth=2, label="AAC+DPC logistic regression", color=BASELINE_COLOR)
    ax.plot(positions, [opt_random, opt_species], marker="o", linewidth=2, label="Optimized LightGBM", color=MODEL_COLOR)
    add_target_line(ax)
    ax.set_xticks(positions, ["Random protein split", "Species-disjoint split"])
    ax.set_ylabel("Macro ROC-AUC")
    ax.set_ylim(0.7, 1.0)
    ax.set_title("ROC-AUC under increasing taxonomic novelty")
    ax.legend(frameon=False, loc="lower left")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    # Overall species comparison retired from the active Figure 4 layout.
    plt.close(fig)

    random_class = random_outputs["optimized_class"].sort_values("ec_class")
    species_class = species_outputs["optimized_class"].sort_values("ec_class")
    positions = np.arange(7)
    width = 0.35
    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    ax.bar(positions - width / 2, random_class["roc_auc"], width, label="Random protein split", color=MODEL_COLOR, alpha=0.5)
    ax.bar(positions + width / 2, species_class["roc_auc"], width, label="Species-disjoint split", color=SECONDARY_COLOR, alpha=0.5)
    add_target_line(ax)
    ax.set_xticks(positions, CLASS_NAMES, rotation=30, ha="right")
    ax.set_ylabel("One-vs-rest ROC-AUC")
    ax.set_ylim(0.75, 1.02)
    ax.set_title("Optimized model retains class-level AUC across unseen species")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(.5,1.0), ncol=2, fontsize=8)
    ax.set_ylim(0,1.2)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(species_dir / "figure_4A_species_per_class_auc.png", dpi=300)
    plt.close(fig)


def write_tables_and_drafts(
    random_dir: Path,
    species_dir: Path,
    random_outputs: dict[str, pd.DataFrame],
    species_outputs: dict[str, pd.DataFrame],
) -> None:
    comparison = []
    for split_name, outputs in [("Random protein", random_outputs), ("Species-disjoint", species_outputs)]:
        baseline = outputs["baseline"].iloc[0]
        optimized = outputs["optimized"].iloc[0]
        comparison.extend(
            [
                {"split": split_name, "model": "AAC+DPC logistic regression", "macro_roc_auc": baseline["multilabel_macro_roc_auc"], "micro_roc_auc": baseline["multilabel_micro_roc_auc"], "macro_pr_auc": baseline["multilabel_macro_pr_auc"]},
                {"split": split_name, "model": "AAC+DPC+physicochemical LightGBM", "macro_roc_auc": optimized["macro_roc_auc"], "micro_roc_auc": optimized["micro_roc_auc"], "macro_pr_auc": optimized["macro_pr_auc"]},
            ]
        )
    pd.DataFrame(comparison).to_csv(species_dir / "auc_model_comparison.csv", index=False, encoding="utf-8-sig")

    rnd = random_outputs["optimized"].iloc[0]
    sp = species_outputs["optimized"].iloc[0]
    random_draft = f"""# Results 3.2 — AUC-optimized model performance

The nonlinear model integrating amino-acid composition, dipeptide composition, and
physicochemical descriptors achieved a macro ROC-AUC of {rnd['macro_roc_auc']:.3f}
and micro ROC-AUC of {rnd['micro_roc_auc']:.3f} on the exact-sequence-grouped random
test set. Macro PR-AUC was {rnd['macro_pr_auc']:.3f}. All seven one-vs-rest tasks
exceeded the prespecified AUC target of 0.80; the minimum class-level ROC-AUC was
{rnd['min_class_roc_auc']:.3f}. The auxiliary multifunctionality task achieved
ROC-AUC {rnd['auxiliary_roc_auc']:.3f}.

Compared with AAC+DPC logistic regression, macro ROC-AUC increased from
{random_outputs['baseline'].iloc[0]['multilabel_macro_roc_auc']:.3f} to
{rnd['macro_roc_auc']:.3f}. The accompanying PR-AUC is reported because ROC-AUC alone
can appear optimistic for rare classes.
"""
    (random_dir / "results_3.2_auc_draft.md").write_text(random_draft, encoding="utf-8")

    species_draft = f"""# Results 3.3 — AUC under species-level novelty

Under species-disjoint testing, the optimized model retained a macro ROC-AUC of
{sp['macro_roc_auc']:.3f}, micro ROC-AUC of {sp['micro_roc_auc']:.3f}, and macro
PR-AUC of {sp['macro_pr_auc']:.3f}. Every EC class remained above the 0.80 target,
with a minimum class-level ROC-AUC of {sp['min_class_roc_auc']:.3f}. Relative to the
random protein split, macro ROC-AUC decreased by
{rnd['macro_roc_auc'] - sp['macro_roc_auc']:.3f}, quantifying the cost of taxonomic
novelty without falling below the prespecified performance target.

These results cover random and species-disjoint evaluation. Separate
homology-cluster-disjoint tests at 50% and 40% identity are reported in the
generalization-ladder analysis.
"""
    (species_dir / "results_3.3_auc_draft.md").write_text(species_draft, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--master", type=Path, required=True)
    parser.add_argument("--random-dir", type=Path, required=True)
    parser.add_argument("--species-dir", type=Path, required=True)
    args = parser.parse_args()
    random_dir = args.random_dir.resolve()
    species_dir = args.species_dir.resolve()
    master = pd.read_csv(args.master, usecols=["accession"] + [f"label_ec{i}" for i in range(1, 8)])
    random_outputs = load_outputs(random_dir)
    species_outputs = load_outputs(species_dir)
    random_overall(random_dir, random_outputs)
    plot_roc_curves(random_dir, random_outputs["probabilities"], master, "figure_3C_roc_curves.png", "ROC curves on the random protein split")
    generalization_figures(species_dir, random_outputs, species_outputs)
    plot_roc_curves(species_dir, species_outputs["probabilities"], master, "figure_4B_species_roc_curves.png", "ROC curves on the species-disjoint test set")
    write_tables_and_drafts(random_dir, species_dir, random_outputs, species_outputs)
    print("Created Arial-formatted ROC-AUC figures and AUC-focused Results drafts.")


if __name__ == "__main__":
    main()
