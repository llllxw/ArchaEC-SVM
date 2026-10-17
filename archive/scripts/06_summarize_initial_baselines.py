#!/usr/bin/env python3
"""Create Arial-formatted figures and computed drafts for initial baselines."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plot_style import CLASS_PALETTE, MODEL_COLOR, SECONDARY_COLOR, apply_arial_style


apply_arial_style()


METRIC_LABELS = {
    "multilabel_macro_f1": "Multilabel macro-F1",
    "multilabel_micro_f1": "Multilabel micro-F1",
    "mono_macro_f1": "Monofunctional macro-F1",
    "mono_balanced_accuracy": "Balanced accuracy",
    "auxiliary_f1": "Multifunctionality F1",
}


def save_random_figures(random_dir: Path, random_metrics: pd.DataFrame, random_class: pd.DataFrame) -> None:
    row = random_metrics.iloc[0]
    metric_names = list(METRIC_LABELS)
    values = [float(row[name]) for name in metric_names]
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    bars = ax.bar(range(len(values)), values, color=CLASS_PALETTE[:len(values)])
    ax.set_xticks(range(len(values)), [METRIC_LABELS[name] for name in metric_names], rotation=25, ha="right")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1)
    ax.set_title("AAC+DPC logistic-regression baseline on the random protein split")
    ax.bar_label(bars, labels=[f"{value:.3f}" for value in values], padding=3)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(random_dir / "figure_2_baseline_overall_metrics.png", dpi=300)
    plt.close(fig)

    ordered = random_class.sort_values("ec_class")
    fig, ax = plt.subplots(figsize=(9, 5.2))
    bars = ax.bar(ordered["class_name"], ordered["f1"], color=CLASS_PALETTE)
    ax.set_ylabel("F1 score")
    ax.set_ylim(0, 1)
    ax.set_title("Per-class baseline performance on the random protein split")
    ax.tick_params(axis="x", rotation=30)
    ax.bar_label(bars, labels=[f"{value:.3f}" for value in ordered["f1"]], padding=3)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(random_dir / "figure_2_baseline_per_class_f1.png", dpi=300)
    plt.close(fig)


def save_generalization_figures(
    species_dir: Path,
    random_metrics: pd.DataFrame,
    species_metrics: pd.DataFrame,
    random_class: pd.DataFrame,
    species_class: pd.DataFrame,
) -> None:
    metric_names = list(METRIC_LABELS)
    random_values = np.array([float(random_metrics.iloc[0][name]) for name in metric_names])
    species_values = np.array([float(species_metrics.iloc[0][name]) for name in metric_names])
    positions = np.arange(len(metric_names))
    width = 0.36
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.bar(positions - width / 2, random_values, width, label="Random protein split", color=MODEL_COLOR)
    ax.bar(positions + width / 2, species_values, width, label="Species-disjoint split", color=SECONDARY_COLOR)
    ax.set_xticks(positions, [METRIC_LABELS[name] for name in metric_names], rotation=25, ha="right")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1)
    ax.set_title("Baseline performance decreases under species-level novelty")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(species_dir / "figure_3_baseline_generalization_drop.png", dpi=300)
    plt.close(fig)

    merged = random_class[["ec_class", "class_name", "f1"]].merge(
        species_class[["ec_class", "f1"]], on="ec_class", suffixes=("_random", "_species")
    ).sort_values("ec_class")
    positions = np.arange(len(merged))
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.bar(positions - width / 2, merged["f1_random"], width, label="Random protein split", color=MODEL_COLOR)
    ax.bar(positions + width / 2, merged["f1_species"], width, label="Species-disjoint split", color=SECONDARY_COLOR)
    ax.set_xticks(positions, merged["class_name"], rotation=30, ha="right")
    ax.set_ylabel("F1 score")
    ax.set_ylim(0, 1)
    ax.set_title("Class-specific effects of species-disjoint evaluation")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(species_dir / "figure_3_per_class_generalization.png", dpi=300)
    plt.close(fig)


def write_drafts(random_dir: Path, species_dir: Path, random_metrics: pd.DataFrame, species_metrics: pd.DataFrame) -> None:
    random = random_metrics.iloc[0]
    species = species_metrics.iloc[0]
    random_text = f"""# Results 3.2 preliminary draft — composition baseline

As an initial non-language-model benchmark, a class-balanced logistic-regression
model was trained on 20 amino-acid-composition and 400 dipeptide-composition
features. On the exact-sequence-grouped random test set, the model achieved a
multilabel macro-F1 of {random['multilabel_macro_f1']:.3f}, a micro-F1 of
{random['multilabel_micro_f1']:.3f}, and an exact-match ratio of
{random['multilabel_exact_match']:.3f}. Restricting evaluation to proteins carrying
one top-level EC label produced a macro-F1 of {random['mono_macro_f1']:.3f}, balanced
accuracy of {random['mono_balanced_accuracy']:.3f}, and MCC of
{random['mono_mcc']:.3f}. The auxiliary cross-class multifunctionality classifier
reached AUROC {random['auxiliary_auroc']:.3f}, AUPRC
{random['auxiliary_auprc']:.3f}, and F1 {random['auxiliary_f1']:.3f}.

These results establish a conventional sequence-composition reference point. They
do not yet support a claim of superiority because protein-language-model and
homology-based baselines remain to be evaluated.
"""
    (random_dir / "results_3.2_preliminary_draft.md").write_text(random_text, encoding="utf-8")

    macro_drop = float(random["multilabel_macro_f1"] - species["multilabel_macro_f1"])
    mono_drop = float(random["mono_macro_f1"] - species["mono_macro_f1"])
    species_text = f"""# Results 3.3 preliminary draft — species-disjoint baseline

When all proteins from each taxonomy identifier were confined to one partition,
multilabel macro-F1 decreased from {random['multilabel_macro_f1']:.3f} under random
protein splitting to {species['multilabel_macro_f1']:.3f} under species-disjoint
testing, an absolute decrease of {macro_drop:.3f}. Monofunctional macro-F1 decreased
by {mono_drop:.3f}, from {random['mono_macro_f1']:.3f} to
{species['mono_macro_f1']:.3f}, while MCC declined from {random['mono_mcc']:.3f} to
{species['mono_mcc']:.3f}. The multifunctionality auxiliary task retained AUROC
{species['auxiliary_auroc']:.3f} but achieved F1 {species['auxiliary_f1']:.3f}.

The observed decline indicates sensitivity of composition-based prediction to
taxonomic novelty. This is a preliminary two-point comparison; the full
generalization ladder requires 50%, 40%, and 30% homology-cluster splits before the
section can be finalized.
"""
    (species_dir / "results_3.3_preliminary_draft.md").write_text(species_text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--random-dir", type=Path, required=True)
    parser.add_argument("--species-dir", type=Path, required=True)
    args = parser.parse_args()
    random_dir = args.random_dir.resolve()
    species_dir = args.species_dir.resolve()
    random_metrics = pd.read_csv(random_dir / "benchmark_metrics.csv")
    species_metrics = pd.read_csv(species_dir / "benchmark_metrics.csv")
    random_class = pd.read_csv(random_dir / "per_class_metrics.csv")
    species_class = pd.read_csv(species_dir / "per_class_metrics.csv")
    save_random_figures(random_dir, random_metrics, random_class)
    save_generalization_figures(species_dir, random_metrics, species_metrics, random_class, species_class)
    write_drafts(random_dir, species_dir, random_metrics, species_metrics)
    print("Created Arial-formatted baseline figures and preliminary Results drafts.")


if __name__ == "__main__":
    main()
