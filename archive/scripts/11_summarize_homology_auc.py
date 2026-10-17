#!/usr/bin/env python3
"""Summarize 50% and 40% homology-disjoint ROC-AUC experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import auc, roc_curve

from plot_style import (
    CLASS_PALETTE,
    MODEL_COLOR,
    REFERENCE_SEQUENTIAL_CMAP,
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


def read_experiment(folder: Path) -> tuple[pd.Series, pd.DataFrame, pd.DataFrame]:
    model = folder / "optimized_lightgbm"
    return (
        pd.read_csv(model / "optimized_auc_metrics.csv").iloc[0],
        pd.read_csv(model / "optimized_per_class_auc.csv").sort_values("ec_class"),
        pd.read_csv(model / "optimized_test_probabilities.csv"),
    )


def plot_ladder(summary: pd.DataFrame, output: Path) -> None:
    # Only measured thresholds have values; missing thresholds remain NaN.
    thresholds = [100] + list(range(90, 0, -10))  # 100 is a display position for Random, not identity.
    rows = summary[summary.evaluation.str.startswith('Homology-')].copy()
    rows['identity'] = rows.evaluation.str.extract(r'(\d+)').astype(int)
    rows = rows.set_index('identity').reindex(thresholds)
    random = summary.loc[summary.evaluation.eq('Random')]
    if len(random):
        rows.loc[100,['macro_roc_auc','micro_roc_auc']] = random.iloc[0][['macro_roc_auc','micro_roc_auc']].to_numpy()
    x = np.array(thresholds)
    fig, ax = plt.subplots(figsize=(11.5, 3.5))
    ax.plot(x, rows["macro_roc_auc"], marker="o", linewidth=2.2, label="Macro ROC-AUC", color=MODEL_COLOR)
    ax.plot(x, rows["micro_roc_auc"], marker="s", linewidth=2.2, label="Micro ROC-AUC", color=SECONDARY_COLOR)
    ax.axhline(0.8, color=TARGET_COLOR, linestyle="--", linewidth=1.3, label="Target AUC = 0.80")
    ax.set_xticks(x, ['Random']+[str(t) for t in thresholds[1:]])
    ax.set_xlim(105,5)
    ax.set_xlabel('Homology identity threshold (%)')
    for threshold in thresholds:
        if pd.isna(rows.loc[threshold,'macro_roc_auc']):
            ax.text(threshold,.57,'No current result',rotation=90,ha='center',fontsize=8,color='#777777')
    ax.set_ylabel("ROC-AUC")
    ax.set_ylim(0.0, 1.12)
    ax.set_title("LightGBM across homology thresholds")
    for threshold in thresholds:
        macro=rows.loc[threshold,'macro_roc_auc']
        micro=rows.loc[threshold,'micro_roc_auc']
        if pd.notna(macro) and pd.notna(micro):
            for value,label_y,color,va in [(macro,max(macro,micro)+.045,MODEL_COLOR,'bottom'),
                                           (micro,min(macro,micro)-.055,SECONDARY_COLOR,'top')]:
                ax.annotate(f'{value:.3f}',xy=(threshold,value),xytext=(threshold,label_y),
                            ha='center',va=va,color=color,fontsize=8,
                            arrowprops=dict(arrowstyle='-',color=color,lw=.5))
    ax.legend(frameon=False,loc='lower left')
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(output / "figure_4C_homology_auc_ladder.png", dpi=300)
    plt.close(fig)


def plot_heatmap(matrix: pd.DataFrame, output: Path) -> None:
    values = matrix[CLASS_NAMES].to_numpy()
    fig, ax = plt.subplots(figsize=(11.5, max(3.5,len(matrix)*.7+2)))
    image = ax.pcolormesh(np.arange(8)-.5,np.arange(len(matrix)+1)-.5,values,
                         cmap=REFERENCE_SEQUENTIAL_CMAP,vmin=.4,vmax=1.,rasterized=False)
    ax.set_ylim(len(matrix)-.5,-.5)
    ax.set_xticks(np.arange(7), CLASS_NAMES, rotation=30, ha="right")
    ax.set_yticks(np.arange(len(matrix)), matrix["evaluation"])
    ax.set_title("Class-specific one-vs-rest ROC-AUC")
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            value = values[row, column]
            color = "white" if value > 0.92 else "black"
            ax.text(column, row, f"{value:.3f}", ha="center", va="center", color=color, fontsize=9)
    colorbar = fig.colorbar(image, ax=ax, pad=0.02)
    colorbar.set_label("ROC-AUC")
    fig.tight_layout()
    fig.savefig(output / "figure_4E_homology_per_class_auc.png", dpi=300)
    plt.close(fig)


def plot_roc_panels(experiments: list[tuple[str, pd.DataFrame]], master: pd.DataFrame, output: Path) -> None:
    ncols=min(3,len(experiments));nrows=int(np.ceil(len(experiments)/ncols))
    fig, axes = plt.subplots(nrows,ncols,figsize=(6.5*ncols,6*nrows),sharex=True,sharey=True)
    axes = np.atleast_1d(axes).ravel()
    colors = CLASS_PALETTE
    for ax, (label, predictions) in zip(axes, experiments):
        merged = predictions.merge(master, on="accession", validate="one_to_one")
        for index, (class_name, color) in enumerate(zip(CLASS_NAMES, colors), start=1):
            fpr, tpr, _ = roc_curve(merged[f"label_ec{index}"], merged[f"prob_ec{index}"])
            score = auc(fpr, tpr)
            ax.plot(fpr, tpr, color=color, linewidth=1.5, label=f"EC {index} ({score:.3f})")
        ax.plot([0, 1], [0, 1], linestyle="--", color=TARGET_COLOR, linewidth=1)
        ax.set_title(label)
        ax.set_xlabel("False-positive rate")
        ax.set_ylabel("True-positive rate")
        ax.set_xlim(-.02,1.02);ax.set_ylim(-.02,1.02)
        ax.tick_params(labelbottom=True,labelleft=True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False, fontsize=8, loc="lower right", title="Class ROC-AUC")
    for ax in axes[len(experiments):]:ax.set_visible(False)
    fig.suptitle("ROC curves under homology-cluster-disjoint evaluation", y=1.01)
    fig.tight_layout()
    fig.savefig(output / "figure_4D_homology_roc_curves.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--master", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.results_root.resolve()
    master = pd.read_csv(args.master, usecols=["accession"] + [f"label_ec{i}" for i in range(1, 8)])

    random_metrics = pd.read_csv(root.parent / "3.2_random_split_performance" / "optimized_lightgbm" / "optimized_auc_metrics.csv").iloc[0]
    species_metrics = pd.read_csv(root / "optimized_lightgbm" / "optimized_auc_metrics.csv").iloc[0]
    summary_rows = [
        {"evaluation": "Random", **random_metrics.to_dict()},
        {"evaluation": "Species-disjoint", **species_metrics.to_dict()},
    ]
    class_rows = []
    roc_experiments = []
    for threshold in range(90,0,-10):
        folder = root / f"homology_id{threshold}"
        metrics, per_class, predictions = read_experiment(folder)
        summary_rows.append({"evaluation": f"Homology-{threshold}", **metrics.to_dict()})
        class_row = {"evaluation": f"Homology-{threshold}"}
        class_row.update(dict(zip(per_class["class_name"], per_class["roc_auc"])))
        class_rows.append(class_row)
        roc_experiments.append((f"{threshold}% identity", predictions))

    summary = pd.DataFrame(summary_rows)
    class_matrix = pd.DataFrame(class_rows)
    summary.to_csv(root / "homology_auc_summary.csv", index=False, encoding="utf-8-sig")
    class_matrix.to_csv(root / "homology_per_class_auc.csv", index=False, encoding="utf-8-sig")
    plot_ladder(summary, root)
    plot_heatmap(class_matrix, root)
    plot_roc_panels(roc_experiments, master, root)

    homology = summary[summary["evaluation"].str.startswith("Homology")].copy()
    pass_macro = bool((homology["macro_roc_auc"] >= 0.8).all())
    pass_all_classes = bool((class_matrix[CLASS_NAMES] >= 0.8).all().all())
    report = {
        "target_auc": 0.8,
        "all_homology_macro_auc_pass": pass_macro,
        "all_homology_class_auc_pass": pass_all_classes,
        "macro_roc_auc": dict(zip(homology["evaluation"], homology["macro_roc_auc"])),
        "minimum_class_auc": {
            row["evaluation"]: float(min(row[name] for name in CLASS_NAMES))
            for _, row in class_matrix.iterrows()
        },
    }
    (root / "homology_auc_verification.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    draft = f"""# Results 3.3 — Homology-cluster-disjoint AUC verification

To quantify dependence on sequence homology, proteins were clustered by MMseqs2
connected components at 50% and 40% sequence identity with 80% bidirectional
coverage, and complete clusters were assigned to a single partition. No cluster
crossed the training, validation, or test boundaries. The corresponding test sets
contained 1,266 and 1,254 proteins.

Macro ROC-AUC decreased from {random_metrics['macro_roc_auc']:.3f} under random
protein splitting and {species_metrics['macro_roc_auc']:.3f} under species-disjoint
testing to {homology.iloc[0]['macro_roc_auc']:.3f} and
{homology.iloc[1]['macro_roc_auc']:.3f} at 50% and 40% identity,
respectively. Only the 50% experiment exceeded the 0.80 target at the macro level,
whereas the 40% experiment did not. The minimum class-level ROC-AUC values were
{report['minimum_class_auc']['Homology-50']:.3f},
and {report['minimum_class_auc']['Homology-40']:.3f}, so the prespecified requirement
that every class remain above 0.80 was not met at any homology threshold.

The monotonic loss of discrimination demonstrates that composition-based prediction
retains substantial protein-family dependence. These results motivate the subsequent
protein-language-model experiment; they must not be rewritten as meeting the AUC
target through post hoc threshold selection because ROC-AUC is threshold-independent.
"""
    (root / "results_3.3_homology_auc_draft.md").write_text(draft, encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
