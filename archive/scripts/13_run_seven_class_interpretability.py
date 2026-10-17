#!/usr/bin/env python3
"""Compare sequence signatures, pairwise AUC, TreeSHAP, and EC co-occurrence."""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import kruskal
from sklearn.metrics import roc_auc_score
from statsmodels.stats.multitest import multipletests

from plot_style import (
    CLASS_PALETTE,
    REFERENCE_DIVERGING_CMAP,
    REFERENCE_SEQUENTIAL_CMAP,
    SECONDARY_COLOR,
    apply_arial_style,
)


apply_arial_style()
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
CLASS_NAMES = [
    "Oxidoreductases", "Transferases", "Hydrolases", "Lyases",
    "Isomerases", "Ligases", "Translocases",
]
CLASS_SHORT = ["EC1", "EC2", "EC3", "EC4", "EC5", "EC6", "EC7"]
LABELS = [f"label_ec{i}" for i in range(1, 8)]
PHYS_NAMES = [
    "Log length", "Sequence entropy", "Longest hydrophobic run",
    "Hydrophobic fraction", "Charged fraction", "Polar fraction",
    "Aromatic fraction", "Small-residue fraction", "Glycine fraction",
    "Proline fraction", "Cysteine fraction", "Histidine fraction",
]
RESIDUE_GROUPS = {
    "hydrophobic": set("AVILMFWY"), "charged": set("DEKR"),
    "polar": set("STNQ"), "aromatic": set("FWY"), "small": set("AGST"),
    "glycine": {"G"}, "proline": {"P"}, "cysteine": {"C"}, "histidine": {"H"},
}
PALETTE = CLASS_PALETTE


def bh(values: list[float] | np.ndarray) -> np.ndarray:
    return multipletests(np.asarray(values, dtype=float), method="fdr_bh")[1]


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


def parse_cluster_map(path: Path) -> dict[str, str]:
    table = pd.read_csv(path, sep="\t", header=None, names=["representative", "member"])
    table["accession"] = table["member"].str.split("|").str[0]
    table["cluster40"] = table["representative"].str.split("|").str[0]
    if table["accession"].duplicated().any():
        raise ValueError("An accession occurs more than once in the 40% cluster table")
    return dict(zip(table["accession"], table["cluster40"]))


def shannon_effective_species(values: pd.Series) -> tuple[float, float]:
    probabilities = values.value_counts(normalize=True).to_numpy()
    entropy = float(-(probabilities * np.log(probabilities)).sum())
    return entropy, float(np.exp(entropy))


def class_descriptives(data: pd.DataFrame, primary_mask: np.ndarray, output: Path) -> None:
    rows, tax_rows = [], []
    for ec, name in enumerate(CLASS_NAMES, start=1):
        subset = data.loc[primary_mask & data[f"label_ec{ec}"].eq(1)]
        entropy, effective = shannon_effective_species(subset["taxonomy_id"])
        rows.append({
            "ec_class": ec, "class_name": name, "n": len(subset),
            "median_length": subset["length"].median(),
            "length_q1": subset["length"].quantile(0.25), "length_q3": subset["length"].quantile(0.75),
            "species_n": subset["taxonomy_id"].nunique(), "phylum_n": subset["phylum"].nunique(),
            "species_shannon": entropy, "effective_species": effective,
            "median_complete_ec_count": subset["n_full_ec"].median(),
        })
        counts = subset["phylum"].fillna("Unclassified").value_counts()
        for phylum, count in counts.items():
            tax_rows.append({"ec_class": ec, "class_name": name, "phylum": phylum,
                             "n": int(count), "proportion": count / len(subset)})
    pd.DataFrame(rows).to_csv(output / "class_descriptive_statistics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(tax_rows).to_csv(output / "class_taxonomic_breadth.csv", index=False, encoding="utf-8-sig")


def property_analysis(
    data: pd.DataFrame,
    physical: np.ndarray,
    primary_mask: np.ndarray,
    rng: np.random.Generator,
    permutations: int,
    output: Path,
) -> pd.DataFrame:
    indices = np.flatnonzero(primary_mask)
    values = physical[indices]
    labels = data.loc[primary_mask, LABELS].to_numpy().argmax(axis=1) + 1
    species = data.loc[primary_mask, "taxonomy_id"].astype(str).to_numpy()
    species_indices = [np.flatnonzero(species == item) for item in np.unique(species)]
    observed_h = []
    for feature_index in range(values.shape[1]):
        observed_h.append(kruskal(*[values[labels == ec, feature_index] for ec in range(1, 8)]).statistic)
    exceed = np.zeros(values.shape[1], dtype=int)
    for _ in range(permutations):
        permuted = labels.copy()
        for block in species_indices:
            permuted[block] = rng.permutation(permuted[block])
        for feature_index in range(values.shape[1]):
            statistic = kruskal(*[values[permuted == ec, feature_index] for ec in range(1, 8)]).statistic
            exceed[feature_index] += statistic >= observed_h[feature_index]
    permutation_p = (exceed + 1) / (permutations + 1)
    q_values = bh(permutation_p)
    rows = []
    for feature_index, feature in enumerate(PHYS_NAMES):
        global_median = np.median(values[:, feature_index])
        scale = (np.quantile(values[:, feature_index], 0.75) - np.quantile(values[:, feature_index], 0.25)) / 1.349
        if scale <= 1e-12:
            scale = np.std(values[:, feature_index]) or 1.0
        epsilon_squared = max(0.0, (observed_h[feature_index] - 7 + 1) / (len(values) - 7))
        for ec, name in enumerate(CLASS_NAMES, start=1):
            class_values = values[labels == ec, feature_index]
            rows.append({
                "feature": feature, "ec_class": ec, "class_name": name,
                "class_median": np.median(class_values), "class_q1": np.quantile(class_values, 0.25),
                "class_q3": np.quantile(class_values, 0.75),
                "robust_standardized_median": (np.median(class_values) - global_median) / scale,
                "global_kruskal_h": observed_h[feature_index], "epsilon_squared": epsilon_squared,
                "species_blocked_permutation_p": permutation_p[feature_index], "bh_q": q_values[feature_index],
            })
    result = pd.DataFrame(rows)
    result.to_csv(output / "class_physicochemical_effects.csv", index=False, encoding="utf-8-sig")
    return result


def pairwise_auc_analysis(
    data: pd.DataFrame,
    predictions_path: Path,
    cluster_map: dict[str, str],
    rng: np.random.Generator,
    bootstraps: int,
    output: Path,
) -> pd.DataFrame:
    predictions = pd.read_csv(predictions_path)
    columns = ["accession", "fragment", "n_top_classes"] + LABELS
    frame = predictions.merge(data[columns], on="accession", validate="one_to_one")
    frame["cluster40"] = frame["accession"].map(cluster_map)
    frame = frame.loc[frame["n_top_classes"].eq(1) & frame["fragment"].fillna("").ne("fragment")].copy()
    rows = []
    for left, right in combinations(range(1, 8), 2):
        pair = frame.loc[frame[f"label_ec{left}"].eq(1) | frame[f"label_ec{right}"].eq(1)].copy()
        truth = pair[f"label_ec{left}"].to_numpy(dtype=int)
        p_left = np.clip(pair[f"prob_ec{left}"].to_numpy(), 1e-7, 1 - 1e-7)
        p_right = np.clip(pair[f"prob_ec{right}"].to_numpy(), 1e-7, 1 - 1e-7)
        score = np.log(p_left / (1 - p_left)) - np.log(p_right / (1 - p_right))
        observed = roc_auc_score(truth, score)
        groups = pair["cluster40"].fillna(pair["accession"]).to_numpy()
        unique_groups = np.unique(groups)
        group_rows = {group: np.flatnonzero(groups == group) for group in unique_groups}
        bootstrap_values = []
        for _ in range(bootstraps):
            sampled = rng.choice(unique_groups, size=len(unique_groups), replace=True)
            sampled_rows = np.concatenate([group_rows[group] for group in sampled])
            if np.unique(truth[sampled_rows]).size == 2:
                bootstrap_values.append(roc_auc_score(truth[sampled_rows], score[sampled_rows]))
        rows.append({
            "class_a": left, "class_a_name": CLASS_NAMES[left - 1],
            "class_b": right, "class_b_name": CLASS_NAMES[right - 1],
            "n_a": int((truth == 1).sum()), "n_b": int((truth == 0).sum()),
            "pairwise_roc_auc": observed,
            "cluster_bootstrap_ci_low": np.quantile(bootstrap_values, 0.025),
            "cluster_bootstrap_ci_high": np.quantile(bootstrap_values, 0.975),
            "valid_bootstraps": len(bootstrap_values), "evaluation": "Homology-40 held-out test",
        })
    result = pd.DataFrame(rows)
    result.to_csv(output / "pairwise_class_auc.csv", index=False, encoding="utf-8-sig")
    return result


def tree_shap_analysis(
    data: pd.DataFrame,
    features: np.ndarray,
    split_path: Path,
    model_dir: Path,
    rng: np.random.Generator,
    bootstraps: int,
    output: Path,
) -> tuple[pd.DataFrame, list[str]]:
    manifest = pd.read_csv(split_path)[["accession", "split_random"]]
    split_map = manifest.set_index("accession")["split_random"]
    test = data["accession"].map(split_map).eq("test").to_numpy()
    clean_mono = data["n_top_classes"].eq(1).to_numpy() & data["fragment"].fillna("").ne("fragment").to_numpy()
    mask = test & clean_mono
    feature_names = [f"AAC_{aa}" for aa in AMINO_ACIDS]
    feature_names += [f"DPC_{left}{right}" for left in AMINO_ACIDS for right in AMINO_ACIDS]
    feature_names += [name.replace(" ", "_") for name in PHYS_NAMES]
    groups = ["AAC"] * 20 + ["DPC"] * 400 + ["Physicochemical"] * 12
    species = data.loc[mask, "taxonomy_id"].astype(str).to_numpy()
    unique_species = np.unique(species)
    species_rows = {item: np.flatnonzero(species == item) for item in unique_species}
    rows = []
    for ec, class_name in enumerate(CLASS_NAMES, start=1):
        model = joblib.load(model_dir / f"lightgbm_ec{ec}.joblib")
        contributions = model.booster_.predict(
            features[mask], pred_contrib=True, num_iteration=model.best_iteration_
        )[:, :-1]
        positive = data.loc[mask, f"label_ec{ec}"].to_numpy(dtype=bool)
        mean_abs = np.mean(np.abs(contributions), axis=0)
        positive_mean = contributions[positive].mean(axis=0)
        negative_mean = contributions[~positive].mean(axis=0)
        contrast = positive_mean - negative_mean
        bootstrap_contrast = np.full((bootstraps, len(feature_names)), np.nan, dtype=np.float32)
        for iteration in range(bootstraps):
            sampled_species = rng.choice(unique_species, size=len(unique_species), replace=True)
            sampled_rows = np.concatenate([species_rows[item] for item in sampled_species])
            sampled_positive = positive[sampled_rows]
            if sampled_positive.any() and (~sampled_positive).any():
                sampled_values = contributions[sampled_rows]
                bootstrap_contrast[iteration] = (
                    sampled_values[sampled_positive].mean(axis=0)
                    - sampled_values[~sampled_positive].mean(axis=0)
                )
        ci_low = np.nanquantile(bootstrap_contrast, 0.025, axis=0)
        ci_high = np.nanquantile(bootstrap_contrast, 0.975, axis=0)
        observed_sign = np.sign(contrast)
        direction_stability = np.nanmean(np.sign(bootstrap_contrast) == observed_sign, axis=0)
        for index, feature in enumerate(feature_names):
            rows.append({
                "ec_class": ec, "class_name": class_name, "feature": feature,
                "feature_group": groups[index], "mean_absolute_shap": mean_abs[index],
                "mean_shap_positive": positive_mean[index], "mean_shap_negative": negative_mean[index],
                "positive_negative_contrast": contrast[index],
                "contrast_ci_low": ci_low[index], "contrast_ci_high": ci_high[index],
                "direction_stability": direction_stability[index],
            })
    result = pd.DataFrame(rows)
    result.to_csv(output / "class_specific_shap.csv", index=False, encoding="utf-8-sig")
    selected = []
    for ec in range(1, 8):
        candidates = result.loc[result.ec_class.eq(ec) & result.direction_stability.ge(0.80)]
        if len(candidates) < 3:
            candidates = result.loc[result.ec_class.eq(ec)]
        top = candidates.nlargest(3, "mean_absolute_shap")["feature"].tolist()
        selected.extend(top)
    return result, list(dict.fromkeys(selected))


def cooccurrence_analysis(
    data: pd.DataFrame,
    rng: np.random.Generator,
    permutations: int,
    output: Path,
) -> pd.DataFrame:
    clean = data["fragment"].fillna("").ne("fragment")
    multi = data.loc[clean & data["n_top_classes"].gt(1), ["taxonomy_id"] + LABELS].copy()
    labels = multi[LABELS].to_numpy(dtype=int)
    species = multi["taxonomy_id"].astype(str).to_numpy()
    blocks = [np.flatnonzero(species == item) for item in np.unique(species)]
    pairs = list(combinations(range(7), 2))
    observed = np.array([(labels[:, left] & labels[:, right]).sum() for left, right in pairs])
    null = np.zeros((permutations, len(pairs)), dtype=int)
    for iteration in range(permutations):
        permuted = labels.copy()
        for column in range(7):
            for block in blocks:
                permuted[block, column] = rng.permutation(permuted[block, column])
        null[iteration] = [(permuted[:, left] & permuted[:, right]).sum() for left, right in pairs]
    p_values = (1 + (null >= observed).sum(axis=0)) / (permutations + 1)
    q_values = bh(p_values)
    rows = []
    for index, (left, right) in enumerate(pairs):
        a, b = labels[:, left].astype(bool), labels[:, right].astype(bool)
        both = int((a & b).sum())
        union = int((a | b).sum())
        neither = int((~a & ~b).sum())
        a_only, b_only = int((a & ~b).sum()), int((~a & b).sum())
        odds = ((both + 0.5) * (neither + 0.5)) / ((a_only + 0.5) * (b_only + 0.5))
        null_mean, null_sd = null[:, index].mean(), null[:, index].std(ddof=1)
        rows.append({
            "class_a": left + 1, "class_a_name": CLASS_NAMES[left],
            "class_b": right + 1, "class_b_name": CLASS_NAMES[right],
            "observed_cooccurrence": both, "jaccard": both / union if union else 0,
            "haldane_odds_ratio": odds, "null_mean": null_mean,
            "enrichment_z": (both - null_mean) / null_sd if null_sd > 0 else np.nan,
            "species_blocked_permutation_p": p_values[index], "bh_q": q_values[index],
            "multifunctional_n": len(multi),
        })
    result = pd.DataFrame(rows)
    result.to_csv(output / "multifunction_cooccurrence.csv", index=False, encoding="utf-8-sig")
    return result


def representative_proteins(data: pd.DataFrame, predictions_path: Path, output: Path) -> None:
    predictions = pd.read_csv(predictions_path)
    frame = predictions.merge(data[["accession", "protein_names", "organism", "fragment", "n_top_classes"] + LABELS],
                              on="accession", validate="one_to_one")
    frame = frame.loc[frame.n_top_classes.eq(1) & frame.fragment.fillna("").ne("fragment")].copy()
    rows = []
    for ec, name in enumerate(CLASS_NAMES, start=1):
        positives = frame.loc[frame[f"label_ec{ec}"].eq(1)].sort_values(f"prob_ec{ec}", ascending=False)
        for rank, (_, item) in enumerate(positives.head(3).iterrows(), start=1):
            rows.append({"ec_class": ec, "class_name": name, "selection": "High-confidence true positive",
                         "rank": rank, "accession": item.accession, "protein_names": item.protein_names,
                         "organism": item.organism, "class_probability": item[f"prob_ec{ec}"]})
        if len(positives):
            item = positives.iloc[-1]
            rows.append({"ec_class": ec, "class_name": name, "selection": "Lowest-confidence true protein",
                         "rank": 1, "accession": item.accession, "protein_names": item.protein_names,
                         "organism": item.organism, "class_probability": item[f"prob_ec{ec}"]})
    pd.DataFrame(rows).to_csv(output / "representative_class_proteins.csv", index=False, encoding="utf-8-sig")


def save_figure(fig: plt.Figure, output: Path, stem: str) -> None:
    fig.tight_layout()
    fig.savefig(output / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(output / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_properties(effects: pd.DataFrame, output: Path) -> None:
    matrix = effects.pivot(index="class_name", columns="feature", values="robust_standardized_median").loc[CLASS_NAMES, PHYS_NAMES]
    fig, ax = plt.subplots(figsize=(12, 5.2))
    sns.heatmap(matrix, cmap=REFERENCE_DIVERGING_CMAP, center=0, linewidths=0.7, linecolor="white",
                cbar_kws={"label": "Robust standardized median"}, ax=ax)
    ax.set(xlabel="", ylabel="", title="Sequence-property differences among seven archaeal enzyme classes")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=35, ha="right")
    save_figure(fig, output, "figure_6A_class_property_heatmap")


def plot_pairwise_auc(pairwise: pd.DataFrame, output: Path) -> None:
    matrix = np.ones((7, 7))
    for row in pairwise.itertuples():
        matrix[row.class_a - 1, row.class_b - 1] = row.pairwise_roc_auc
        matrix[row.class_b - 1, row.class_a - 1] = row.pairwise_roc_auc
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(matrix, annot=True, fmt=".3f", cmap=REFERENCE_SEQUENTIAL_CMAP, vmin=0.0, vmax=1.0,
                xticklabels=CLASS_SHORT, yticklabels=CLASS_SHORT, linewidths=1, linecolor="white",
                cbar_kws={"label": "Pairwise ROC-AUC"}, ax=ax)
    ax.set(xlabel="Enzyme class", ylabel="Enzyme class", title="Pairwise class discrimination on the Homology-40 test set")
    ax.tick_params(axis='y',labelrotation=0)
    save_figure(fig, output, "figure_6B_pairwise_auc_heatmap")


def plot_shap(shap_table: pd.DataFrame, selected: list[str], output: Path) -> None:
    matrix = shap_table.loc[shap_table.feature.isin(selected)].pivot(
        index="class_name", columns="feature", values="positive_negative_contrast"
    ).loc[CLASS_NAMES, selected]
    values = matrix.to_numpy()
    denom = np.max(np.abs(values), axis=1, keepdims=True)
    normalized = values / np.where(denom == 0, 1, denom)
    fig, ax = plt.subplots(figsize=(max(10, 0.55 * len(selected)), 5.2))
    sns.heatmap(normalized, cmap=REFERENCE_DIVERGING_CMAP, center=0, vmin=-1, vmax=1,
                xticklabels=selected, yticklabels=CLASS_SHORT, linewidths=0.5, linecolor="white",
                cbar_kws={"label": "Within-class normalized SHAP contrast"}, ax=ax)
    ax.set(xlabel="Predictive feature", ylabel="", title="Class-specific LightGBM TreeSHAP signatures")
    ax.tick_params(axis='y',labelrotation=0)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    save_figure(fig, output, "figure_6C_class_shap_heatmap")


def plot_network(cooccurrence: pd.DataFrame, data: pd.DataFrame, output: Path) -> None:
    graph = nx.Graph()
    multi = data.loc[data.n_top_classes.gt(1)]
    supports = [int(multi[label].sum()) for label in LABELS]
    for index in range(7):
        graph.add_node(index + 1, support=supports[index])
    candidates = cooccurrence.loc[cooccurrence.observed_cooccurrence.gt(0)].copy()
    significant = candidates.loc[candidates.bh_q.lt(0.10) & candidates.enrichment_z.gt(0)]
    shown = significant if len(significant) else candidates.nlargest(8, "observed_cooccurrence")
    for row in shown.itertuples():
        graph.add_edge(row.class_a, row.class_b, weight=row.observed_cooccurrence,
                       significant=row.bh_q < 0.10, z=row.enrichment_z)
    position = nx.circular_layout(graph)
    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    node_sizes = [450 + 22 * graph.nodes[node]["support"] for node in graph.nodes]
    nx.draw_networkx_nodes(graph, position, node_size=node_sizes, node_color=PALETTE,
                           edgecolors="white", linewidths=1.5, ax=ax)
    edges = list(graph.edges(data=True))
    widths = [1 + 0.12 * item[2]["weight"] for item in edges]
    colors = [SECONDARY_COLOR if item[2]["significant"] else "#999999" for item in edges]
    nx.draw_networkx_edges(graph, position, width=widths, edge_color=colors, alpha=0.8, ax=ax)
    nx.draw_networkx_labels(graph, position, labels={i: CLASS_SHORT[i - 1] for i in range(1, 8)},
                            font_family="Arial", font_size=10, ax=ax)
    edge_labels = {(a, b): str(attributes["weight"]) for a, b, attributes in edges}
    nx.draw_networkx_edge_labels(graph, position, edge_labels=edge_labels, font_family="Arial", font_size=8, ax=ax)
    ax.set_title("Top-level EC co-occurrence in multifunctional archaeal proteins")
    note = (
        "Edge labels: observed proteins; wine red: positive enrichment at BH q < 0.10"
        if len(significant)
        else "Eight most frequent descriptive edges; no positive enrichment at BH q < 0.10"
    )
    ax.text(0.5, -0.04, note,
            transform=ax.transAxes, ha="center", fontsize=8)
    ax.axis("off")
    save_figure(fig, output, "figure_6D_multifunction_network")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--master", type=Path, required=True)
    parser.add_argument("--feature-cache", type=Path, required=True)
    parser.add_argument("--random-split", type=Path, required=True)
    parser.add_argument("--random-model-dir", type=Path, required=True)
    parser.add_argument("--homology40-predictions", type=Path, required=True)
    parser.add_argument("--cluster40", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstraps", type=int, default=1000)
    parser.add_argument("--permutations", type=int, default=1000)
    args = parser.parse_args()

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    data = pd.read_csv(args.master, dtype={"taxonomy_id": str})
    accessions = args.feature_cache.with_suffix(".accessions.txt").read_text(encoding="utf-8").splitlines()
    if accessions != data.accession.tolist():
        raise ValueError("Feature cache order does not match master dataset")
    composition = np.load(args.feature_cache)
    physical = physicochemical_features(data.sequence)
    features = np.hstack([composition, physical])
    nonfragment = data.fragment.fillna("").ne("fragment").to_numpy()
    monofunctional = data.n_top_classes.eq(1).to_numpy()
    primary_mask = nonfragment & monofunctional
    cluster_map = parse_cluster_map(args.cluster40)

    class_descriptives(data, primary_mask, output)
    effects = property_analysis(data, physical, primary_mask, rng, args.permutations, output)
    pairwise = pairwise_auc_analysis(data, args.homology40_predictions, cluster_map, rng, args.bootstraps, output)
    shap_table, selected = tree_shap_analysis(
        data, features, args.random_split, args.random_model_dir, rng, args.bootstraps, output
    )
    cooccurrence = cooccurrence_analysis(data, rng, args.permutations, output)
    representative_proteins(data, args.homology40_predictions, output)

    plot_properties(effects, output)
    plot_pairwise_auc(pairwise, output)
    plot_shap(shap_table, selected, output)
    plot_network(cooccurrence, data, output)

    class_stats = pd.read_csv(output / "class_descriptive_statistics.csv")
    strongest_property = effects.drop_duplicates("feature").sort_values("epsilon_squared", ascending=False).iloc[0]
    easiest = pairwise.sort_values("pairwise_roc_auc", ascending=False).iloc[0]
    hardest = pairwise.sort_values("pairwise_roc_auc").iloc[0]
    significant_edges = cooccurrence.loc[cooccurrence.bh_q.lt(0.10) & cooccurrence.enrichment_z.gt(0)]
    top_edge = cooccurrence.sort_values("enrichment_z", ascending=False).iloc[0]
    group_shap = shap_table.groupby(["ec_class", "class_name", "feature_group"], as_index=False)["mean_absolute_shap"].sum()
    group_shap["group_share"] = group_shap["mean_absolute_shap"] / group_shap.groupby("ec_class")["mean_absolute_shap"].transform("sum")
    group_shap.to_csv(output / "class_shap_group_summary.csv", index=False, encoding="utf-8-sig")

    report = {
        "status": "completed", "seed": args.seed, "bootstraps": args.bootstraps,
        "permutations": args.permutations, "primary_monofunctional_nonfragment_n": int(primary_mask.sum()),
        "multifunctional_nonfragment_n": int((nonfragment & data.n_top_classes.gt(1).to_numpy()).sum()),
        "class_support": dict(zip(CLASS_SHORT, class_stats.n.astype(int))),
        "strongest_property": {"feature": strongest_property.feature, "epsilon_squared": strongest_property.epsilon_squared,
                               "bh_q": strongest_property.bh_q},
        "easiest_pair": {"classes": f"EC{int(easiest.class_a)}-EC{int(easiest.class_b)}",
                         "roc_auc": easiest.pairwise_roc_auc, "ci_low": easiest.cluster_bootstrap_ci_low,
                         "ci_high": easiest.cluster_bootstrap_ci_high},
        "hardest_pair": {"classes": f"EC{int(hardest.class_a)}-EC{int(hardest.class_b)}",
                         "roc_auc": hardest.pairwise_roc_auc, "ci_low": hardest.cluster_bootstrap_ci_low,
                         "ci_high": hardest.cluster_bootstrap_ci_high},
        "significantly_enriched_cooccurrence_edges_bh_q_0_10": int(len(significant_edges)),
        "top_cooccurrence_edge": {"classes": f"EC{int(top_edge.class_a)}-EC{int(top_edge.class_b)}",
                                  "observed": int(top_edge.observed_cooccurrence),
                                  "enrichment_z": top_edge.enrichment_z, "bh_q": top_edge.bh_q},
        "stable_shap_features_direction_ge_0_80": int(shap_table.direction_stability.ge(0.80).sum()),
        "interpretation_boundary": "Predictive association, not biochemical mechanism or evolutionary causation",
    }
    (output / "seven_class_interpretability_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    edge_sentence = (
        f"{len(significant_edges)} class pairs showed positive enrichment at BH q < 0.10."
        if len(significant_edges)
        else "No class pair showed positive enrichment at BH q < 0.10; the network therefore displays the strongest descriptive edges."
    )
    draft = f"""# Results 3.5 - Comparative interpretation of the seven enzyme classes

After excluding 33 fragment-flagged entries and separating multifunctional proteins,
the primary comparison contained {int(primary_mask.sum()):,} monofunctional archaeal enzymes.
Class support ranged from {int(class_stats.n.min()):,} to {int(class_stats.n.max()):,} proteins and
spanned {int(class_stats.species_n.min())}-{int(class_stats.species_n.max())} species per class.
Among the evaluated sequence properties, {strongest_property.feature} showed the largest
global class effect (epsilon-squared = {strongest_property.epsilon_squared:.3f};
species-blocked permutation BH q = {strongest_property.bh_q:.3g}).

On the Homology-40 held-out test set, pairwise discrimination varied substantially
among the 21 class pairs. The most separable pair was EC{int(easiest.class_a)} versus
EC{int(easiest.class_b)} (ROC-AUC = {easiest.pairwise_roc_auc:.3f}, 95% cluster-bootstrap CI
{easiest.cluster_bootstrap_ci_low:.3f}-{easiest.cluster_bootstrap_ci_high:.3f}), whereas the most overlapping pair was
EC{int(hardest.class_a)} versus EC{int(hardest.class_b)} (ROC-AUC = {hardest.pairwise_roc_auc:.3f}, 95% CI
{hardest.cluster_bootstrap_ci_low:.3f}-{hardest.cluster_bootstrap_ci_high:.3f}). This pairwise analysis
localizes the generalization difficulty more precisely than a single macro-average.

Native LightGBM TreeSHAP values revealed distinct class-specific combinations of
amino-acid composition, dipeptide composition, and physicochemical features.
Because these values quantify contributions to model output, they are interpreted
as predictive signatures rather than catalytic residues or mechanistic determinants.

The secondary analysis contained {int((nonfragment & data.n_top_classes.gt(1).to_numpy()).sum())} non-fragment
multifunctional proteins. {edge_sentence} The strongest descriptive association was
EC{int(top_edge.class_a)}-EC{int(top_edge.class_b)} ({int(top_edge.observed_cooccurrence)} observed proteins;
enrichment z = {top_edge.enrichment_z:.2f}; BH q = {top_edge.bh_q:.3g}). Co-occurrence
indicates shared annotation within a protein and does not establish pathway coupling.
"""
    (output / "results_3.5_draft.md").write_text(draft, encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
