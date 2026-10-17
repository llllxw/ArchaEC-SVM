#!/usr/bin/env python3
"""Prepare the current Swiss-Prot archaeal EC dataset for modelling.

Creates one row per accession, preserves full EC annotations, derives top-level
multi-label targets, compares the current release with the audited legacy data,
and updates Results 3.1 with current-release tables and figures.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from plot_style import CLASS_PALETTE, SECONDARY_COLOR, apply_arial_style


apply_arial_style()


EC_CLASS_NAMES = {
    1: "Oxidoreductases",
    2: "Transferases",
    3: "Hydrolases",
    4: "Lyases",
    5: "Isomerases",
    6: "Ligases",
    7: "Translocases",
}


def find_single(path_or_pattern: str) -> Path:
    candidate = Path(path_or_pattern)
    if candidate.exists():
        return candidate.resolve()
    matches = sorted(Path().glob(path_or_pattern))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one input for {path_or_pattern!r}, found {len(matches)}")
    return matches[0].resolve()


def parse_ec_numbers(value: object) -> list[str]:
    if pd.isna(value):
        return []
    return sorted({item.strip() for item in str(value).split(";") if item.strip()})


def top_classes(ec_numbers: list[str]) -> list[int]:
    classes = set()
    for ec in ec_numbers:
        match = re.match(r"^([1-7])(?:\.|$)", ec)
        if match:
            classes.add(int(match.group(1)))
    return sorted(classes)


def extract_rank(lineage: object, rank: str) -> str | None:
    if pd.isna(lineage):
        return None
    for item in str(lineage).split(","):
        item = item.strip()
        if item.endswith(f"({rank})"):
            return item[: -len(f"({rank})")].strip()
    return None


def prepare(raw: pd.DataFrame) -> pd.DataFrame:
    renamed = raw.rename(
        columns={
            "Entry": "accession",
            "Entry Name": "entry_name",
            "Protein names": "protein_names",
            "Organism": "organism",
            "Organism (ID)": "taxonomy_id",
            "Taxonomic lineage": "lineage",
            "Length": "length",
            "EC number": "ec_numbers_raw",
            "Protein existence": "protein_existence",
            "Fragment": "fragment",
            "Date of creation": "date_created",
            "Date of last modification": "date_modified",
            "Sequence": "sequence",
            "Active site": "active_site",
            "Binding site": "binding_site",
            "Transmembrane": "transmembrane",
        }
    )
    required = {"accession", "sequence", "ec_numbers_raw", "taxonomy_id", "lineage"}
    missing = required - set(renamed.columns)
    if missing:
        raise ValueError(f"Missing UniProt columns: {sorted(missing)}")

    renamed["ec_number_list"] = renamed["ec_numbers_raw"].map(parse_ec_numbers)
    renamed["top_class_list"] = renamed["ec_number_list"].map(top_classes)
    renamed = renamed[renamed["top_class_list"].map(bool)].copy()
    renamed["ec_numbers"] = renamed["ec_number_list"].map(lambda values: ";".join(values))
    renamed["top_classes"] = renamed["top_class_list"].map(lambda values: ";".join(map(str, values)))
    renamed["top_class_names"] = renamed["top_class_list"].map(
        lambda values: ";".join(EC_CLASS_NAMES[value] for value in values)
    )
    renamed["n_full_ec"] = renamed["ec_number_list"].map(len)
    renamed["n_top_classes"] = renamed["top_class_list"].map(len)
    renamed["is_multifunctional_full_ec"] = renamed["n_full_ec"] > 1
    renamed["is_multifunctional_top_level"] = renamed["n_top_classes"] > 1
    renamed["length"] = pd.to_numeric(renamed["length"], errors="raise").astype(int)
    renamed["taxonomy_id"] = renamed["taxonomy_id"].astype(str)
    renamed["phylum"] = renamed["lineage"].map(lambda value: extract_rank(value, "phylum"))
    renamed["class_taxonomy"] = renamed["lineage"].map(lambda value: extract_rank(value, "class"))
    renamed["genus"] = renamed["lineage"].map(lambda value: extract_rank(value, "genus"))
    renamed["sequence"] = renamed["sequence"].str.upper()
    renamed["sequence_sha256"] = renamed["sequence"].map(
        lambda seq: hashlib.sha256(seq.encode("ascii")).hexdigest()
    )
    for ec_class in range(1, 8):
        renamed[f"label_ec{ec_class}"] = renamed["top_class_list"].map(lambda values, x=ec_class: int(x in values))

    duplicate_accessions = renamed["accession"].duplicated(keep=False)
    if duplicate_accessions.any():
        raise ValueError(f"Current UniProt table contains duplicated accessions: {duplicate_accessions.sum()}")
    return renamed.drop(columns=["ec_number_list", "top_class_list"])


def class_summary(master: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for ec_class, class_name in EC_CLASS_NAMES.items():
        group = master[master[f"label_ec{ec_class}"] == 1]
        rows.append(
            {
                "ec_class": ec_class,
                "class_name": class_name,
                "n_accessions": len(group),
                "n_unique_sequences": group["sequence_sha256"].nunique(),
                "n_species": group["taxonomy_id"].nunique(),
                "n_phyla": group["phylum"].nunique(),
                "median_length": float(group["length"].median()),
                "mean_length": float(group["length"].mean()),
                "min_length": int(group["length"].min()),
                "max_length": int(group["length"].max()),
            }
        )
    return pd.DataFrame(rows)


def compare_legacy(master: pd.DataFrame, legacy: pd.DataFrame) -> pd.DataFrame:
    current = master.set_index("accession")
    rows = []
    for record in legacy.itertuples(index=False):
        accession = record.accession
        if accession not in current.index:
            rows.append({"accession": accession, "status": "absent_from_current_query", "legacy_labels": record.ec_classes, "current_labels": ""})
            continue
        current_labels = current.loc[accession, "top_classes"]
        rows.append(
            {
                "accession": accession,
                "status": "unchanged" if str(record.ec_classes) == str(current_labels) else "top_level_label_changed",
                "legacy_labels": record.ec_classes,
                "current_labels": current_labels,
            }
        )
    current_only = sorted(set(master["accession"]) - set(legacy["accession"]))
    rows.extend(
        {
            "accession": accession,
            "status": "new_relative_to_legacy_subset",
            "legacy_labels": "",
            "current_labels": current.loc[accession, "top_classes"],
        }
        for accession in current_only
    )
    return pd.DataFrame(rows)


def figures(master: pd.DataFrame, summary: pd.DataFrame, output: Path) -> None:
    ordered = summary.sort_values("n_accessions", ascending=True)
    fig, ax = plt.subplots(figsize=(9, 5.2))
    class_color = dict(zip(EC_CLASS_NAMES.values(), CLASS_PALETTE))
    colors = [class_color[name] for name in ordered["class_name"]]
    ax.barh(ordered["class_name"], ordered["n_accessions"], color=colors, edgecolor="#666666", linewidth=0.7)
    for y, value in enumerate(ordered["n_accessions"]):
        ax.text(value + summary["n_accessions"].max() * 0.012, y, f"{value:,}", va="center")
    ax.set_xlabel("Reviewed archaeal accessions")
    ax.set_title("Top-level EC membership in UniProtKB/Swiss-Prot release 2026_02")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(output / "figure_2A_class_distribution.png", dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    class_names = list(EC_CLASS_NAMES.values())
    length_data = [master.loc[master[f"label_ec{index}"].eq(1), "length"] for index in range(1, 8)]
    boxes = ax.boxplot(length_data, tick_labels=class_names, showfliers=False, patch_artist=True)
    for patch, color in zip(boxes["boxes"], CLASS_PALETTE):
        patch.set_facecolor(color)
        patch.set_edgecolor("#666666")
        patch.set_linewidth(0.8)
    for median in boxes["medians"]:
        median.set_color("black")
        median.set_linewidth(1.3)
    for item in boxes["whiskers"] + boxes["caps"]:
        item.set_color("#666666")
        item.set_linewidth(0.8)
    ax.set_ylabel("Protein length (amino acids)")
    ax.set_title("Sequence-length distributions in the current Swiss-Prot dataset")
    ax.tick_params(axis="x", rotation=35)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(output / "figure_2B_length_distribution.png", dpi=300)
    plt.close(fig)

    phyla = master["phylum"].fillna("Unresolved").value_counts().head(15).sort_values()
    fig, ax = plt.subplots(figsize=(9, 6))
    phylum_colors = [CLASS_PALETTE[index % len(CLASS_PALETTE)] for index in range(len(phyla))]
    ax.barh(phyla.index, phyla.values, color=phylum_colors, edgecolor="#666666", linewidth=0.6)
    for y, value in enumerate(phyla.values):
        ax.text(value + max(phyla.values) * 0.012, y, f"{value:,}", va="center")
    ax.set_xlabel("Unique accessions")
    ax.set_title("Major archaeal phyla represented in the current dataset")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(output / "figure_2C_taxonomic_distribution.png", dpi=300)
    plt.close(fig)

    combinations = master["top_class_names"].value_counts().reset_index()
    combinations.columns = ["class_combination", "n_accessions"]
    multi = combinations[combinations["class_combination"].str.contains(";", regex=False)].head(15)
    fig, ax = plt.subplots(figsize=(10, max(4.5, 0.42 * len(multi) + 1.5)))
    if len(multi):
        multi_plot = multi.sort_values("n_accessions", ascending=True)
        bar_colors = [CLASS_PALETTE[index % len(CLASS_PALETTE)] for index in range(len(multi_plot))]
        ax.barh(multi_plot["class_combination"], multi_plot["n_accessions"], color=bar_colors,
                edgecolor="#666666", linewidth=0.6)
        for y, value in enumerate(multi_plot["n_accessions"]):
            ax.text(value + max(multi_plot["n_accessions"]) * 0.012, y, str(value), va="center")
    else:
        ax.text(0.5, 0.5, "No cross-class accessions", ha="center", va="center")
    ax.set_xlabel("Reviewed archaeal accessions")
    ax.set_ylabel("")
    ax.set_title("Cross-class multifunctional label combinations in the current dataset")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(output / "figure_2D_label_combinations.png", dpi=300)
    plt.close(fig)


def updated_draft(master: pd.DataFrame, summary: pd.DataFrame, comparison: pd.DataFrame, output: Path) -> None:
    count = dict(zip(summary["class_name"], summary["n_accessions"]))
    ratio = max(count.values()) / min(count.values())
    top_multi = int(master["is_multifunctional_top_level"].sum())
    full_multi = int(master["is_multifunctional_full_ec"].sum())
    duplicate_sequences = int(master.duplicated("sequence_sha256", keep=False).sum())
    unchanged = int((comparison["status"] == "unchanged").sum())
    changed = int((comparison["status"] == "top_level_label_changed").sum())
    absent = int((comparison["status"] == "absent_from_current_query").sum())
    draft = f"""# Results 3.1 — Current-release dataset construction

Retrieval of reviewed, EC-annotated archaeal proteins from UniProtKB/Swiss-Prot
release 2026_02 yielded {len(master):,} unique accessions and
{master['sequence_sha256'].nunique():,} unique amino-acid sequences. Transferases
were the most frequent top-level class ({count['Transferases']:,} proteins), whereas
translocases were the least frequent ({count['Translocases']:,} proteins), resulting
in a {ratio:.2f}-fold imbalance in top-level class membership. The dataset comprised
{master['taxonomy_id'].nunique():,} species-level taxonomy identifiers and
{master['phylum'].nunique():,} resolved archaeal phyla.

At the annotation level, {full_multi:,} proteins carried more than one full EC number
and {top_multi:,} crossed two or more top-level EC classes. Because the latter proteins
cannot be represented consistently by a single softmax target, they were retained as
multi-label observations and marked for the auxiliary multifunctionality task.
There were {duplicate_sequences:,} accession rows belonging to exact-sequence duplicate
groups; these groups will be constrained to a single data partition.

Comparison with the original seven FASTA files found {unchanged:,} legacy accessions
with unchanged top-level labels, {changed:,} with changed top-level labels, and
{absent:,} absent from the current reviewed archaeal EC query. Current-release
annotations will be treated as authoritative for model development, while the legacy
comparison remains available for provenance and annotation-drift analysis.

> All values in this section are computed. Homology clustering has not yet been
> applied, so cluster counts must not be inferred from these totals.
"""
    (output / "results_3.1_updated_draft.md").write_text(draft, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--uniprot", required=True, help="TSV path or glob pattern")
    parser.add_argument("--legacy", type=Path, required=True, help="Results 3.1 unique_proteins.csv")
    parser.add_argument("--processed-dir", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    args = parser.parse_args()

    uniprot_path = find_single(args.uniprot)
    processed = args.processed_dir.resolve()
    results = args.results_dir.resolve()
    processed.mkdir(parents=True, exist_ok=True)
    results.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(uniprot_path, sep="\t", dtype=str)
    legacy = pd.read_csv(args.legacy, dtype=str)
    master = prepare(raw)
    summary = class_summary(master)
    comparison = compare_legacy(master, legacy)

    master.to_csv(processed / "archaea_swissprot_ec_master.csv", index=False, encoding="utf-8-sig")
    with (processed / "archaea_swissprot_ec_master.fasta").open("w", encoding="ascii", newline="\n") as handle:
        for row in master.itertuples(index=False):
            handle.write(f">{row.accession}|EC_TOP={row.top_classes}|OX={row.taxonomy_id}\n{row.sequence}\n")
    summary.to_csv(results / "updated_dataset_summary.csv", index=False, encoding="utf-8-sig")
    comparison.to_csv(results / "legacy_update_comparison.csv", index=False, encoding="utf-8-sig")
    master["top_class_names"].value_counts().rename_axis("class_combination").reset_index(name="n_accessions").to_csv(
        results / "updated_label_combinations.csv", index=False, encoding="utf-8-sig"
    )
    figures(master, summary, results)
    updated_draft(master, summary, comparison, results)

    report = {
        "uniprot_input": str(uniprot_path),
        "master_accessions": int(len(master)),
        "unique_sequences": int(master["sequence_sha256"].nunique()),
        "species_taxonomy_ids": int(master["taxonomy_id"].nunique()),
        "top_level_multifunctional": int(master["is_multifunctional_top_level"].sum()),
        "full_ec_multifunctional": int(master["is_multifunctional_full_ec"].sum()),
        "processed_csv": str(processed / "archaea_swissprot_ec_master.csv"),
    }
    (processed / "master_dataset_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
