#!/usr/bin/env python3
"""Create balanced train/validation/test splits from MMseqs2 cluster TSV files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


LABELS = [f"label_ec{index}" for index in range(1, 8)]
SPLITS = ("train", "validation", "test")
FRACTIONS = np.array([0.8, 0.1, 0.1])


def accession(header: str) -> str:
    return header.split("|", 1)[0]


def load_clusters(path: Path) -> pd.DataFrame:
    pairs = pd.read_csv(path, sep="\t", header=None, names=["representative_header", "member_header"], dtype=str)
    pairs["homology_cluster_id"] = pairs["representative_header"].map(accession)
    pairs["accession"] = pairs["member_header"].map(accession)
    if pairs["accession"].duplicated().any():
        raise ValueError(f"Cluster TSV contains duplicated members: {path}")
    return pairs[["accession", "homology_cluster_id"]]


def allocate(data: pd.DataFrame, seed: int) -> pd.Series:
    groups = []
    for group_id, frame in data.groupby("homology_cluster_id", sort=False):
        groups.append(
            {
                "group_id": group_id,
                "size": len(frame),
                "labels": frame[LABELS].sum(axis=0).to_numpy(dtype=float),
                "multi": float(frame["is_multifunctional_top_level"].astype(int).sum()),
            }
        )
    rng = np.random.default_rng(seed)
    rng.shuffle(groups)
    total_size = len(data)
    total_labels = data[LABELS].sum(axis=0).to_numpy(dtype=float)
    total_multi = max(float(data["is_multifunctional_top_level"].astype(int).sum()), 1.0)
    target_size = FRACTIONS * total_size
    target_groups = FRACTIONS * len(groups)
    target_labels = FRACTIONS[:, None] * total_labels[None, :]
    target_multi = FRACTIONS * total_multi
    rarity = 1.0 / np.maximum(total_labels, 1.0)
    groups.sort(key=lambda g: (float(np.dot(g["labels"], rarity)), g["size"]), reverse=True)

    current_size = np.zeros(3)
    current_groups = np.zeros(3)
    current_labels = np.zeros((3, 7))
    current_multi = np.zeros(3)
    assignments: dict[str, str] = {}
    for group in groups:
        scores = []
        for index in range(3):
            size = current_size.copy(); size[index] += group["size"]
            n_groups = current_groups.copy(); n_groups[index] += 1
            labels = current_labels.copy(); labels[index] += group["labels"]
            multi = current_multi.copy(); multi[index] += group["multi"]
            size_error = np.mean(((size - target_size) / np.maximum(target_size, 1)) ** 2)
            group_error = np.mean(((n_groups - target_groups) / np.maximum(target_groups, 1)) ** 2)
            label_error = np.mean(((labels - target_labels) / np.maximum(target_labels, 1)) ** 2)
            multi_error = np.mean(((multi - target_multi) / np.maximum(target_multi, 1)) ** 2)
            size_overflow = max(0, size[index] - target_size[index] * 1.05) / max(target_size[index], 1)
            group_overflow = max(0, n_groups[index] - target_groups[index] * 1.05) / max(target_groups[index], 1)
            scores.append(label_error * 3 + size_error * 2 + group_error + multi_error + 10 * size_overflow + 5 * group_overflow)
        selected = int(np.argmin(scores))
        assignments[group["group_id"]] = SPLITS[selected]
        current_size[selected] += group["size"]
        current_groups[selected] += 1
        current_labels[selected] += group["labels"]
        current_multi[selected] += group["multi"]
    return data["homology_cluster_id"].map(assignments)


def summary(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for split in SPLITS:
        frame = data[data["split_homology"] == split]
        row = {
            "split": split,
            "n_accessions": len(frame),
            "n_clusters": frame["homology_cluster_id"].nunique(),
            "n_species": frame["taxonomy_id"].nunique(),
            "n_multifunctional": int(frame["is_multifunctional_top_level"].astype(bool).sum()),
        }
        row.update({label: int(frame[label].sum()) for label in LABELS})
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--master", type=Path, required=True)
    parser.add_argument("--cluster-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--thresholds", type=int, nargs="+", default=[50, 40, 30])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    master = pd.read_csv(args.master, dtype={"taxonomy_id": str})
    output_root = args.output_root.resolve()
    reports = []
    for threshold in args.thresholds:
        tag = f"id{threshold}"
        cluster_path = args.cluster_root / tag / f"archaea_{tag}_cluster.tsv"
        clusters = load_clusters(cluster_path)
        data = master.merge(clusters, on="accession", how="inner", validate="one_to_one")
        if len(data) != len(master):
            missing = sorted(set(master["accession"]) - set(data["accession"]))
            raise ValueError(f"{tag}: cluster coverage incomplete; missing {len(missing)} accessions")
        data["split_homology"] = allocate(data, args.seed + threshold)
        overlap = data.groupby("homology_cluster_id")["split_homology"].nunique()
        if (overlap > 1).any():
            raise AssertionError(f"{tag}: homology clusters cross partitions")
        out = output_root / f"homology_{tag}"
        out.mkdir(parents=True, exist_ok=True)
        data[["accession", "homology_cluster_id", "taxonomy_id", "top_classes", "is_multifunctional_top_level", "split_homology"]].to_csv(
            out / "split_manifest.csv", index=False, encoding="utf-8-sig"
        )
        split_summary = summary(data)
        split_summary.to_csv(out / "split_summary.csv", index=False, encoding="utf-8-sig")
        report = {
            "threshold_percent_identity": threshold,
            "coverage": 0.8,
            "seed": args.seed + threshold,
            "clusters": int(data["homology_cluster_id"].nunique()),
            "accessions": int(len(data)),
            "cluster_overlap_count": int((overlap > 1).sum()),
            "split_accessions": dict(zip(split_summary["split"], split_summary["n_accessions"].astype(int))),
            "split_clusters": dict(zip(split_summary["split"], split_summary["n_clusters"].astype(int))),
        }
        (out / "split_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        reports.append(report)
    print(json.dumps(reports, indent=2))


if __name__ == "__main__":
    main()

