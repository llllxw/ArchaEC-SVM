#!/usr/bin/env python3
"""Create leakage-controlled random and species-disjoint initial splits.

The random split groups identical amino-acid sequences. The species split groups all
accessions sharing a UniProt taxonomy identifier. A deterministic greedy allocator
balances sample counts, seven labels, and top-level multifunctionality.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


LABELS = [f"label_ec{index}" for index in range(1, 8)]
SPLITS = ("train", "validation", "test")
FRACTIONS = np.array([0.8, 0.1, 0.1])


def group_statistics(data: pd.DataFrame, group_column: str) -> list[dict[str, object]]:
    groups = []
    for group_id, frame in data.groupby(group_column, sort=False):
        groups.append(
            {
                "group_id": str(group_id),
                "indices": frame.index.to_numpy(),
                "size": len(frame),
                "labels": frame[LABELS].sum(axis=0).to_numpy(dtype=float),
                "multi": float(frame["is_multifunctional_top_level"].astype(int).sum()),
            }
        )
    return groups


def allocate_groups(data: pd.DataFrame, group_column: str, seed: int) -> pd.Series:
    groups = group_statistics(data, group_column)
    rng = np.random.default_rng(seed)
    rng.shuffle(groups)
    total_size = len(data)
    total_labels = data[LABELS].sum(axis=0).to_numpy(dtype=float)
    total_multi = float(data["is_multifunctional_top_level"].astype(int).sum())
    rarity = 1.0 / np.maximum(total_labels, 1.0)

    groups.sort(
        key=lambda group: (
            float(np.dot(group["labels"], rarity)),
            group["size"],
        ),
        reverse=True,
    )
    target_size = FRACTIONS * total_size
    target_group_count = FRACTIONS * len(groups)
    target_labels = FRACTIONS[:, None] * total_labels[None, :]
    target_multi = FRACTIONS * total_multi
    current_size = np.zeros(3)
    current_group_count = np.zeros(3)
    current_labels = np.zeros((3, 7))
    current_multi = np.zeros(3)
    assignments: dict[str, str] = {}

    for group in groups:
        scores = []
        for split_index in range(3):
            next_size = current_size.copy()
            next_group_count = current_group_count.copy()
            next_labels = current_labels.copy()
            next_multi = current_multi.copy()
            next_size[split_index] += group["size"]
            next_group_count[split_index] += 1
            next_labels[split_index] += group["labels"]
            next_multi[split_index] += group["multi"]
            size_error = np.mean(((next_size - target_size) / np.maximum(target_size, 1)) ** 2)
            group_error = np.mean(
                ((next_group_count - target_group_count) / np.maximum(target_group_count, 1)) ** 2
            )
            label_error = np.mean(((next_labels - target_labels) / np.maximum(target_labels, 1)) ** 2)
            multi_error = np.mean(((next_multi - target_multi) / np.maximum(target_multi, 1)) ** 2)
            overflow = max(0.0, next_size[split_index] - target_size[split_index] * 1.03) / max(target_size[split_index], 1)
            group_overflow = max(
                0.0, next_group_count[split_index] - target_group_count[split_index] * 1.03
            ) / max(target_group_count[split_index], 1)
            scores.append(
                label_error * 2.0
                + size_error
                + group_error * 2.0
                + multi_error
                + overflow * 10.0
                + group_overflow * 10.0
            )
        chosen = int(np.argmin(scores))
        assignments[group["group_id"]] = SPLITS[chosen]
        current_size[chosen] += group["size"]
        current_group_count[chosen] += 1
        current_labels[chosen] += group["labels"]
        current_multi[chosen] += group["multi"]

    return data[group_column].astype(str).map(assignments)


def randomized_group_allocation(
    data: pd.DataFrame, group_column: str, seed: int, n_trials: int = 200
) -> pd.Series:
    """Random group split selected only by aggregate balance, never test performance."""
    group_ids = data[group_column].astype(str).unique()
    n_groups = len(group_ids)
    n_train = round(FRACTIONS[0] * n_groups)
    n_validation = round(FRACTIONS[1] * n_groups)
    best_score = float("inf")
    best_assignment: pd.Series | None = None
    total_labels = data[LABELS].sum(axis=0).to_numpy(dtype=float)
    total_multi = max(float(data["is_multifunctional_top_level"].astype(int).sum()), 1.0)

    for trial in range(n_trials):
        shuffled = group_ids.copy()
        np.random.default_rng(seed + trial).shuffle(shuffled)
        mapping = {group_id: "train" for group_id in shuffled[:n_train]}
        mapping.update(
            {group_id: "validation" for group_id in shuffled[n_train : n_train + n_validation]}
        )
        mapping.update({group_id: "test" for group_id in shuffled[n_train + n_validation :]})
        assignment = data[group_column].astype(str).map(mapping)
        size_fraction = np.array([(assignment == split).mean() for split in SPLITS])
        label_fraction = np.vstack(
            [data.loc[assignment == split, LABELS].sum(axis=0).to_numpy(dtype=float) for split in SPLITS]
        ) / np.maximum(total_labels, 1.0)
        multi_fraction = np.array(
            [
                data.loc[assignment == split, "is_multifunctional_top_level"].astype(int).sum()
                for split in SPLITS
            ],
            dtype=float,
        ) / total_multi
        score = (
            np.mean((size_fraction - FRACTIONS) ** 2)
            + np.mean((label_fraction - FRACTIONS[:, None]) ** 2) * 2.0
            + np.mean((multi_fraction - FRACTIONS) ** 2)
        )
        if score < best_score:
            best_score = score
            best_assignment = assignment
    assert best_assignment is not None
    return best_assignment


def summarize(data: pd.DataFrame, split_column: str, group_column: str) -> pd.DataFrame:
    rows = []
    for split in SPLITS:
        frame = data[data[split_column] == split]
        row = {
            "split": split,
            "n_accessions": len(frame),
            "n_groups": frame[group_column].nunique(),
            "n_species": frame["taxonomy_id"].nunique(),
            "n_multifunctional_top_level": int(frame["is_multifunctional_top_level"].astype(bool).sum()),
        }
        row.update({label: int(frame[label].sum()) for label in LABELS})
        rows.append(row)
    return pd.DataFrame(rows)


def verify_no_overlap(data: pd.DataFrame, split_column: str, group_column: str) -> None:
    counts = data.groupby(group_column)[split_column].nunique()
    if (counts > 1).any():
        raise AssertionError(f"{int((counts > 1).sum())} groups cross partitions for {split_column}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--master", type=Path, required=True)
    parser.add_argument("--random-output", type=Path, required=True)
    parser.add_argument("--species-output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data = pd.read_csv(args.master, dtype={"taxonomy_id": str})
    data["split_random"] = randomized_group_allocation(data, "sequence_sha256", args.seed)
    data["split_species"] = allocate_groups(data, "taxonomy_id", args.seed)
    verify_no_overlap(data, "split_random", "sequence_sha256")
    verify_no_overlap(data, "split_species", "taxonomy_id")

    random_out = args.random_output.resolve()
    species_out = args.species_output.resolve()
    random_out.mkdir(parents=True, exist_ok=True)
    species_out.mkdir(parents=True, exist_ok=True)

    manifest_columns = ["accession", "sequence_sha256", "taxonomy_id", "top_classes", "is_multifunctional_top_level"]
    data[manifest_columns + ["split_random"]].to_csv(random_out / "split_manifest.csv", index=False, encoding="utf-8-sig")
    data[manifest_columns + ["split_species"]].to_csv(species_out / "species_split_manifest.csv", index=False, encoding="utf-8-sig")
    random_summary = summarize(data, "split_random", "sequence_sha256")
    species_summary = summarize(data, "split_species", "taxonomy_id")
    random_summary.to_csv(random_out / "split_summary.csv", index=False, encoding="utf-8-sig")
    species_summary.to_csv(species_out / "species_split_summary.csv", index=False, encoding="utf-8-sig")

    report = {
        "seed": args.seed,
        "fractions": dict(zip(SPLITS, FRACTIONS.tolist())),
        "random_group": "exact amino-acid sequence SHA-256",
        "species_group": "UniProt organism taxonomy ID",
        "random_counts": dict(zip(random_summary["split"], random_summary["n_accessions"].astype(int))),
        "species_counts": dict(zip(species_summary["split"], species_summary["n_accessions"].astype(int))),
    }
    (random_out / "split_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (species_out / "species_split_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
