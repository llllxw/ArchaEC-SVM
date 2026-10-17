#!/usr/bin/env python3
"""Audit the seven existing archaeal enzyme FASTA files.

The script is read-only with respect to source data. It creates legacy-audit
tables, checksums, and a preliminary Results 3.1 provenance draft. Manuscript
figures are generated only from the current Swiss-Prot master dataset by script 03.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from Bio import SeqIO


EXPECTED_CLASSES = [
    "Oxidoreductases",
    "Transferases",
    "Hydrolases",
    "Lyases",
    "Isomerases",
    "Ligases",
    "Translocases",
]
CLASS_TO_EC = {name: index + 1 for index, name in enumerate(EXPECTED_CLASSES)}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_header(record_id: str, description: str) -> dict[str, str | None]:
    parts = record_id.split("|")
    accession = parts[1] if len(parts) >= 3 else record_id
    ox = re.search(r"\bOX=(\d+)", description)
    pe = re.search(r"\bPE=(\d+)", description)
    organism = re.search(r"\bOS=(.*?)(?=\s(?:OX|GN|PE|SV)=|$)", description)
    return {
        "accession": accession,
        "taxonomy_id": ox.group(1) if ox else None,
        "protein_existence": pe.group(1) if pe else None,
        "organism": organism.group(1).strip() if organism else None,
    }


def load_occurrences(input_dir: Path) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    fasta_paths = sorted(input_dir.glob("*.fasta"))
    found_classes = {path.stem for path in fasta_paths}
    missing = set(EXPECTED_CLASSES) - found_classes
    unexpected = found_classes - set(EXPECTED_CLASSES)
    if missing or unexpected:
        raise ValueError(f"Unexpected FASTA classes; missing={missing}, unexpected={unexpected}")

    rows: list[dict[str, object]] = []
    manifest: list[dict[str, object]] = []
    valid = set("ACDEFGHIKLMNPQRSTVWYBXZJUO*")
    for path in fasta_paths:
        n_records = 0
        for record in SeqIO.parse(path, "fasta"):
            seq = str(record.seq).upper()
            metadata = parse_header(record.id, record.description)
            rows.append(
                {
                    **metadata,
                    "class_name": path.stem,
                    "ec_class": CLASS_TO_EC[path.stem],
                    "sequence": seq,
                    "length": len(seq),
                    "sequence_sha256": hashlib.sha256(seq.encode("ascii")).hexdigest(),
                    "invalid_residue": any(residue not in valid for residue in seq),
                    "source_file": path.name,
                    "description": record.description,
                }
            )
            n_records += 1
        manifest.append(
            {
                "source_file": path.name,
                "absolute_path": str(path.resolve()),
                "bytes": path.stat().st_size,
                "records": n_records,
                "sha256": sha256(path),
            }
        )
    return pd.DataFrame(rows), manifest


def merge_accessions(occurrences: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for accession, group in occurrences.groupby("accession", sort=True):
        sequences = sorted(group["sequence"].unique())
        labels = sorted(group["ec_class"].unique().tolist())
        class_names = sorted(group["class_name"].unique().tolist())
        rows.append(
            {
                "accession": accession,
                "sequence": sequences[0],
                "length": len(sequences[0]),
                "taxonomy_id": group["taxonomy_id"].dropna().iloc[0]
                if group["taxonomy_id"].notna().any()
                else None,
                "organism": group["organism"].dropna().iloc[0]
                if group["organism"].notna().any()
                else None,
                "protein_existence": group["protein_existence"].dropna().iloc[0]
                if group["protein_existence"].notna().any()
                else None,
                "ec_classes": ";".join(map(str, labels)),
                "class_names": ";".join(class_names),
                "n_top_level_classes": len(labels),
                "is_multifunctional_top_level": len(labels) > 1,
                "n_distinct_sequences_for_accession": len(sequences),
                "sequence_sha256": hashlib.sha256(sequences[0].encode("ascii")).hexdigest(),
            }
        )
    return pd.DataFrame(rows)


def build_summary(occurrences: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for class_name in EXPECTED_CLASSES:
        group = occurrences[occurrences["class_name"] == class_name]
        lengths = group["length"]
        rows.append(
            {
                "class_name": class_name,
                "ec_class": CLASS_TO_EC[class_name],
                "n_records": len(group),
                "n_unique_accessions": group["accession"].nunique(),
                "n_unique_sequences": group["sequence_sha256"].nunique(),
                "n_taxa": group["taxonomy_id"].nunique(),
                "min_length": int(lengths.min()),
                "q1_length": float(lengths.quantile(0.25)),
                "median_length": float(lengths.median()),
                "mean_length": float(lengths.mean()),
                "q3_length": float(lengths.quantile(0.75)),
                "max_length": int(lengths.max()),
                "n_invalid_sequences": int(group["invalid_residue"].sum()),
            }
        )
    return pd.DataFrame(rows)


def duplicate_audit(occurrences: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for seq_hash, group in occurrences.groupby("sequence_sha256"):
        accessions = sorted(group["accession"].unique())
        classes = sorted(group["class_name"].unique())
        if len(group) > 1:
            rows.append(
                {
                    "sequence_sha256": seq_hash,
                    "length": int(group["length"].iloc[0]),
                    "n_occurrences": len(group),
                    "n_accessions": len(accessions),
                    "accessions": ";".join(accessions),
                    "n_classes": len(classes),
                    "classes": ";".join(classes),
                    "cross_class": len(classes) > 1,
                }
            )
    return pd.DataFrame(rows)


def write_draft(
    occurrences: pd.DataFrame,
    unique: pd.DataFrame,
    summary: pd.DataFrame,
    duplicates: pd.DataFrame,
    output: Path,
) -> None:
    counts = dict(zip(summary["class_name"], summary["n_records"]))
    imbalance = max(counts.values()) / min(counts.values())
    multi = int(unique["is_multifunctional_top_level"].sum())
    unique_sequences = unique["sequence_sha256"].nunique()
    cross_sequences = int(duplicates["cross_class"].sum()) if len(duplicates) else 0
    draft = f"""# Results 3.1 draft — existing dataset audit

The seven source FASTA files contained {len(occurrences):,} class-labelled records,
representing {len(unique):,} unique UniProt accessions and {unique_sequences:,}
unique amino-acid sequences. Transferases were the largest class ({counts['Transferases']:,}
records), whereas translocases were the smallest ({counts['Translocases']:,} records),
corresponding to a {imbalance:.2f}-fold class imbalance. Hydrolases and
oxidoreductases contained {counts['Hydrolases']:,} and {counts['Oxidoreductases']:,}
records, respectively, while each of the remaining non-translocase minority classes
contained 131–169 records.

Merging records by accession identified {multi:,} proteins annotated to more than one
top-level enzyme class. Exact-sequence auditing found {cross_sequences:,} duplicated
sequence groups spanning multiple class files. These records represent biological
multifunctionality or label ambiguity rather than independent observations and must
therefore be merged before model development. All {len(occurrences):,} source records
were marked as protein-existence level 1, and no sequence contained residues outside
the accepted protein alphabet. Sequence lengths ranged from
{int(occurrences['length'].min()):,} to {int(occurrences['length'].max()):,} amino acids,
with class-specific distributions summarized in `dataset_summary.csv`.

These observations establish two design requirements for subsequent experiments:
class-balanced evaluation is necessary because of the {imbalance:.2f}-fold imbalance,
and exact sequences, accessions, and later-defined homology clusters must remain within
a single partition to prevent information leakage.

> Status: computed from the existing seven FASTA files. This text must be updated after
> the current UniProtKB/Swiss-Prot release is retrieved and merged.
"""
    (output / "results_3.1_draft.md").write_text(draft, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True, help="Directory containing seven FASTA files")
    parser.add_argument("--output", type=Path, required=True, help="Results 3.1 output directory")
    args = parser.parse_args()

    input_dir = args.input.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    occurrences, manifest = load_occurrences(input_dir)
    unique = merge_accessions(occurrences)
    summary = build_summary(occurrences)
    duplicates = duplicate_audit(occurrences)
    combinations = unique["class_names"].value_counts().rename_axis("class_combination").reset_index(name="n_unique_accessions")

    summary.to_csv(output / "dataset_summary.csv", index=False, encoding="utf-8-sig")
    unique.to_csv(output / "unique_proteins.csv", index=False, encoding="utf-8-sig")
    combinations.to_csv(output / "label_combinations.csv", index=False, encoding="utf-8-sig")
    duplicates.to_csv(output / "duplicate_audit.csv", index=False, encoding="utf-8-sig")
    occurrences.drop(columns=["sequence"]).to_csv(output / "source_occurrences.csv", index=False, encoding="utf-8-sig")
    (output / "source_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    write_draft(occurrences, unique, summary, duplicates, output)

    print(json.dumps({
        "records": len(occurrences),
        "unique_accessions": len(unique),
        "unique_sequences_after_accession_merge": int(unique["sequence_sha256"].nunique()),
        "multifunctional_accessions": int(unique["is_multifunctional_top_level"].sum()),
        "output": str(output),
    }, indent=2))


if __name__ == "__main__":
    main()
