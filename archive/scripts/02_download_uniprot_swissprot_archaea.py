#!/usr/bin/env python3
"""Download reviewed archaeal proteins with EC annotations from UniProtKB.

The exact query, response headers, timestamp, URL, record count, and checksum are
written to a JSON manifest. Existing raw files are not overwritten unless --force
is explicitly supplied.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests


QUERY = "(reviewed:true) AND (taxonomy_id:2157) AND (ec:*)"
FIELDS = [
    "accession",
    "id",
    "protein_name",
    "organism_name",
    "organism_id",
    "lineage",
    "length",
    "ec",
    "protein_existence",
    "fragment",
    "date_created",
    "date_modified",
    "sequence",
    "ft_act_site",
    "ft_binding",
    "ft_transmem",
]
API_URL = "https://rest.uniprot.org/uniprotkb/stream"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    retrieved_at = datetime.now(timezone.utc)
    date_tag = retrieved_at.strftime("%Y%m%d")
    output_path = output_dir / f"uniprot_swissprot_archaea_ec_{date_tag}.tsv"
    manifest_path = output_dir / f"uniprot_swissprot_archaea_ec_{date_tag}.manifest.json"

    if output_path.exists() and not args.force:
        raise FileExistsError(f"Raw file already exists: {output_path}; use --force only for an intentional replacement")

    params = {"query": QUERY, "format": "tsv", "fields": ",".join(FIELDS)}
    with requests.get(API_URL, params=params, stream=True, timeout=(30, 300)) as response:
        response.raise_for_status()
        with output_path.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)
        response_headers = {
            key: response.headers.get(key)
            for key in ["x-total-results", "x-uniprot-release", "x-uniprot-release-date", "content-type"]
        }
        final_url = response.url

    table = pd.read_csv(output_path, sep="\t", dtype=str)
    manifest = {
        "retrieved_at_utc": retrieved_at.isoformat(),
        "api_url": API_URL,
        "final_url": final_url,
        "query": QUERY,
        "fields": FIELDS,
        "response_headers": response_headers,
        "records": int(len(table)),
        "columns": list(table.columns),
        "bytes": output_path.stat().st_size,
        "sha256": file_sha256(output_path),
        "raw_file": str(output_path),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

