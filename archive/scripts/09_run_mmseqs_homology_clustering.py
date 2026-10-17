#!/usr/bin/env python3
"""Run connected-component MMseqs2 clustering at specified identity thresholds."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def to_wsl(path: Path) -> str:
    resolved = path.resolve()
    drive = resolved.drive.rstrip(":").lower()
    tail = resolved.as_posix().split(":", 1)[1]
    return f"/mnt/{drive}{tail}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fasta", type=Path, required=True)
    parser.add_argument("--mmseqs", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.5, 0.4, 0.3])
    parser.add_argument("--coverage", type=float, default=0.8)
    parser.add_argument("--sensitivity", type=float, default=7.5)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--wsl-distro", default="Ubuntu-22.04")
    args = parser.parse_args()

    fasta = args.fasta.resolve()
    mmseqs = args.mmseqs.resolve()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    if not fasta.exists() or not mmseqs.exists():
        raise FileNotFoundError(f"Missing FASTA or MMseqs2 binary: {fasta}, {mmseqs}")

    version_result = subprocess.run(
        ["wsl", "-d", args.wsl_distro, "--", to_wsl(mmseqs), "version"],
        check=True,
        capture_output=True,
    )
    version = version_result.stdout.decode("utf-8", errors="replace").strip()
    manifest = {
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "mmseqs_version": version,
        "input_fasta": str(fasta),
        "input_sha256": sha256(fasta),
        "coverage": args.coverage,
        "coverage_mode": 0,
        "cluster_mode": 1,
        "sensitivity": args.sensitivity,
        "threads": args.threads,
        "runs": [],
    }

    for threshold in args.thresholds:
        tag = f"id{int(round(threshold * 100))}"
        run_dir = output_root / tag
        run_dir.mkdir(parents=True, exist_ok=True)
        prefix = run_dir / f"archaea_{tag}"
        temp = run_dir / "tmp"
        expected = Path(str(prefix) + "_cluster.tsv")
        if expected.exists():
            raise FileExistsError(f"Clustering output already exists: {expected}")
        command = [
            "wsl",
            "-d",
            args.wsl_distro,
            "--",
            to_wsl(mmseqs),
            "easy-cluster",
            to_wsl(fasta),
            to_wsl(prefix),
            to_wsl(temp),
            "--min-seq-id",
            str(threshold),
            "-c",
            str(args.coverage),
            "--cov-mode",
            "0",
            "--cluster-mode",
            "1",
            "--single-step-clustering",
            "1",
            "-s",
            str(args.sensitivity),
            "--threads",
            str(args.threads),
            "--remove-tmp-files",
            "1",
        ]
        # Keep WSL output as bytes so Windows does not decode UTF-8 logs using
        # the active GBK code page. The child process inherits the terminal.
        completed = subprocess.run(command, check=True)
        if not expected.exists() or expected.stat().st_size == 0:
            raise RuntimeError(f"Expected cluster TSV was not created: {expected}")
        line_count = sum(1 for _ in expected.open("r", encoding="utf-8"))
        representatives = set()
        with expected.open("r", encoding="utf-8") as handle:
            for line in handle:
                representative, _member = line.rstrip("\n").split("\t")[:2]
                representatives.add(representative)
        manifest["runs"].append(
            {
                "threshold": threshold,
                "tag": tag,
                "command": command,
                "cluster_tsv": str(expected),
                "cluster_tsv_sha256": sha256(expected),
                "members": line_count,
                "clusters": len(representatives),
                "exit_code": completed.returncode,
            }
        )

    manifest["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    manifest_path = output_root / "mmseqs_clustering_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    for run in manifest['runs']:
        individual={**manifest,'runs':[run]}
        (output_root/run['tag']/'clustering_manifest.json').write_text(
            json.dumps(individual,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
