"""Export the locked KlaDB human study holdout for a private Kaggle dataset."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
SOURCE_SHA256 = "2a46fd62b1d4d67ded1c31d8a9956f48eab6b7c33aba146f9d9f24c267b34748"
EXPECTED = {"rows": 5095, "positives": 254, "putative_negatives": 4841, "proteins": 81}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def export(source: Path, destination: Path) -> dict:
    if sha256(source) != SOURCE_SHA256:
        raise ValueError("KlaDB full manifest does not match the audited source checksum")
    destination.mkdir(parents=True, exist_ok=True)
    output = destination / "kladb_human_external_test.csv"
    rows = []
    with source.open(newline="") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames
        if fields is None or "partition" not in fields:
            raise ValueError("Source manifest is missing its partition column")
        for row in reader:
            if row["partition"] == "external_test":
                rows.append(row)
    positives = sum(int(row["label"]) for row in rows)
    counts = {
        "rows": len(rows),
        "positives": positives,
        "putative_negatives": len(rows) - positives,
        "proteins": len({row["accession"] for row in rows}),
    }
    if counts != EXPECTED:
        raise ValueError(f"Unexpected external-study counts: {counts}")
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    manifest = {
        "file": output.name,
        "sha256": sha256(output),
        "source_full_manifest_sha256": SOURCE_SHA256,
        "source_study_pmid": "41192556",
        "counts": counts,
        "negative_definition": "Unreported lysines in positive-bearing proteins; putative, not confirmed non-Kla",
    }
    (destination / "external_test_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=PROJECT / "outputs/kladb_human_locked/full_manifest.csv")
    parser.add_argument("--destination", type=Path, default=PROJECT / "outputs/kladb_external_upload")
    args = parser.parse_args()
    print(json.dumps(export(args.source, args.destination), indent=2))


if __name__ == "__main__":
    main()
