from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from map_windows_to_proteome import parse_pcbert_window_file


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit_mapping(
    mapping_path: Path,
    train_path: Path,
    test_path: Path,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    records = [
        *parse_pcbert_window_file(train_path, "train"),
        *parse_pcbert_window_file(test_path, "test"),
    ]
    record_by_key = {(record.split, record.record_name): record for record in records}
    if len(record_by_key) != len(records):
        raise ValueError("Benchmark record names are not unique within each split")

    with mapping_path.open(newline="") as handle:
        mapping_rows = list(csv.DictReader(handle))
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in mapping_rows:
        key = (row["split"], row["record_name"])
        record = record_by_key.get(key)
        if record is None:
            raise ValueError(f"Mapping contains an unknown benchmark record: {key}")
        if (
            row["window_sequence"] != record.sequence
            or int(row["label"]) != record.label
            or int(row["window_position"]) != record.window_position
        ):
            raise ValueError(f"Mapping disagrees with benchmark record: {key}")
        grouped[key].append(row)
    if set(grouped) != set(record_by_key):
        raise ValueError("Mapping does not cover every benchmark record")

    train_sequences = {record.sequence for record in records if record.split == "train"}
    manifest: list[dict[str, object]] = []
    accession_splits: dict[str, set[str]] = defaultdict(set)
    unique_accession_splits: dict[str, set[str]] = defaultdict(set)
    for record in records:
        key = (record.split, record.record_name)
        rows = grouped[key]
        sites = {
            (row["accession"], int(row["central_lysine_full_position_1based"]))
            for row in rows
            if row["accession"] and row["central_lysine_full_position_1based"]
        }
        expected_status = (
            "no_match" if not sites else "unique_match" if len(sites) == 1 else "multiple_matches"
        )
        if any(row["match_status"] != expected_status for row in rows):
            raise ValueError(f"Mapping status disagrees with candidate sites for {key}")
        accessions = {accession for accession, _ in sites}
        for accession in accessions:
            accession_splits[accession].add(record.split)
        if len(sites) == 1:
            unique_accession_splits[next(iter(accessions))].add(record.split)
        manifest.append(
            {
                "original_split": record.split,
                "record_name": record.record_name,
                "label": record.label,
                "window_sequence": record.sequence,
                "exact_window_in_original_train": record.sequence in train_sequences,
                "candidate_site_count": len(sites),
                "candidate_accession_count": len(accessions),
                "candidate_sites": ";".join(
                    f"{accession}:K{position}" for accession, position in sorted(sites)
                ),
                "mapping_interpretation": (
                    "unmatched"
                    if not sites
                    else "unique_site"
                    if len(sites) == 1
                    else "ambiguous_site"
                ),
            }
        )

    status_counts = Counter(row["mapping_interpretation"] for row in manifest)
    status_by_split = {}
    for split in ("train", "test"):
        status_by_split[split] = dict(
            Counter(
                row["mapping_interpretation"]
                for row in manifest
                if row["original_split"] == split
            )
        )
    shared_unique_accessions = {
        accession for accession, splits in unique_accession_splits.items() if len(splits) > 1
    }
    shared_candidate_accessions = {
        accession for accession, splits in accession_splits.items() if len(splits) > 1
    }
    def test_rows_on_shared_accessions(*, new_windows_only: bool) -> int:
        return sum(
            row["original_split"] == "test"
            and row["mapping_interpretation"] == "unique_site"
            and (not new_windows_only or not row["exact_window_in_original_train"])
            and row["candidate_sites"].split(":", 1)[0] in shared_unique_accessions
            for row in manifest
        )

    report: dict[str, object] = {
        "input_sha256": {
            "mapping_csv": sha256_file(mapping_path),
            "train_windows": sha256_file(train_path),
            "test_windows": sha256_file(test_path),
        },
        "benchmark_rows": len(records),
        "benchmark_unique_windows": len({record.sequence for record in records}),
        "mapping_rows": len(mapping_rows),
        "match_status_rows": dict(status_counts),
        "match_status_by_original_split": status_by_split,
        "unique_candidate_accessions": len(accession_splits),
        "accessions_in_both_original_splits_from_unique_mappings": len(shared_unique_accessions),
        "test_rows_on_accessions_in_both_original_splits_from_unique_mappings":
            test_rows_on_shared_accessions(new_windows_only=False),
        "test_rows_without_exact_train_window": sum(
            row["original_split"] == "test" and not row["exact_window_in_original_train"]
            for row in manifest
        ),
        "test_rows_without_exact_train_window_on_shared_unique_accessions":
            test_rows_on_shared_accessions(new_windows_only=True),
        "accessions_in_both_original_splits_from_any_candidate_mapping": len(
            shared_candidate_accessions
        ),
        "interpretation": (
            "Exact matches to a candidate proteome do not prove that a benchmark label "
            "came from that protein or species. Ambiguous mappings cannot be assigned "
            "a single parent. No new split is created by this audit."
        ),
    }
    return report, manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit candidate protein sites for benchmark windows"
    )
    parser.add_argument("--mapping-csv", type=Path, required=True)
    parser.add_argument("--train-windows", type=Path, required=True)
    parser.add_argument("--test-windows", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report, manifest = audit_mapping(args.mapping_csv, args.train_windows, args.test_windows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "candidate_protein_mapping_audit.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    with (args.output_dir / "candidate_protein_sites.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(manifest)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
