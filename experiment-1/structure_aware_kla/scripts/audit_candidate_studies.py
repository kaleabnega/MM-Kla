from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from collections import Counter
from pathlib import Path

from map_windows_to_proteome import parse_fasta, parse_pcbert_window_file


STUDIES = {
    "rice": ("Oryza sativa", "34264677"),
    "wheat": ("Triticum aestivum", "38003390"),
    "soybean": ("Glycine max", "40440508"),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_curated(path: Path) -> dict[str, list[dict[str, str]]]:
    rows: dict[str, list[dict[str, str]]] = {name: [] for name in STUDIES}
    with gzip.open(path, "rt", newline="") as source:
        for row in csv.DictReader(source):
            for name, (species, pmid) in STUDIES.items():
                if row["species"] == species and pmid in row["pmids"].split(";"):
                    rows[name].append(row)
    return rows


def study_summary(rows: list[dict[str, str]]) -> dict[str, int]:
    complete = [row["window_51"] for row in rows if len(row["window_51"]) == 51 and "*" not in row["window_51"]]
    return {
        "curated_site_rows": len(rows),
        "distinct_accession_position_sites": len({(row["uniprot_id"], row["position"]) for row in rows}),
        "accessions": len({row["uniprot_id"] for row in rows}),
        "missing_accession_rows": sum(not row["uniprot_id"] for row in rows),
        "invalid_center_rows": sum(len(row["window_51"]) != 51 or row["window_51"][25] != "K" for row in rows),
        "unpadded_51aa_rows": len(complete),
        "distinct_unpadded_windows": len(set(complete)),
        "repeated_unpadded_window_rows": len(complete) - len(set(complete)),
    }


def audit_rice(
    primary_path: Path,
    curated: list[dict[str, str]],
    proteome_path: Path,
) -> tuple[dict[str, object], list[dict[str, str]]]:
    with primary_path.open(newline="") as source:
        primary = list(csv.DictReader(source))
    proteins = {record.accession: record.sequence for record in parse_fasta(proteome_path)}
    by_site = {(row["uniprot_id"], int(row["position"])): row for row in curated}
    primary_sites = {(row["Protein accession"], int(float(row["Position"]))) for row in primary}
    curated_sites = set(by_site)
    if len(primary_sites) != len(primary) or len(curated_sites) != len(curated):
        raise ValueError("Duplicate accession-position site in primary or curated rice data")

    counts: Counter[str] = Counter()
    verified: list[dict[str, str]] = []
    for row in primary:
        accession = row["Protein accession"]
        position = int(float(row["Position"]))
        sequence = proteins.get(accession)
        if row["Amino acid"] != "K":
            counts["source_row_not_lysine"] += 1
        elif sequence is None:
            counts["accession_absent_from_proteome"] += 1
        elif not 1 <= position <= len(sequence):
            counts["position_out_of_range"] += 1
        elif sequence[position - 1] != "K":
            counts["reported_position_not_lysine"] += 1
        elif position < 26 or position + 25 > len(sequence):
            counts["site_too_close_to_terminus"] += 1
        else:
            window = sequence[position - 26 : position + 25]
            curated_row = by_site.get((accession, position))
            if curated_row is None:
                counts["full_window_absent_from_curated_sites"] += 1
            elif curated_row["window_51"] != window:
                counts["curated_window_mismatch"] += 1
            else:
                counts["primary_and_curated_window_verified"] += 1
                verified.append({
                    "accession": accession,
                    "position_1based": str(position),
                    "window_51aa": window,
                    "label": "1",
                    "source_pmid": "34264677",
                })

    summary: dict[str, object] = {
        "primary_rows": len(primary),
        "primary_accessions": len({row["Protein accession"] for row in primary}),
        "curated_sites_matching_primary_accession_and_position": len(curated_sites & primary_sites),
        "primary_sites_not_in_curated_release": len(primary_sites - curated_sites),
        "curated_sites_not_in_primary_table": len(curated_sites - primary_sites),
        "proteome_protein_count": len(proteins),
        "site_status_counts": dict(sorted(counts.items())),
        "distinct_verified_windows": len({row["window_51aa"] for row in verified}),
    }
    return summary, verified


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit candidate Kla study sites against published rice records and sequences.")
    parser.add_argument("--kladb-sites", type=Path, required=True)
    parser.add_argument("--rice-table", type=Path, required=True)
    parser.add_argument("--rice-proteome", type=Path, required=True)
    parser.add_argument("--gao-candidates", type=Path, required=True)
    parser.add_argument("--benchmark-train", type=Path, required=True)
    parser.add_argument("--benchmark-test", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--rice-positive-csv", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    inputs = {
        "kladb_sites": args.kladb_sites,
        "rice_table": args.rice_table,
        "rice_proteome": args.rice_proteome,
        "gao_candidates": args.gao_candidates,
        "benchmark_train": args.benchmark_train,
        "benchmark_test": args.benchmark_test,
    }
    curated = load_curated(args.kladb_sites)
    rice_audit, verified_rice = audit_rice(args.rice_table, curated["rice"], args.rice_proteome)
    with args.gao_candidates.open(newline="") as source:
        gao_windows = {row["window_51aa"] for row in csv.DictReader(source)}
    benchmark_windows = {
        row.sequence
        for split, path in (("train", args.benchmark_train), ("test", args.benchmark_test))
        for row in parse_pcbert_window_file(path, split)
    }
    full_windows = {
        name: {row["window_51"] for row in rows if len(row["window_51"]) == 51 and "*" not in row["window_51"]}
        for name, rows in curated.items()
    }
    report = {
        "inputs_sha256": {name: sha256(path) for name, path in inputs.items()},
        "curated_studies": {name: study_summary(rows) for name, rows in curated.items()},
        "rice_primary_sequence_audit": rice_audit,
        "exact_window_overlap": {
            name: {
                "gao_candidate_windows": len(windows & gao_windows),
                "original_benchmark_windows": len(windows & benchmark_windows),
            }
            for name, windows in full_windows.items()
        },
        "cross_study_exact_window_overlap": {
            f"{left}_vs_{right}": len(full_windows[left] & full_windows[right])
            for left, right in (("rice", "wheat"), ("rice", "soybean"), ("wheat", "soybean"))
        },
        "limitations": [
            "Wheat and soybean curated rows have not been checked against their original supplementary tables or a versioned proteome.",
            "Exact 51-aa window disjointness does not establish full-protein homology disjointness.",
            "Only experimentally reported rice positives are exported; no negative labels or external test set are established.",
        ],
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.rice_positive_csv.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    with args.rice_positive_csv.open("w", newline="") as target:
        writer = csv.DictWriter(
            target,
            fieldnames=["accession", "position_1based", "window_51aa", "label", "source_pmid"],
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(verified_rice)
    print(json.dumps({"rice_verified_positives": len(verified_rice), "output_json": str(args.output_json)}, indent=2))


if __name__ == "__main__":
    main()
