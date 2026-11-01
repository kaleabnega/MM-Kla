from __future__ import annotations

import argparse
import csv
import gzip
import json
from collections import Counter
from pathlib import Path

from audit_candidate_studies import audit_rice, sha256
from map_windows_to_proteome import parse_fasta, parse_pcbert_window_file
from validate_windows_against_modified_peptides import load_modified_peptides


SOURCE_PMID = "34264677"


def candidate_rows(
    proteins: dict[str, str],
    verified_positives: list[dict[str, str]],
    reported_sites: set[tuple[str, int]],
    peptide_supported_sites: set[tuple[str, int]],
    known_positive_windows: set[str],
    development_windows: set[str],
) -> tuple[list[dict[str, str]], dict[str, int]]:
    positive_sites = {(row["accession"], int(row["position_1based"])) for row in verified_positives}
    positive_windows = {row["window_51aa"] for row in verified_positives}
    if len(positive_sites) != len(verified_positives):
        raise ValueError("Duplicate verified positive accession-position site")
    if positive_windows & development_windows:
        raise ValueError("A verified rice positive window occurs in development data")

    candidates = [dict(row, evidence="reported_Kla_site") for row in verified_positives]
    counts: Counter[str] = Counter()
    for accession in sorted({row["accession"] for row in verified_positives}):
        sequence = proteins[accession]
        for offset, residue in enumerate(sequence):
            if residue != "K":
                continue
            position = offset + 1
            site = (accession, position)
            if site in reported_sites:
                continue
            if offset < 25 or offset + 25 >= len(sequence):
                counts["unreported_lysine_near_terminus"] += 1
                continue
            window = sequence[offset - 25 : offset + 26]
            if site in peptide_supported_sites:
                counts["excluded_modified_peptide_match"] += 1
            elif window in known_positive_windows or window in positive_windows:
                counts["excluded_known_positive_window"] += 1
            elif window in development_windows:
                counts["excluded_development_window"] += 1
            else:
                candidates.append({
                    "accession": accession,
                    "position_1based": str(position),
                    "window_51aa": window,
                    "label": "0",
                    "source_pmid": SOURCE_PMID,
                    "evidence": "unreported_lysine_putative_negative",
                })

    by_window: dict[str, set[str]] = {}
    for row in candidates:
        by_window.setdefault(row["window_51aa"], set()).add(row["label"])
    if any(len(labels) > 1 for labels in by_window.values()):
        raise ValueError("Conflicting labels remain for an identical rice window")
    counts.update({
        "verified_positive_sites": len(verified_positives),
        "putative_negative_sites": len(candidates) - len(verified_positives),
        "candidate_sites_total": len(candidates),
        "distinct_candidate_windows": len(by_window),
        "repeated_window_rows": len(candidates) - len(by_window),
        "candidate_proteins": len({row["accession"] for row in candidates}),
    })
    return candidates, dict(sorted(counts.items()))


def matched_modified_peptide_sites(
    source_table: Path,
    proteins: dict[str, str],
    target_accessions: set[str],
) -> set[tuple[str, int]]:
    peptides = load_modified_peptides(source_table, "Modified sequence")
    sites: set[tuple[str, int]] = set()
    for accession in target_accessions:
        sequence = proteins[accession]
        for peptide in peptides:
            start = sequence.find(peptide.plain_sequence)
            while start != -1:
                sites.add((accession, start + peptide.modified_index + 1))
                start = sequence.find(peptide.plain_sequence, start + 1)
    return sites


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build screened rice site candidates; negatives remain putative.")
    parser.add_argument("--rice-table", type=Path, required=True)
    parser.add_argument("--rice-proteome", type=Path, required=True)
    parser.add_argument("--kladb-sites", type=Path, required=True)
    parser.add_argument("--verified-positives", type=Path, required=True)
    parser.add_argument("--gao-candidates", type=Path, required=True)
    parser.add_argument("--benchmark-train", type=Path, required=True)
    parser.add_argument("--benchmark-test", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_paths = {
        "rice_table": args.rice_table,
        "rice_proteome": args.rice_proteome,
        "kladb_sites": args.kladb_sites,
        "verified_positives": args.verified_positives,
        "gao_candidates": args.gao_candidates,
        "benchmark_train": args.benchmark_train,
        "benchmark_test": args.benchmark_test,
    }
    with args.verified_positives.open(newline="") as source:
        verified_positives = list(csv.DictReader(source))
    curated_rice: list[dict[str, str]] = []
    known_positive_windows: set[str] = set()
    with gzip.open(args.kladb_sites, "rt", newline="") as source:
        for row in csv.DictReader(source):
            if len(row["window_51"]) == 51 and "*" not in row["window_51"]:
                known_positive_windows.add(row["window_51"])
            if row["species"] == "Oryza sativa" and SOURCE_PMID in row["pmids"].split(";"):
                curated_rice.append(row)
    with args.rice_table.open(newline="") as source:
        reported_sites = {
            (row["Protein accession"], int(float(row["Position"])))
            for row in csv.DictReader(source)
        }
    _, recomputed_positives = audit_rice(args.rice_table, curated_rice, args.rice_proteome)
    if recomputed_positives != verified_positives:
        raise ValueError("Verified-positive CSV does not match the primary table and dated proteome")
    proteins = {record.accession: record.sequence for record in parse_fasta(args.rice_proteome)}
    target_accessions = {row["accession"] for row in verified_positives}
    peptide_sites = matched_modified_peptide_sites(args.rice_table, proteins, target_accessions)
    with args.gao_candidates.open(newline="") as source:
        development_windows = {row["window_51aa"] for row in csv.DictReader(source)}
    for split, path in (("train", args.benchmark_train), ("test", args.benchmark_test)):
        development_windows.update(row.sequence for row in parse_pcbert_window_file(path, split))
    candidates, counts = candidate_rows(
        proteins, verified_positives, reported_sites, peptide_sites,
        known_positive_windows, development_windows,
    )
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="") as target:
        writer = csv.DictWriter(
            target,
            fieldnames=["accession", "position_1based", "window_51aa", "label", "source_pmid", "evidence"],
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(candidates)
    report = {
        "input_sha256": {name: sha256(path) for name, path in input_paths.items()},
        "candidate_csv_sha256": sha256(args.output_csv),
        "screened_peptide_supported_sites": len(peptide_sites),
        "known_positive_windows_screened": len(known_positive_windows),
        "development_windows_screened": len(development_windows),
        "counts": counts,
        "limitations": [
            "Negative labels mean unreported lysines, not experimentally verified absence of lactylation.",
            "No negative sampling, external test lock, or full-protein homology screen is performed by this command.",
        ],
    }
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(counts, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
