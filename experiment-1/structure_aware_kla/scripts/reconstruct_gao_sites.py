from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

from audit_source_site_evidence import load_gao_sites, sha256
from map_windows_to_proteome import parse_fasta, parse_pcbert_window_file
from validate_windows_against_modified_peptides import parse_modified_sequence


SOURCE_DOI = "10.3389/fmicb.2020.594743"
WINDOW_RADIUS = 25


def site_window(sequence: str, position: int) -> str | None:
    if position <= WINDOW_RADIUS or position + WINDOW_RADIUS > len(sequence):
        return None
    if sequence[position - 1] != "K":
        raise ValueError(f"Position {position} is not lysine")
    return sequence[position - WINDOW_RADIUS - 1 : position + WINDOW_RADIUS]


def reconstruct(
    source_rows: list[dict[str, str]],
    proteins_by_accession: dict[str, str],
    benchmark_positive_windows: set[str],
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]], dict[str, int]]:
    reported_sites = {
        (row["Protein accession"].strip(), int(float(row["Position"])))
        for row in source_rows
    }
    if len(reported_sites) != len(source_rows):
        raise ValueError("Repeated accession-position rows in source table need manual review")

    evidence_rows: list[dict[str, object]] = []
    eligible_positives: list[dict[str, object]] = []
    annotated_windows = set(benchmark_positive_windows)
    for index, row in enumerate(source_rows, start=2):
        accession = row["Protein accession"].strip()
        position = int(float(row["Position"]))
        sequence = proteins_by_accession.get(accession)
        if row["Amino acid"].strip() != "K":
            raise ValueError(f"Source row {index} is not annotated as lysine")
        if sequence is None:
            status, window = "accession_missing_from_proteome", None
        elif not 1 <= position <= len(sequence) or sequence[position - 1] != "K":
            status, window = "position_not_lysine_in_proteome", None
        else:
            window = site_window(sequence, position)
            status = "window_available" if window else "site_near_protein_terminus"
        evidence_rows.append({
            "source_row": index,
            "accession": accession,
            "position_1based": position,
            "source_doi": SOURCE_DOI,
            "evidence": "reported_Kla_site",
            "reconstruction_status": status,
            "window_51aa": window or "",
        })
        if window:
            annotated_windows.add(window)
            eligible_positives.append({
                "accession": accession,
                "position_1based": position,
                "window_51aa": window,
                "label": 1,
                "evidence": "reported_Kla_site",
                "source_doi": SOURCE_DOI,
                "source_row": index,
            })

    candidate_rows = list(eligible_positives)
    excluded_rows: list[dict[str, object]] = []
    counts: Counter[str] = Counter()
    peptide_supported_sites: set[tuple[str, int]] = set()
    for index, row in enumerate(source_rows, start=2):
        peptide = parse_modified_sequence(row["Modified sequence"], index)
        if peptide is None:
            continue
        for accession in {accession for accession, _ in reported_sites}:
            sequence = proteins_by_accession.get(accession)
            if sequence is None:
                continue
            start = sequence.find(peptide.plain_sequence)
            while start != -1:
                peptide_supported_sites.add((accession, start + peptide.modified_index + 1))
                start = sequence.find(peptide.plain_sequence, start + 1)
    for accession in sorted({accession for accession, _ in reported_sites}):
        sequence = proteins_by_accession.get(accession)
        if sequence is None:
            continue
        for index, residue in enumerate(sequence):
            position = index + 1
            if residue != "K" or (accession, position) in reported_sites:
                continue
            window = site_window(sequence, position)
            if window is None:
                counts["unannotated_lysines_near_terminus"] += 1
                continue
            if (accession, position) in peptide_supported_sites:
                counts["unannotated_lysines_matching_modified_peptide"] += 1
                excluded_rows.append({
                    "accession": accession, "position_1based": position,
                    "window_51aa": window, "reason": "reported_modified_peptide_match",
                })
                continue
            if window in annotated_windows:
                counts["unannotated_lysines_matching_known_positive_window"] += 1
                excluded_rows.append({
                    "accession": accession, "position_1based": position,
                    "window_51aa": window, "reason": "benchmark_or_source_positive_window_match",
                })
                continue
            candidate_rows.append({
                "accession": accession,
                "position_1based": position,
                "window_51aa": window,
                "label": 0,
                "evidence": "unannotated_lysine_putative_negative",
                "source_doi": SOURCE_DOI,
                "source_row": "",
            })

    labels_by_window: dict[str, set[int]] = defaultdict(set)
    for row in candidate_rows:
        labels_by_window[str(row["window_51aa"])].add(int(row["label"]))
    if any(len(labels) > 1 for labels in labels_by_window.values()):
        raise ValueError("Contradictory labels remain for an identical sequence window")
    counts.update({
        "source_reported_sites": len(source_rows),
        "source_reported_proteins": len({a for a, _ in reported_sites}),
        "reconstructable_positive_sites": len(eligible_positives),
        "putative_negative_sites": len(candidate_rows) - len(eligible_positives),
        "candidate_sites_total": len(candidate_rows),
        "candidate_unique_windows": len(labels_by_window),
        "candidate_windows_repeated_across_sites": sum(
            n - 1 for n in Counter(str(row["window_51aa"]) for row in candidate_rows).values()
        ),
    })
    counts.update(Counter(str(row["reconstruction_status"]) for row in evidence_rows))
    return evidence_rows, candidate_rows, excluded_rows, dict(sorted(counts.items()))


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str] | None = None) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames or list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Reconstruct documented Gao Kla sites and provisional controls")
    parser.add_argument("--gao-xlsx", type=Path, required=True)
    parser.add_argument("--proteome-fasta", type=Path, required=True)
    parser.add_argument("--benchmark-train", type=Path, required=True)
    parser.add_argument("--benchmark-test", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    source_rows, _ = load_gao_sites(args.gao_xlsx)
    proteins = parse_fasta(args.proteome_fasta)
    proteins_by_accession = {record.accession: record.sequence for record in proteins}
    if len(proteins_by_accession) != len(proteins):
        raise ValueError("Proteome contains duplicate accessions")
    benchmark_positive_windows = {
        row.sequence
        for path, split in ((args.benchmark_train, "train"), (args.benchmark_test, "test"))
        for row in parse_pcbert_window_file(path, split)
        if row.label == 1
    }
    evidence, candidates, exclusions, counts = reconstruct(
        source_rows, proteins_by_accession, benchmark_positive_windows
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "reported_sites.csv", evidence)
    write_csv(args.output_dir / "candidate_sites.csv", candidates)
    write_csv(
        args.output_dir / "excluded_putative_negatives.csv",
        exclusions,
        ["accession", "position_1based", "window_51aa", "reason"],
    )
    report = {
        "source_doi": SOURCE_DOI,
        "window_length": 51,
        "input_sha256": {
            key: sha256(path)
            for key, path in (
                ("gao_xlsx", args.gao_xlsx),
                ("proteome_fasta", args.proteome_fasta),
                ("benchmark_train", args.benchmark_train),
                ("benchmark_test", args.benchmark_test),
            )
        },
        "counts": counts,
        "limitations": [
            "Only Gao-reported accession-position pairs are experimental positives.",
            "Unannotated lysines are putative negatives, not experimentally verified non-lactylated sites.",
            "The candidate sites have not been split or screened against independent Kla databases.",
            "This source overlaps the original benchmark and cannot serve as an unseen external test.",
        ],
    }
    (args.output_dir / "reconstruction_audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(counts, indent=2))


if __name__ == "__main__":
    main()
