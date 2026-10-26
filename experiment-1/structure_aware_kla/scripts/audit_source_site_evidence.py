from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from zipfile import ZipFile

from audit_dataset_provenance import multiset_summary, parse_deepkla_fasta
from extract_xlsx_table import read_sheet, sheet_paths
from map_windows_to_proteome import map_windows, parse_fasta, parse_pcbert_window_file
from validate_windows_against_modified_peptides import (
    matching_peptides,
    parse_modified_sequence,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_gao_sites(path: Path) -> tuple[list[dict[str, str]], set[tuple[str, int]]]:
    with ZipFile(path) as workbook:
        sheets = sheet_paths(workbook)
        rows = read_sheet(workbook, next(iter(sheets.values())))
    required = {"Protein accession", "Position", "Amino acid", "Modified sequence"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError("Gao supplement is missing expected site columns")
    headers = rows[0]
    records = [dict(zip(headers, row)) for row in rows[1:] if row]
    sites: set[tuple[str, int]] = set()
    for row in records:
        accession = row["Protein accession"].strip()
        position = int(float(row["Position"]))
        if not accession or row["Amino acid"].strip() != "K":
            raise ValueError("Gao supplement contains a non-lysine or unidentified site")
        sites.add((accession, position))
    return records, sites


def source_site_window_summary(
    source_sites: set[tuple[str, int]],
    proteins_by_accession: dict[str, str],
    train_windows: set[str],
    test_windows: set[str],
) -> dict[str, int]:
    counts: Counter[str] = Counter()
    reconstructed: set[str] = set()
    for accession, position in source_sites:
        sequence = proteins_by_accession.get(accession)
        if sequence is None:
            counts["accession_not_in_proteome"] += 1
        elif not 1 <= position <= len(sequence) or sequence[position - 1] != "K":
            counts["site_not_a_lysine_at_reported_position"] += 1
        elif position < 26 or position + 25 > len(sequence):
            counts["site_too_close_to_terminus_for_51aa_window"] += 1
        else:
            counts["reconstructed_sites"] += 1
            reconstructed.add(sequence[position - 26 : position + 25])
    return {
        **{key: counts[key] for key in (
            "accession_not_in_proteome",
            "site_not_a_lysine_at_reported_position",
            "site_too_close_to_terminus_for_51aa_window",
            "reconstructed_sites",
        )},
        "unique_reconstructed_windows": len(reconstructed),
        "windows_in_original_train": len(reconstructed & train_windows),
        "windows_in_original_test": len(reconstructed & test_windows),
    }


def audit_records(windows, matches, source_sites, source_rows):
    by_record = defaultdict(list)
    for row in matches:
        by_record[(row["split"], row["record_name"])].append(row)

    details = []
    counts: Counter[str] = Counter()
    unique_status: dict[tuple[str, int, str], str] = {}
    peptides = [
        peptide
        for index, row in enumerate(source_rows)
        if (peptide := parse_modified_sequence(row["Modified sequence"], index)) is not None
    ]
    for record in windows:
        key = (record.split, record.record_name)
        candidate_sites = {
            (str(row["accession"]), int(row["central_lysine_full_position_1based"]))
            for row in by_record[key]
            if row["accession"]
        }
        supported = candidate_sites & source_sites
        peptide_supported = {
            (
                source_rows[peptide.source_row]["Protein accession"].strip(),
                int(float(source_rows[peptide.source_row]["Position"])),
            )
            for peptide in matching_peptides(record, peptides)
        }
        if not candidate_sites:
            status = "no_proteome_match"
        elif len(supported) == 1:
            status = "one_source_site_match"
        elif supported:
            status = "multiple_source_site_matches"
        elif peptide_supported:
            status = "peptide_match_only"
        else:
            status = "no_source_site_match"
        counts[f"{record.split}_label_{record.label}_{status}"] += 1
        unique_status[(record.split, record.label, record.sequence)] = status
        details.append({
            "split": record.split,
            "record_name": record.record_name,
            "label": record.label,
            "window_sequence": record.sequence,
            "candidate_site_count": len(candidate_sites),
            "candidate_sites": ";".join(f"{accession}:K{position}" for accession, position in sorted(candidate_sites)),
            "source_site_match_count": len(supported),
            "source_site_matches": ";".join(f"{accession}:K{position}" for accession, position in sorted(supported)),
            "source_peptide_site_matches": ";".join(
                f"{accession}:K{position}" for accession, position in sorted(peptide_supported)
            ),
            "evidence_status": status,
        })
    unique_counts = Counter(
        f"{split}_label_{label}_{status}"
        for (split, label, _), status in unique_status.items()
    )
    return details, dict(sorted(counts.items())), dict(sorted(unique_counts.items()))


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def compare_previous_mapping(path: Path, details: list[dict]) -> dict[str, int]:
    with path.open(newline="") as handle:
        previous = {
            (row["original_split"], row["record_name"]): row["candidate_sites"]
            for row in csv.DictReader(handle)
        }
    current = {(row["split"], row["record_name"]): row["candidate_sites"] for row in details}
    if previous.keys() != current.keys():
        raise ValueError("Previous mapping does not cover the same benchmark records")
    return {
        "records_compared": len(current),
        "identical_candidate_site_assignments": sum(previous[key] == value for key, value in current.items()),
        "changed_candidate_site_assignments": sum(previous[key] != value for key, value in current.items()),
    }


def write_markdown_report(path: Path, report: dict) -> None:
    rows = report["benchmark_unique_window_evidence_by_split_and_label"]
    overlap = report["exact_gao_positive_site_overlap"]
    mapping = report.get("previous_mapping_comparison")
    lines = [
        "# DeepKla Source-Site Audit",
        "",
        "## Evidence",
        "",
        "- The original DeepKla and PCBert-Kla train/test files have identical sequence-label multisets.",
        "- DeepKla headers contain only a class label; PCBert-Kla headers use generic row names. Neither provides a verified parent protein accession.",
        f"- Gao et al. report {report['gao_unique_reported_sites']} distinct Botrytis Kla sites in their supplementary table.",
        "- Exact matches below require both a full-protein accession and a lysine position from that table.",
        "",
        "| Benchmark subset | Exact Gao site | Modified peptide only | No Gao site/peptide match |",
        "|---|---:|---:|---:|",
    ]
    for split, label, title in (
        ("train", 1, "Training positives (distinct windows)"),
        ("test", 1, "Test positives"),
        ("test", 0, "Test negatives"),
    ):
        prefix = f"{split}_label_{label}_"
        lines.append(
            f"| {title} | {rows.get(prefix + 'one_source_site_match', 0)} | "
            f"{rows.get(prefix + 'peptide_match_only', 0)} | "
            f"{rows.get(prefix + 'no_source_site_match', 0)} |"
        )
    lines.extend([
        "",
        f"Of the exact Gao-positive site assignments, {overlap['sites_in_both_splits']} "
        f"occur in both original partitions; only {overlap['test_only_sites']} exact Gao sites occur "
        "in the test partition without appearing in training.",
        "The four test negatives with an exact Gao site match are label-discordant. "
        "One further test negative shares a modified peptide with a Gao site but lacks a current-accession match.",
        "",
        "Three distinct positive windows have peptide-only evidence. Their current proteome "
        "accessions differ from the Gao table accessions; this is not an independently verified accession crosswalk.",
        "",
        "## Mapping Reproduction",
        "",
        f"- Dated proteome records searched: {report['proteome_proteins']}.",
    ])
    if mapping is not None:
        lines.append(
            f"- Candidate-site assignments unchanged from the earlier mapping: "
            f"{mapping['identical_candidate_site_assignments']}/{mapping['records_compared']}."
        )
    lines.extend([
        "- Input paths, SHA-256 checksums, and the DeepKla repository commit are recorded in `source_site_audit.json`.",
        "- Record-level candidate and evidence fields are in `benchmark_site_evidence.csv`.",
        "",
        "## Interpretation",
        "",
        "The positive benchmark windows are strongly consistent with the Gao Botrytis lactylome, "
        "including the nominal training positives. This conflicts with the published description "
        "of rice-derived training data and requires clarification from the dataset authors. "
        "An exact window or peptide match does not prove that every benchmark row originated "
        "from the matched protein. In particular, negative rows have no source accession or "
        "experimentally verified non-lactylated label. This audit is not a protein-disjoint split.",
        "",
        "[DeepKla source](https://github.com/linDing-group/DeepKla) | "
        "[Gao et al. 2020](https://doi.org/10.3389/fmicb.2020.594743) | "
        "[PCBert-Kla source](https://github.com/ZhangHongqi215/PCBert-Kla)",
        "",
    ])
    path.write_text("\n".join(lines))


def run_audit(args: argparse.Namespace) -> dict:
    train = parse_pcbert_window_file(args.pcbert_train, "train")
    test = parse_pcbert_window_file(args.pcbert_test, "test")
    original_train = parse_deepkla_fasta(args.deepkla_train, "train")
    original_test = parse_deepkla_fasta(args.deepkla_test, "test")
    comparisons = {
        "train": multiset_summary(train, original_train),
        "test": multiset_summary(test, original_test),
    }
    if not all(item["sequence_label_multiset_equal"] for item in comparisons.values()):
        raise ValueError("DeepKla and PCBert-Kla sequence-label records differ")

    proteins = parse_fasta(args.proteome_fasta)
    proteins_by_accession = {record.accession: record.sequence for record in proteins}
    source_rows, source_sites = load_gao_sites(args.gao_xlsx)
    windows = [*train, *test]
    matches, _ = map_windows(windows, proteins)
    details, row_counts, unique_counts = audit_records(windows, matches, source_sites, source_rows)
    train_source_sites = {
        row["source_site_matches"]
        for row in details if row["split"] == "train" and row["label"] == 1 and row["source_site_matches"]
    }
    test_source_sites = {
        row["source_site_matches"]
        for row in details if row["split"] == "test" and row["label"] == 1 and row["source_site_matches"]
    }

    inputs = {
        "pcbert_train": args.pcbert_train,
        "pcbert_test": args.pcbert_test,
        "deepkla_train": args.deepkla_train,
        "deepkla_test": args.deepkla_test,
        "proteome_fasta": args.proteome_fasta,
        "gao_xlsx": args.gao_xlsx,
    }
    report = {
        "input_files": {name: {"path": str(path), "sha256": sha256(path)} for name, path in inputs.items()},
        "deepkla_commit": args.deepkla_commit,
        "deepkla_vs_pcbert": comparisons,
        "proteome_proteins": len(proteins),
        "gao_table_rows": len(source_rows),
        "gao_unique_reported_sites": len(source_sites),
        "gao_site_window_reconstruction": source_site_window_summary(
            source_sites,
            proteins_by_accession,
            {record.sequence for record in train},
            {record.sequence for record in test},
        ),
        "exact_gao_positive_site_overlap": {
            "train_sites": len(train_source_sites),
            "test_sites": len(test_source_sites),
            "sites_in_both_splits": len(train_source_sites & test_source_sites),
            "test_only_sites": len(test_source_sites - train_source_sites),
        },
        "benchmark_row_evidence": row_counts,
        "benchmark_unique_window_evidence_by_split_and_label": unique_counts,
        "interpretation": (
            "An exact accession-and-position match to Gao et al. is evidence for a Kla site, "
            "but sequence-only mapping cannot establish the original source protein of every "
            "benchmark row. A modified-peptide-only match is weaker evidence and cannot verify "
            "an accession crosswalk. An unmatched positive may have another source. A negative "
            "matching a reported Kla site is label-discordant and should not be treated as a verified negative."
        ),
    }
    if args.previous_mapping_csv is not None:
        report["previous_mapping_comparison"] = compare_previous_mapping(
            args.previous_mapping_csv, details
        )
        report["input_files"]["previous_mapping_csv"] = {
            "path": str(args.previous_mapping_csv), "sha256": sha256(args.previous_mapping_csv)
        }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "benchmark_site_evidence.csv", details)
    (args.output_dir / "source_site_audit.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    write_markdown_report(args.output_dir / "source_site_audit.md", report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit DeepKla windows against Gao 2020 sites")
    parser.add_argument("--pcbert-train", type=Path, required=True)
    parser.add_argument("--pcbert-test", type=Path, required=True)
    parser.add_argument("--deepkla-train", type=Path, required=True)
    parser.add_argument("--deepkla-test", type=Path, required=True)
    parser.add_argument("--deepkla-commit", default="unknown")
    parser.add_argument("--proteome-fasta", type=Path, required=True)
    parser.add_argument("--gao-xlsx", type=Path, required=True)
    parser.add_argument("--previous-mapping-csv", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    report = run_audit(parse_args())
    print(json.dumps({
        "deepkla_vs_pcbert": report["deepkla_vs_pcbert"],
        "gao_site_window_reconstruction": report["gao_site_window_reconstruction"],
        "benchmark_row_evidence": report["benchmark_row_evidence"],
    }, indent=2, sort_keys=True))
