from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from prepare_kladb_human import (
    AMINO_ACIDS, CANDIDATE_COLUMNS, load_evidence, load_proteins, sha256, window_at,
)


EXPECTED_COLUMNS = {"Protein accession", "Gene name", "Positions", "Sequence", "label"}


def prepare(
    source_csv: Path, proteome_fasta: Path, kladb_csv: Path, output_dir: Path
) -> dict[str, object]:
    proteins = load_proteins(proteome_fasta)
    known_positive_sites: set[tuple[str, int]] = set()
    known_positive_windows: set[str] = set()
    for record in load_evidence(kladb_csv):
        if record["species"] != "Homo sapiens":
            continue
        known_positive_sites.add((record["uniprot_id"].strip(), int(record["position"])))
        window = record["window_51"].strip().upper()
        if len(window) == 51 and window[25] == "K" and set(window) <= AMINO_ACIDS:
            known_positive_windows.add(window)

    counts: Counter[str] = Counter()
    retained: list[dict[str, object]] = []
    seen_sites: dict[tuple[str, int], int] = {}
    seen_windows: dict[str, int] = {}
    with source_csv.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if set(reader.fieldnames or ()) != EXPECTED_COLUMNS:
            raise ValueError(f"Unexpected CLEAR-Lactyl columns: {reader.fieldnames}")
        for record in reader:
            counts["source_rows"] += 1
            accession = record["Protein accession"].strip()
            try:
                position = int(record["Positions"])
                label = int(record["label"])
            except ValueError as error:
                raise ValueError(f"Invalid position or label: {record}") from error
            if label not in (0, 1) or position < 1 or not accession:
                raise ValueError(f"Invalid accession, position, or label: {record}")
            site = (accession, position)
            window = record["Sequence"].strip().upper().replace("<PAD>", "*")
            if site in seen_sites:
                if seen_sites[site] != label:
                    raise ValueError(f"Conflicting source labels at {site}")
                counts["duplicate_source_site"] += 1
                continue
            seen_sites[site] = label
            if len(window) != 51 or window[25] != "K":
                counts["invalid_window_length_or_center"] += 1
                continue
            if not set(window) <= AMINO_ACIDS:
                counts["padded_or_noncanonical_window"] += 1
                continue
            sequence = proteins.get(accession)
            if sequence is None:
                counts["accession_absent_from_reference"] += 1
                continue
            if position > len(sequence) or window_at(sequence, position) != window:
                counts["reference_position_or_window_mismatch"] += 1
                continue
            if label == 0 and (site in known_positive_sites or window in known_positive_windows):
                counts["negative_conflicts_with_KlaDB_positive"] += 1
                continue
            if window in seen_windows:
                if seen_windows[window] != label:
                    raise ValueError(f"Conflicting retained labels for window {window}")
                counts["duplicate_retained_window"] += 1
                continue
            seen_windows[window] = label
            retained.append({
                "accession": accession,
                "position_1based": position,
                "window_51aa": window,
                "label": label,
                "evidence": "reported_Kla_site" if label else "unreported_lysine_putative_negative",
                "source_pmids": "",
                "protein_source_pmids": "",
            })
            counts["retained_positive" if label else "retained_putative_negative"] += 1

    if not counts["retained_positive"] or not counts["retained_putative_negative"]:
        raise ValueError("Preparation lost one class")
    output_dir.mkdir(parents=True, exist_ok=True)
    candidates = output_dir / "candidate_sites.csv"
    with candidates.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CANDIDATE_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(sorted(retained, key=lambda row: (
            str(row["accession"]), int(row["position_1based"])
        )))
    fasta = output_dir / "candidate_proteins.fasta"
    with fasta.open("w") as handle:
        for accession in sorted({str(row["accession"]) for row in retained}):
            handle.write(f">{accession}\n{proteins[accession]}\n")
    report: dict[str, object] = {
        "protocol": "CLEAR-Lactyl reference-verified 51-aa canonical-window inventory",
        "counts": dict(sorted(counts.items())),
        "retained_proteins": len({str(row["accession"]) for row in retained}),
        "negative_definition": "Authors' unlabeled lysines; putative negatives, not verified non-Kla",
        "study_provenance": "Not present in the released CLEAR-Lactyl CSV",
        "source_sha256": {
            "clear_lactyl_csv": sha256(source_csv),
            "proteome_fasta": sha256(proteome_fasta),
            "kladb_human_csv": sha256(kladb_csv),
        },
        "output_sha256": {
            "candidate_sites.csv": sha256(candidates),
            "candidate_proteins.fasta": sha256(fasta),
        },
    }
    (output_dir / "preparation_audit.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare the published CLEAR-Lactyl site table")
    parser.add_argument("--source-csv", type=Path, required=True)
    parser.add_argument("--proteome-fasta", type=Path, required=True)
    parser.add_argument("--kladb-human-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.source_csv, args.proteome_fasta, args.kladb_human_csv, args.output_dir)["counts"], indent=2))


if __name__ == "__main__":
    main()
