"""Prepare reference-verified human S1 and mouse S7 Kla candidate sites.

S7 is the paper-designated mouse evaluation source. This script does not
establish independence: a subsequent full-protein homology lock is required.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path
from zipfile import ZipFile

from prepare_kladb_human import load_proteins, sha256, window_at


EXPECTED_SHA256 = {
    "s1": "1f088f64d38c6744ecb45acc11afcea6a1baf8f984c6d69de0b20a585b706c40",
    "s7": "97ae1f3364c5f4584a4f890395a5a2c1d5a203217e7db82b04e132bd466d7c8b",
    "human_fasta": "64299d01e8f41fb6ef03cdf7ec051d3d4e5f041003b6b0fa75b09d439af79dce",
    "mouse_fasta": "c24568f70ce9ffd7693a1ecd31c23eaa97a1f43d9305e2ff5401bdf30e2a0a93",
}
NAMESPACE = {"x": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
AA = frozenset("ACDEFGHIKLMNPQRSTVWY")
MAX_PROTEIN_LENGTH = 8192
SPECIES = {"Homo sapiens": "9606", "Mus musculus": "10090"}
COLUMNS = (
    "protein_id", "accession", "species", "position_1based", "window_51aa",
    "label", "evidence", "source_table", "partition_eligibility",
)


def read_sites(path: Path) -> list[dict[str, str]]:
    with ZipFile(path) as workbook:
        shared = []
        if "xl/sharedStrings.xml" in workbook.namelist():
            root = ET.fromstring(workbook.read("xl/sharedStrings.xml"))
            shared = ["".join(node.itertext()) for node in root.findall("x:si", NAMESPACE)]
        sheet = ET.fromstring(workbook.read("xl/worksheets/sheet1.xml"))
    rows = []
    for row in sheet.findall("x:sheetData/x:row", NAMESPACE):
        values = {}
        for cell in row.findall("x:c", NAMESPACE):
            match = re.match(r"[A-Z]+", cell.attrib["r"])
            if match is None:
                raise ValueError("Invalid XLSX cell address")
            column = match.group()
            if cell.attrib.get("t") == "inlineStr":
                node = cell.find("x:is", NAMESPACE)
                values[column] = "".join(node.itertext()) if node is not None else ""
            else:
                node = cell.find("x:v", NAMESPACE)
                value = node.text if node is not None and node.text else ""
                values[column] = shared[int(value)] if cell.attrib.get("t") == "s" else value
        rows.append(values)
    expected = ("UniProt ID", "Position", "Sequence", "Species")
    if len(rows) < 3 or tuple(rows[1].get(column) for column in "ABCD") != expected:
        raise ValueError(f"Unexpected HybridKla supplementary header in {path}")
    result = []
    for row in rows[2:]:
        values = [row.get(column, "").strip() for column in "ABCD"]
        if not all(values) or not values[1].isdigit():
            raise ValueError(f"Malformed supplementary site row: {values}")
        result.append(dict(zip(("accession", "position", "window", "species"), values)))
    return result


def prepare(
    s1_path: Path, s7_path: Path, human_fasta: Path, mouse_fasta: Path,
    output_dir: Path,
) -> dict:
    inputs = {"s1": s1_path, "s7": s7_path, "human_fasta": human_fasta,
              "mouse_fasta": mouse_fasta}
    hashes = {name: sha256(path) for name, path in inputs.items()}
    if hashes != EXPECTED_SHA256:
        raise ValueError("HybridKla supplement or reference proteome differs from the audited version")
    human = load_proteins(human_fasta)
    mouse = load_proteins(mouse_fasta)
    proteomes = {"Homo sapiens": human, "Mus musculus": mouse}
    source_rows = {"S1": read_sites(s1_path), "S7": read_sites(s7_path)}
    source_s1_windows = {
        row["window"] for row in source_rows["S1"]
        if len(row["window"]) == 51 and row["window"][25] == "K"
    }
    known_sites: dict[str, set[tuple[str, int]]] = defaultdict(set)
    known_windows: set[str] = set(source_s1_windows)
    usable: dict[str, list[dict[str, str]]] = {"S1": [], "S7": []}
    exclusions: Counter[str] = Counter()
    seen_site: dict[tuple[str, str, int], str] = {}
    for source, rows in source_rows.items():
        for row in rows:
            species = row["species"]
            if source == "S1" and species not in SPECIES:
                continue
            if source == "S7" and species != "Mus musculus":
                raise ValueError("HybridKla Table S7 contains a non-mouse site")
            accession = row["accession"]
            position = int(row["position"])
            window = row["window"]
            if len(window) != 51 or window[25] != "K":
                raise ValueError(f"Invalid source positive window: {accession}_{position}")
            key = (species, accession, position)
            if key in seen_site:
                if seen_site[key] != window:
                    raise ValueError(f"Conflicting positive windows for {key}")
                exclusions[f"{source}_repeated_site"] += 1
                continue
            seen_site[key] = window
            known_sites[species].add((accession, position))
            known_windows.add(window)
            sequence = proteomes[species].get(accession)
            if sequence is None:
                exclusions[f"{source}_{SPECIES[species]}_missing_accession"] += 1
                continue
            if position > len(sequence):
                exclusions[f"{source}_{SPECIES[species]}_outside_protein"] += 1
                continue
            if len(sequence) > MAX_PROTEIN_LENGTH:
                exclusions[f"{source}_{SPECIES[species]}_protein_over_8192aa"] += 1
                continue
            if window != window_at(sequence, position):
                exclusions[f"{source}_{SPECIES[species]}_window_mismatch"] += 1
                continue
            if set(window) - AA:
                exclusions[f"{source}_{SPECIES[species]}_noncanonical"] += 1
                continue
            if source == "S1" and species == "Mus musculus":
                continue  # S1 mouse sites protect negatives but are not S7 test positives.
            usable[source].append({
                "protein_id": f"{SPECIES[species]}:{accession}",
                "accession": accession,
                "species": species,
                "position_1based": str(position),
                "window_51aa": window,
                "label": "1",
                "evidence": "published_reported_Kla_site",
                "source_table": source,
                "partition_eligibility": "development" if source == "S1" else "external_candidate",
            })

    if not usable["S1"] or not usable["S7"]:
        raise ValueError("Reference verification left an empty positive source")
    positive_windows = {row["window_51aa"] for rows in usable.values() for row in rows}
    development_windows = source_s1_windows
    development_proteins = {row["protein_id"] for row in usable["S1"]}
    positive_rows = []
    for source in ("S1", "S7"):
        for row in usable[source]:
            if source == "S7" and (row["window_51aa"] in development_windows
                                   or row["protein_id"] in development_proteins):
                exclusions["S7_exact_window_or_protein_overlap_with_S1"] += 1
                continue
            positive_rows.append(row)
    by_protein = {row["protein_id"]: (row["species"], row["accession"])
                  for row in positive_rows}
    negative_rows = []
    for protein_id, (species, accession) in sorted(by_protein.items()):
        sequence = proteomes[species][accession]
        source = "S1" if species == "Homo sapiens" else "S7"
        for index in range(25, len(sequence) - 25):
            if sequence[index] != "K" or (accession, index + 1) in known_sites[species]:
                continue
            window = sequence[index - 25:index + 26]
            if set(window) - AA or window in known_windows or window in positive_windows:
                continue
            negative_rows.append({
                "protein_id": protein_id, "accession": accession, "species": species,
                "position_1based": str(index + 1), "window_51aa": window,
                "label": "0", "evidence": "unreported_lysine_putative_negative",
                "source_table": source,
                "partition_eligibility": "development" if source == "S1" else "external_candidate",
            })
    candidates = sorted(positive_rows + negative_rows,
                        key=lambda row: (row["protein_id"], int(row["position_1based"])))
    labels_by_window: dict[str, set[str]] = defaultdict(set)
    for row in candidates:
        labels_by_window[row["window_51aa"]].add(row["label"])
    if any(len(labels) > 1 for labels in labels_by_window.values()):
        raise ValueError("Positive and putative negative candidates share a window")
    output_dir.mkdir(parents=True, exist_ok=True)
    candidate_path = output_dir / "candidate_sites.csv"
    with candidate_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(candidates)
    fasta_path = output_dir / "candidate_proteins.fasta"
    with fasta_path.open("w") as handle:
        for protein_id, (species, accession) in sorted(by_protein.items()):
            handle.write(f">{protein_id}\n{proteomes[species][accession]}\n")
    audit = {
        "status": "candidate inventory only; homology and external independence not yet established",
        "source": "HybridKla published supplementary Tables S1 and S7",
        "source_url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC12309240/",
        "source_rows": {name: len(rows) for name, rows in source_rows.items()},
        "reference_verified_positive_rows": {name: len(rows) for name, rows in usable.items()},
        "candidate_counts": {
            source: {
                "positives": sum(r["source_table"] == source and r["label"] == "1" for r in candidates),
                "putative_negatives": sum(r["source_table"] == source and r["label"] == "0" for r in candidates),
                "proteins": len({r["protein_id"] for r in candidates if r["source_table"] == source}),
            } for source in ("S1", "S7")
        },
        "exclusions": dict(sorted(exclusions.items())),
        "input_sha256": hashes,
        "output_sha256": {"candidate_sites.csv": sha256(candidate_path),
                          "candidate_proteins.fasta": sha256(fasta_path)},
        "negative_definition": "Canonical unreported lysines on positive-bearing proteins; not experimentally verified negatives",
        "limitations": [
            "Table S1 gives no per-site PMID; its human positives are curated source records, not study-labelled.",
            "Table S7 is paper-designated mouse test data; its study independence has not been independently verified.",
            "Full-protein homology screening and partition locking remain mandatory.",
            "Reference-proteome mismatch or missing accession rows are excluded, not reassigned.",
            "Proteins longer than 8192 residues are excluded because the pinned MMseqs2 search omitted their self-hits.",
        ],
    }
    (output_dir / "preparation_audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--s1", type=Path, required=True)
    parser.add_argument("--s7", type=Path, required=True)
    parser.add_argument("--human-fasta", type=Path, required=True)
    parser.add_argument("--mouse-fasta", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.s1, args.s7, args.human_fasta, args.mouse_fasta,
                             args.output_dir), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
