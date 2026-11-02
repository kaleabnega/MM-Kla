from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


AMINO_ACIDS = frozenset("ACDEFGHIKLMNPQRSTVWY")
EXPECTED_COLUMNS = {"species", "uniprot_id", "position", "window_51", "pmid"}
CANDIDATE_COLUMNS = (
    "accession", "position_1based", "window_51aa", "label", "evidence",
    "source_pmids", "protein_source_pmids",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def open_text(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8-sig", newline="")
    return path.open("r", encoding="utf-8-sig", newline="")


def load_proteins(path: Path) -> dict[str, str]:
    proteins: dict[str, str] = {}
    accession = None
    with open_text(path) as handle:
        for line in handle:
            if line.startswith(">"):
                identifier = line[1:].split()[0]
                accession = identifier.split("|")[1] if "|" in identifier else identifier
                if not accession or accession in proteins:
                    raise ValueError(f"Duplicate or empty FASTA accession: {accession!r}")
                proteins[accession] = ""
            elif line.strip():
                if accession is None:
                    raise ValueError("FASTA sequence precedes its header")
                proteins[accession] += line.strip().upper()
    return proteins


def load_evidence(path: Path) -> list[dict[str, str]]:
    with open_text(path) as handle:
        reader = csv.DictReader(handle)
        if not EXPECTED_COLUMNS.issubset(reader.fieldnames or []):
            raise ValueError(f"KlaDB table needs {sorted(EXPECTED_COLUMNS)}")
        rows = list(reader)
    if not rows:
        raise ValueError("KlaDB table has no rows")
    return rows


def window_at(sequence: str, position: int) -> str:
    start = position - 26
    end = position + 25
    return (
        "*" * max(0, -start)
        + sequence[max(0, start):min(len(sequence), end)]
        + "*" * max(0, end - len(sequence))
    )


def prepare_candidates(
    evidence: list[dict[str, str]], proteins: dict[str, str]
) -> tuple[list[dict[str, object]], dict[str, object], set[str]]:
    sites: dict[tuple[str, int], dict[str, object]] = {}
    for row in evidence:
        if row["species"].strip() != "Homo sapiens":
            raise ValueError("Input includes a non-human record")
        accession = row["uniprot_id"].strip()
        pmid = row["pmid"].strip()
        if not accession or not pmid:
            raise ValueError("KlaDB record lacks accession or PMID")
        try:
            position = int(row["position"])
        except ValueError as error:
            raise ValueError(f"Invalid position: {row['position']!r}") from error
        if position < 1:
            raise ValueError(f"Invalid position: {position}")
        window = row["window_51"].strip().upper()
        key = (accession, position)
        if key not in sites:
            sites[key] = {"window": window, "pmids": set()}
        if sites[key]["window"] != window:
            raise ValueError(f"Conflicting windows for site {key}")
        sites[key]["pmids"].add(pmid)

    protein_studies: dict[str, set[str]] = defaultdict(set)
    all_reported_windows: set[str] = set()
    for (accession, _), detail in sites.items():
        protein_studies[accession].update(detail["pmids"])
        window = str(detail["window"])
        if len(window) == 51 and window[25] == "K" and set(window) <= AMINO_ACIDS:
            all_reported_windows.add(window)

    exclusions: Counter[str] = Counter()
    positive_rows: list[dict[str, object]] = []
    used_proteins: set[str] = set()
    for (accession, position), detail in sorted(sites.items()):
        sequence = proteins.get(accession)
        if sequence is None:
            exclusions["accession_not_in_proteome"] += 1
            continue
        if position > len(sequence):
            exclusions["position_outside_protein"] += 1
            continue
        window = str(detail["window"])
        if window != window_at(sequence, position):
            raise ValueError(f"KlaDB window disagrees with proteome at {(accession, position)}")
        if len(window) != 51 or window[25] != "K":
            exclusions["invalid_center_or_length"] += 1
            continue
        if not set(window) <= AMINO_ACIDS:
            exclusions["terminal_or_noncanonical_window"] += 1
            continue
        used_proteins.add(accession)
        positive_rows.append({
            "accession": accession,
            "position_1based": position,
            "window_51aa": window,
            "label": 1,
            "evidence": "reported_Kla_site",
            "source_pmids": ";".join(sorted(detail["pmids"])),
            "protein_source_pmids": ";".join(sorted(protein_studies[accession])),
        })

    positive_windows = {str(row["window_51aa"]) for row in positive_rows}
    negative_rows: list[dict[str, object]] = []
    for accession in sorted(used_proteins):
        sequence = proteins[accession]
        for index in range(25, len(sequence) - 25):
            if sequence[index] != "K" or (accession, index + 1) in sites:
                continue
            window = sequence[index - 25:index + 26]
            if not set(window) <= AMINO_ACIDS:
                continue
            if window in all_reported_windows:
                exclusions["negative_matches_positive_window"] += 1
                continue
            negative_rows.append({
                "accession": accession,
                "position_1based": index + 1,
                "window_51aa": window,
                "label": 0,
                "evidence": "unreported_lysine_putative_negative",
                "source_pmids": "",
                "protein_source_pmids": ";".join(sorted(protein_studies[accession])),
            })
    rows = sorted(positive_rows + negative_rows, key=lambda row: (
        str(row["accession"]), int(row["position_1based"])
    ))
    report = {
        "protocol": "KlaDB human site candidate inventory; not a train/test split",
        "evidence_rows": len(evidence),
        "unique_reported_sites": len(sites),
        "usable_positive_sites": len(positive_rows),
        "usable_positive_windows": len(positive_windows),
        "putative_negative_sites": len(negative_rows),
        "positive_bearing_proteins": len(used_proteins),
        "exclusions": dict(sorted(exclusions.items())),
        "study_positive_sites": dict(sorted(Counter(
            pmid for row in positive_rows for pmid in str(row["source_pmids"]).split(";")
        ).items())),
        "negative_definition": "Complete-window lysines in positive-bearing proteins, absent from all KlaDB reported sites for the same accession; not experimentally confirmed negatives",
        "limitations": [
            "The candidate inventory is not protein- or homology-partitioned.",
            "A study-held-out test and development split must be locked before model selection.",
            "KlaDB unreported lysines may contain undiscovered Kla sites.",
            "Exact windows may recur at different protein sites; split locking must remove cross-partition overlap.",
        ],
    }
    return rows, report, used_proteins


def write_inventory(
    rows: list[dict[str, object]], report: dict[str, object], proteins: dict[str, str],
    used_proteins: set[str], output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    candidates_path = output_dir / "candidate_sites.csv"
    with candidates_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CANDIDATE_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    fasta_path = output_dir / "candidate_proteins.fasta"
    with fasta_path.open("w", newline="") as handle:
        for accession in sorted(used_proteins):
            handle.write(f">{accession}\n{proteins[accession]}\n")
    report["output_sha256"] = {
        "candidate_sites.csv": sha256(candidates_path),
        "candidate_proteins.fasta": sha256(fasta_path),
    }
    (output_dir / "preparation_audit.json").write_text(json.dumps(report, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit and prepare KlaDB human site candidates")
    parser.add_argument("--site-csv", type=Path, required=True)
    parser.add_argument("--proteome-fasta", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    evidence = load_evidence(args.site_csv)
    proteins = load_proteins(args.proteome_fasta)
    rows, report, used_proteins = prepare_candidates(evidence, proteins)
    report["source_sha256"] = {
        "site_csv": sha256(args.site_csv),
        "proteome_fasta": sha256(args.proteome_fasta),
    }
    report["proteome_entries"] = len(proteins)
    write_inventory(rows, report, proteins, used_proteins, args.output_dir)
    print(json.dumps({key: report[key] for key in (
        "unique_reported_sites", "usable_positive_sites", "putative_negative_sites",
        "positive_bearing_proteins", "exclusions"
    )}, indent=2))


if __name__ == "__main__":
    main()
