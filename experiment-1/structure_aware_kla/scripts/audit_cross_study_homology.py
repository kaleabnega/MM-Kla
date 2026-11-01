from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from audit_candidate_studies import sha256
from map_windows_to_proteome import parse_fasta


def accessions(path: Path) -> set[str]:
    with path.open(newline="") as source:
        return {row["accession"] for row in csv.DictReader(source)}


def read_hits(
    forward_path: Path,
    reverse_path: Path,
    gao_accessions: set[str],
    rice_accessions: set[str],
    min_identity_percent: float,
    min_coverage: float,
) -> dict[tuple[str, str], dict[str, object]]:
    hits: dict[tuple[str, str], dict[str, object]] = {}
    for direction, path in (("forward", forward_path), ("reverse", reverse_path)):
        with path.open(newline="") as source:
            for line_number, row in enumerate(csv.reader(source, delimiter="\t"), start=1):
                if len(row) != 8:
                    raise ValueError(f"{path}:{line_number}: expected eight MMseqs2 fields")
                query, target, identity, qcov, tcov, alnlen, qlen, tlen = row
                gao, rice = (query, target) if direction == "forward" else (target, query)
                if gao not in gao_accessions or rice not in rice_accessions:
                    raise ValueError(f"{path}:{line_number}: accession absent from candidate proteins")
                identity_value = float(identity)
                qcov_value = float(qcov)
                tcov_value = float(tcov)
                if identity_value + 1e-6 < min_identity_percent or min(qcov_value, tcov_value) + 1e-6 < min_coverage:
                    raise ValueError(f"{path}:{line_number}: alignment fails stated thresholds")
                if int(alnlen) > max(int(qlen), int(tlen)) or min(int(qlen), int(tlen)) < 1:
                    raise ValueError(f"{path}:{line_number}: invalid alignment lengths")
                key = (gao, rice)
                existing = hits.get(key)
                if existing is None:
                    existing = {
                        "gao_accession": gao,
                        "rice_accession": rice,
                        "max_identity_percent": identity_value,
                        "max_min_coverage": min(qcov_value, tcov_value),
                        "forward_seen": False,
                        "reverse_seen": False,
                    }
                    hits[key] = existing
                else:
                    existing["max_identity_percent"] = max(float(existing["max_identity_percent"]), identity_value)
                    existing["max_min_coverage"] = max(float(existing["max_min_coverage"]), min(qcov_value, tcov_value))
                existing[f"{direction}_seen"] = True
    return hits


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit Gao/rice full-protein homology and exclude rice homologues.")
    parser.add_argument("--gao-candidates", type=Path, required=True)
    parser.add_argument("--rice-candidates", type=Path, required=True)
    parser.add_argument("--gao-fasta", type=Path, required=True)
    parser.add_argument("--rice-fasta", type=Path, required=True)
    parser.add_argument("--forward-40", type=Path, required=True)
    parser.add_argument("--reverse-40", type=Path, required=True)
    parser.add_argument("--forward-80", type=Path, required=True)
    parser.add_argument("--reverse-80", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    inputs = {
        "gao_candidates": args.gao_candidates,
        "rice_candidates": args.rice_candidates,
        "gao_fasta": args.gao_fasta,
        "rice_fasta": args.rice_fasta,
        "forward_40": args.forward_40,
        "reverse_40": args.reverse_40,
        "forward_80": args.forward_80,
        "reverse_80": args.reverse_80,
    }
    gao_accessions = accessions(args.gao_candidates)
    rice_accessions = accessions(args.rice_candidates)
    if not gao_accessions or not rice_accessions:
        raise ValueError("Both source candidate files need accessions")
    for name, path, expected in (
        ("Gao", args.gao_fasta, gao_accessions),
        ("rice", args.rice_fasta, rice_accessions),
    ):
        records = parse_fasta(path)
        found = {record.accession for record in records}
        if len(found) != len(records) or found != expected:
            raise ValueError(
                f"{name} FASTA has {len(expected - found)} missing, "
                f"{len(found - expected)} extra, and {len(records) - len(found)} duplicate accessions"
            )
    with args.rice_candidates.open(newline="") as source:
        original_rows = list(csv.DictReader(source))
    if {row["accession"] for row in original_rows} != rice_accessions:
        raise ValueError("Rice candidate accession inventory changed")

    hits_40 = read_hits(args.forward_40, args.reverse_40, gao_accessions, rice_accessions, 30.0, 0.4)
    hits_80 = read_hits(args.forward_80, args.reverse_80, gao_accessions, rice_accessions, 30.0, 0.8)
    if not set(hits_80).issubset(hits_40):
        raise ValueError("80% coverage pairs must also occur in the 40% coverage search")
    excluded_rice = {rice for _, rice in hits_40}
    kept = [row for row in original_rows if row["accession"] not in excluded_rice]
    if {row["accession"] for row in kept} & excluded_rice:
        raise ValueError("A rice homolog remains after filtering")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    filtered_path = args.output_dir / "rice_homology_screened_candidates.csv"
    hits_path = args.output_dir / "gao_rice_30id_40cov_hits.csv"
    fields = list(original_rows[0])
    write_csv(filtered_path, kept, fields)
    hit_fields = ["gao_accession", "rice_accession", "max_identity_percent", "max_min_coverage", "forward_seen", "reverse_seen"]
    write_csv(hits_path, [hits_40[key] for key in sorted(hits_40)], hit_fields)

    before = Counter(row["label"] for row in original_rows)
    after = Counter(row["label"] for row in kept)
    report = {
        "rule": "MMseqs2 18-8cc5c+ds-2, identity >=30%, alignment coverage >=40% of both full proteins; union of both search directions",
        "input_sha256": {name: sha256(path) for name, path in inputs.items()},
        "output_sha256": {"filtered_candidates": sha256(filtered_path), "homology_hits": sha256(hits_path)},
        "candidate_proteins": {"gao": len(gao_accessions), "rice": len(rice_accessions)},
        "screen_80_percent_coverage": {
            "cross_pairs": len(hits_80),
            "gao_proteins": len({gao for gao, _ in hits_80}),
            "rice_proteins": len({rice for _, rice in hits_80}),
        },
        "screen_40_percent_coverage": {
            "cross_pairs": len(hits_40),
            "gao_proteins": len({gao for gao, _ in hits_40}),
            "rice_proteins_excluded": len(excluded_rice),
        },
        "site_counts_before": dict(sorted(before.items())),
        "site_counts_after": dict(sorted(after.items())),
        "rice_proteins_after": len({row["accession"] for row in kept}),
        "limitations": [
            "The protein search threshold cannot rule out more remote or shorter local/domain homology.",
            "Remaining rice negative labels are putative; this is not a locked external test or a performance result.",
            "The screen is against the Gao source-derived development proteins only, not arbitrary future training data.",
        ],
    }
    report_path = args.output_dir / "cross_study_homology_audit.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "rice_proteins_excluded": len(excluded_rice),
        "rice_proteins_remaining": report["rice_proteins_after"],
        "site_counts_after": report["site_counts_after"],
    }, indent=2))


if __name__ == "__main__":
    main()
