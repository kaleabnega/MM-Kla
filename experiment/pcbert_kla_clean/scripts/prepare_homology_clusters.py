from __future__ import annotations

import argparse
import csv
import gzip
from pathlib import Path


def candidate_accessions(mapping_csv: Path) -> set[str]:
    with mapping_csv.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if "accession" not in (reader.fieldnames or []):
            raise ValueError("Mapping CSV needs an accession column")
        return {row["accession"].strip() for row in reader if row["accession"].strip()}


def export_candidate_fasta(mapping_csv: Path, proteome_fasta: Path, output_fasta: Path) -> int:
    wanted = candidate_accessions(mapping_csv)
    if not wanted:
        raise ValueError("No candidate accessions in mapping CSV")
    open_input = gzip.open if proteome_fasta.suffix == ".gz" else open
    found: dict[str, list[str]] = {}
    with open_input(proteome_fasta, "rt") as source:
        accession = None
        for line in source:
            if line.startswith(">"):
                name = line[1:].split()[0]
                parsed = name.split("|")[1] if "|" in name else name
                accession = parsed if parsed in wanted else None
                if accession is not None:
                    if accession in found:
                        raise ValueError(f"Duplicate proteome accession {accession}")
                    found[accession] = []
            elif accession is not None:
                found[accession].append(line.strip())
    missing = wanted - set(found)
    if missing:
        raise ValueError(f"Proteome FASTA omits {len(missing)} candidate accessions")
    output_fasta.parent.mkdir(parents=True, exist_ok=True)
    with output_fasta.open("w") as output:
        for accession in sorted(found):
            output.write(f">{accession}\n{''.join(found[accession])}\n")
    return len(found)


def convert_cluster_tsv(mapping_csv: Path, cluster_tsv: Path, output_csv: Path) -> int:
    wanted = candidate_accessions(mapping_csv)
    assignments: dict[str, str] = {}
    with cluster_tsv.open(newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        for row in reader:
            if len(row) != 2 or not row[0] or not row[1]:
                raise ValueError("MMseqs2 cluster TSV must have representative and member columns")
            representative, member = row
            if member in assignments and assignments[member] != representative:
                raise ValueError(f"Conflicting cluster assignment for {member}")
            assignments[member] = representative
    if set(assignments) != wanted:
        raise ValueError(
            f"Cluster TSV has {len(wanted - set(assignments))} missing and "
            f"{len(set(assignments) - wanted)} extra candidate accessions"
        )
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["accession", "cluster_id"])
        writer.writerows((accession, assignments[accession]) for accession in sorted(wanted))
    return len(assignments)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare full-protein homology clusters for the review split")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("export-fasta", "convert-tsv"):
        subparser = subparsers.add_parser(command)
        subparser.add_argument("--mapping-csv", type=Path, required=True)
        if command == "export-fasta":
            subparser.add_argument("--proteome-fasta", type=Path, required=True)
            subparser.add_argument("--output-fasta", type=Path, required=True)
        else:
            subparser.add_argument("--cluster-tsv", type=Path, required=True)
            subparser.add_argument("--output-csv", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "export-fasta":
        count = export_candidate_fasta(args.mapping_csv, args.proteome_fasta, args.output_fasta)
    else:
        count = convert_cluster_tsv(args.mapping_csv, args.cluster_tsv, args.output_csv)
    print(f"Prepared {count} candidate proteins")


if __name__ == "__main__":
    main()
