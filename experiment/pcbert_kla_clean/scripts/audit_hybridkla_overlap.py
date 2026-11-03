"""Audit exact sequence-window overlap with the HybridKla training workbook."""

import argparse
import csv
import hashlib
import json
import re
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path
from zipfile import ZipFile


PARTITIONS = ("train", "validation", "internal_test", "external_test")
NS = {"x": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_training_workbook(path: Path) -> tuple[dict[str, set[str]], dict, set[str], set[str]]:
    with ZipFile(path) as workbook:
        shared = []
        if "xl/sharedStrings.xml" in workbook.namelist():
            root = ET.fromstring(workbook.read("xl/sharedStrings.xml"))
            shared = ["".join(node.itertext()) for node in root.findall("x:si", NS)]
        sheet = ET.fromstring(workbook.read("xl/worksheets/sheet1.xml"))

    def row_values(row):
        values = {}
        for cell in row.findall("x:c", NS):
            match = re.match(r"[A-Z]+", cell.attrib["r"])
            if match is None:
                raise ValueError("Invalid workbook cell reference")
            if cell.attrib.get("t") == "inlineStr":
                node = cell.find("x:is", NS)
                value = "".join(node.itertext()) if node is not None else ""
            else:
                node = cell.find("x:v", NS)
                value = node.text if node is not None and node.text is not None else ""
                if cell.attrib.get("t") == "s":
                    value = shared[int(value)]
            values[match.group()] = value
        return values

    rows = sheet.findall("x:sheetData/x:row", NS)
    if not rows:
        raise ValueError("HybridKla workbook has no rows")
    header = {value: column for column, value in row_values(rows[0]).items()}
    if not {"UniProt_ID", "Sequence", "Label"}.issubset(header):
        raise ValueError("HybridKla workbook has unexpected columns")

    sequences = defaultdict(set)
    generic_ids = 0
    site_ids = set()
    accessions = set()
    for row in rows[1:]:
        values = row_values(row)
        sequence = values.get(header["Sequence"], "")
        label = values.get(header["Label"], "")
        source_id = values.get(header["UniProt_ID"], "")
        if len(sequence) != 51 or sequence[25] != "K" or label not in {"0", "1"}:
            raise ValueError(f"Invalid sequence or label at workbook row {row.attrib.get('r')}")
        sequences[sequence].add(label)
        generic_ids += bool(re.fullmatch(r"train\d+", source_id))
        site_match = re.fullmatch(r"([^_]+)_([1-9]\d*)", source_id)
        if site_match:
            site_ids.add(source_id)
            accessions.add(site_match.group(1))
    return dict(sequences), {
        "rows": len(rows) - 1,
        "unique_windows": len(sequences),
        "conflicting_window_labels": sum(len(labels) > 1 for labels in sequences.values()),
        "generic_train_ids": generic_ids,
        "accession_position_ids": len(site_ids),
        "accession_like_prefixes": len(accessions),
    }, site_ids, accessions


def audit(workbook: Path, split_dir: Path) -> dict:
    source, source_summary, source_sites, source_accessions = load_training_workbook(workbook)
    partitions = {}
    hashes = {"hybridkla_train_data.xlsx": sha256(workbook)}
    for partition in PARTITIONS:
        path = split_dir / f"{partition}.csv"
        hashes[path.name] = sha256(path)
        with path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            required = {"Site_ID", "ACC_ID", "Sequence", "Label"}
            if reader.fieldnames is None or not required.issubset(reader.fieldnames):
                raise ValueError(f"Missing required columns in {path}")
            rows = list(reader)
        matched = [row for row in rows if row["Sequence"] in source]
        matching_accessions = {row["ACC_ID"] for row in rows if row["ACC_ID"] in source_accessions}
        partitions[partition] = {
            "rows": len(rows),
            "exact_window_overlap": len(matched),
            "exact_site_id_overlap": sum(row["Site_ID"] in source_sites for row in rows),
            "rows_on_matching_accessions": sum(row["ACC_ID"] in source_accessions for row in rows),
            "distinct_matching_accessions": len(matching_accessions),
            "overlap_positive_in_locked_split": sum(row["Label"] == "1" for row in matched),
            "matched_same_label": sum(row["Label"] in source[row["Sequence"]] for row in matched),
            "matched_opposite_label": sum(row["Label"] not in source[row["Sequence"]] for row in matched),
        }
    return {
        "source": source_summary,
        "partitions": partitions,
        "sha256": hashes,
        "scope": "Exact 51-residue windows and accession-position strings only; accession identities and homology are not independently verified.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workbook", type=Path, required=True)
    parser.add_argument("--split-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = audit(args.workbook, args.split_dir)
    formatted = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(formatted)
    print(formatted, end="")


if __name__ == "__main__":
    main()
