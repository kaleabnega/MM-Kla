"""Export the locked human benchmark with site IDs for HybridKla adaptation."""

import argparse
import csv
import hashlib
import json
from pathlib import Path


PARTITIONS = ("train", "validation", "internal_test", "external_test")
OUTPUT_COLUMNS = ("Site_ID", "UniProt_ID", "ACC_ID", "Mod_positions", "Sequence", "Label")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def export_splits(lock: Path, output_dir: Path) -> dict:
    lock = lock.resolve()
    manifest = lock.with_name("model_manifest.csv")
    audit = json.loads(lock.read_text())
    expected_digest = audit["output_sha256"]["model_manifest.csv"]
    if sha256(manifest) != expected_digest:
        raise ValueError("Model manifest does not match the locked split audit")

    rows = {partition: [] for partition in PARTITIONS}
    site_partition = {}
    protein_partition = {}
    group_partition = {}
    window_partition = {}
    with manifest.open(newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"accession", "position_1based", "window_51aa", "label", "homology_group", "partition"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError("Model manifest is missing required columns")
        for row in reader:
            partition = row["partition"]
            if partition not in rows:
                raise ValueError(f"Unexpected partition: {partition}")
            accession = row["accession"]
            position = int(row["position_1based"])
            window = row["window_51aa"]
            label = row["label"]
            group = row["homology_group"]
            if not accession or not group or position < 1 or len(window) != 51 or window[25] != "K":
                raise ValueError(f"Invalid site or sequence window: {accession}:{position}")
            if label not in {"0", "1"}:
                raise ValueError(f"Invalid label: {accession}:{position}")
            site_id = f"{accession}_{position}"
            if site_id in site_partition:
                raise ValueError(f"Repeated site: {site_id}")
            site_partition[site_id] = partition
            for name, key, mapping in (
                ("protein", accession, protein_partition),
                ("homology group", group, group_partition),
            ):
                previous = mapping.setdefault(key, partition)
                if previous != partition:
                    raise ValueError(f"{name} crosses partitions: {key}")
            if window in window_partition:
                if window_partition[window] != partition:
                    raise ValueError(f"Window crosses partitions: {site_id}")
                raise ValueError(f"Repeated window in {partition}: {site_id}")
            window_partition[window] = partition
            rows[partition].append({
                "Site_ID": site_id,
                "UniProt_ID": accession,
                "ACC_ID": accession,
                "Mod_positions": position,
                "Sequence": window,
                "Label": int(label),
            })

    counts = {name: {"rows": len(values), "positives": sum(r["Label"] for r in values)}
              for name, values in rows.items()}
    partition_audit = audit["partitions_before_training_resampling"]
    for partition in PARTITIONS:
        expected_rows = (audit["sampled_training_rows"] if partition == "train"
                         else partition_audit[partition]["rows"])
        expected_positives = partition_audit[partition]["positives"]
        if counts[partition] != {"rows": expected_rows, "positives": expected_positives}:
            raise ValueError(f"Locked {partition} counts do not match the manifest")

    output_dir.mkdir(parents=True, exist_ok=True)
    output_hashes = {}
    for partition, values in rows.items():
        output_path = output_dir / f"{partition}.csv"
        with output_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS, lineterminator="\n")
            writer.writeheader()
            writer.writerows(values)
        output_hashes[output_path.name] = sha256(output_path)
    report = {
        "source_audit_sha256": sha256(lock),
        "source_manifest_sha256": expected_digest,
        "counts": counts,
        "output_sha256": output_hashes,
        "site_id_note": "Site_ID is the unique accession_position key for peptide joins; UniProt_ID and ACC_ID retain the parent protein accession.",
    }
    (output_dir / "export_audit.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--human-lock", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(export_splits(args.human_lock, args.output_dir), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
