"""Freeze a leakage-screened diagnostic split of the released PCBert-Kla rows."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
ROOT = PROJECT.parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from pcbert_kla_clean.candidate_groups import prepare_candidate_group_partitions
from pcbert_kla_clean.data import load_split


EXPECTED_SHA256 = {
    "train.csv": "5a2f2f220c4761b963b9c4487573f4d6fad516d820ef29837c8553075fa05824",
    "test.csv": "f37a441710fb7198c17a104585ac723aa76afd51ff21dc31224aacb89b855d21",
    "feature_train.csv": "dc1f19a95bf420804203ab2db1010e15cb925f1bb050b5cdd4066a23c15a942b",
    "feature_test.csv": "1f40f209bdf55b6edbc663785271d17e9ea637207dbd0e46ce6c25fc46b17738",
    "candidate_mapping": "e6cc23a8355864fb8c426fbbd57ab1201f23e3e52c0e3f2ef29cbfebcd3ef341",
    "homology_clusters": "8fffc04dcfa335d752acb195f94f05fc791dad3543528d1812abda82d5f2ffab",
    "homology_hits": "3ca3bbf2fb38a131beda2133c0124fd273a9361decc214dca7428f6067bc240a",
}
FIELDNAMES = (
    "window_id", "partition", "sequence", "position_1based", "label",
    "candidate_accessions", "candidate_group", "original_train_rows", "original_test_rows",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_lock(
    baseline_root: Path, mapping_csv: Path, homology_csv: Path,
    homology_hits_tsv: Path, output_dir: Path, *, seed: int = 42,
    negative_ratio: int = 3,
) -> dict:
    if negative_ratio < 1:
        raise ValueError("Training negative ratio must be positive")
    data = baseline_root / "data"
    inputs = {name: data / name for name in
              ("train.csv", "test.csv", "feature_train.csv", "feature_test.csv")}
    inputs.update({"candidate_mapping": mapping_csv, "homology_clusters": homology_csv,
                   "homology_hits": homology_hits_tsv})
    digests = {name: sha256(path) for name, path in inputs.items()}
    if digests != EXPECTED_SHA256:
        raise ValueError("Released benchmark or candidate-homology audit differs from pinned inputs")
    train = load_split(data / "train.csv", data / "feature_train.csv")
    test = load_split(data / "test.csv", data / "feature_test.csv")
    partitions = prepare_candidate_group_partitions(
        train, test, mapping_csv=mapping_csv, homology_csv=homology_csv,
        homology_hits_tsv=homology_hits_tsv, seed=seed,
        test_fraction=.2, validation_fraction=.1,
    )
    inventory = {row["sequence"]: row for row in partitions.inventory}
    feature_names = train.feature_names
    feature_columns = [f"feature_{index}" for index in range(len(feature_names))]
    full_rows = []
    for name, split in (("train", partitions.train), ("validation", partitions.validation),
                        ("internal_test", partitions.test)):
        for record, features in zip(split.records, split.features):
            source = inventory[record.sequence]
            if source["assignment"] != ("test" if name == "internal_test" else name):
                raise AssertionError("Inventory assignment differs from partition")
            row = {
                "window_id": record.name, "partition": name, "sequence": record.sequence,
                "position_1based": str(record.position), "label": str(record.label),
                "candidate_accessions": source["candidate_accessions"],
                "candidate_group": source["candidate_group"],
                "original_train_rows": str(source["original_train_rows"]),
                "original_test_rows": str(source["original_test_rows"]),
            }
            row.update({column: repr(float(value)) for column, value in zip(feature_columns, features)})
            full_rows.append(row)
    full_rows.sort(key=lambda row: (row["partition"], row["window_id"]))
    if len({row["sequence"] for row in full_rows}) != len(full_rows):
        raise AssertionError("The locked manifest repeats a sequence")
    rng = random.Random(seed)
    positives = [row for row in full_rows if row["partition"] == "train" and row["label"] == "1"]
    negatives = [row for row in full_rows if row["partition"] == "train" and row["label"] == "0"]
    selected_negatives = rng.sample(negatives, min(len(negatives), negative_ratio * len(positives)))
    model_rows = sorted(positives + selected_negatives +
                        [row for row in full_rows if row["partition"] != "train"],
                        key=lambda row: (row["partition"], row["window_id"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    for filename, records in (("full_manifest.csv", full_rows),
                              ("model_manifest.csv", model_rows)):
        with (output_dir / filename).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=(*FIELDNAMES, *feature_columns),
                                    lineterminator="\n")
            writer.writeheader()
            writer.writerows(records)
    report = {
        "protocol": "Repartitioned released PCBert-Kla windows; candidate-homology-screened diagnostic",
        "status": "not verified protein-disjoint and not an independent-study test",
        "source_sha256": digests,
        "split_seed": seed,
        "test_fraction": .2,
        "validation_fraction": .1,
        "training_negative_ratio_cap": negative_ratio,
        "feature_names": feature_names,
        "source_audit": partitions.report,
        "original_test_if_train_overlaps_removed": {
            "rows": len(test.records) - partitions.report["original_overlap"]["overlap_test_rows"],
            "positives": int(test.labels.sum()) - partitions.report["original_test_overlap_rows_by_label"]["1"],
            "negatives": len(test.records) - int(test.labels.sum())
                         - partitions.report["original_test_overlap_rows_by_label"]["0"],
        },
        "partitions_before_training_resampling": {
            name: {
                "rows": sum(row["partition"] == name for row in full_rows),
                "positives": sum(row["partition"] == name and row["label"] == "1" for row in full_rows),
                "putative_negatives": sum(row["partition"] == name and row["label"] == "0" for row in full_rows),
                "candidate_groups": len({row["candidate_group"] for row in full_rows
                                         if row["partition"] == name}),
            } for name in ("train", "validation", "internal_test")
        },
        "sampled_training_rows": sum(row["partition"] == "train" for row in model_rows),
        "output_sha256": {name: sha256(output_dir / name)
                          for name in ("full_manifest.csv", "model_manifest.csv")},
        "negative_definition": "Released non-Kla labels; experimental non-lactylation not established",
        "model_selection": "Training and validation only; internal test is not used for tuning",
    }
    (output_dir / "split_audit.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-root", type=Path,
                        default=ROOT / "baselines" / "PCBert-Kla-original")
    parser.add_argument("--mapping-csv", type=Path,
                        default=ROOT / "experiment-1/structure_aware_kla/results/botrytis_b0510_window_mapping/all_window_matches.csv")
    parser.add_argument("--homology-csv", type=Path,
                        default=PROJECT / "review_data/homology_clusters_mmseqs2.csv")
    parser.add_argument("--homology-hits-tsv", type=Path,
                        default=PROJECT / "review_data/all_vs_all_mmseqs2.tsv")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-negative-ratio", type=int, default=3)
    args = parser.parse_args()
    report = build_lock(args.baseline_root, args.mapping_csv, args.homology_csv,
                        args.homology_hits_tsv, args.output_dir,
                        seed=args.seed, negative_ratio=args.train_negative_ratio)
    print(json.dumps(report["partitions_before_training_resampling"], indent=2))


if __name__ == "__main__":
    main()
