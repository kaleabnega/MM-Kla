# CLEAR-Lactyl Protein-Grouped Benchmark

## Purpose

This is a second human Kla benchmark prepared from the [AttentionKla authors'
CLEAR-Lactyl release](https://github.com/lee190632/AttentionKla/tree/main/data)
for a matched comparison of MM-Kla and competing models. It is **not** the
authors' published random 80/10/10 split, and published AttentionKla metrics
cannot be compared directly with metrics from this split. The source paper is
the [AttentionKla preprint](https://doi.org/10.64898/2026.07.28.741267).

The authors' CSV supplies accession, position, 51-residue sequence, and label,
but no per-site study identifier. The held-out test below is independent by
protein and qualifying full-protein homology, **not demonstrably independent
by experimental study**. It does not replace a separate-source external test.

## Source Integrity

| Input | SHA-256 |
|---|---|
| `downloaded-dataset/CLEAR-Lactyl.csv` | `088d15a9c0119e64723dcd613b64ae8c6812740659129cabbb56b8f70eb02a21` |
| `downloaded-dataset/UP000005640_9606.fasta.gz` | `64299d01e8f41fb6ef03cdf7ec051d3d4e5f041003b6b0fa75b09d439af79dce` |
| `downloaded-dataset/kladb_homo_sapiens_2026-09-20.csv.gz` | `6d39eda5c7ac2ac4231ba05e77c68065d9cd726a14fed147f0f601a5ae802fce` |

The raw CSV and generated manifests are local, ignored data artifacts. The
published CSV has 33,208 rows (16,604 of each label), 4,968 accessions, no
duplicate exact strings or accession-position pairs, and no conflicting
labels for an identical raw sequence. The `<pad>` token occupies five text
characters, not one residue; this pipeline converts it to a single position
before validation. Only complete, canonical 51-residue windows with a central
lysine and an exact accession-position match to the pinned UniProt human
reference proteome are retained. This means isoforms absent from that
reference and padded terminal windows are deliberately excluded.

Authors-labelled negatives matching a known KlaDB human positive at the same
accession-position or at an identical window are removed. KlaDB is used here
only as a **positive-label conflict screen**, not to add training examples or
select a model. Other negatives are still *putative*: absence of reported Kla
is not experimental evidence of absence.

| Preparation outcome | Rows |
|---|---:|
| Retained positives | 14,561 |
| Retained putative negatives | 14,870 |
| Padded or noncanonical windows | 2,112 |
| Invalid window length or central position | 166 |
| Accession absent from pinned reference | 1,218 |
| Reference position/window mismatch | 19 |
| Labelled negatives matching KlaDB positives | 262 |

The 29,431 retained rows occupy 4,402 verified proteins. All exclusion counts
and file hashes are recorded in `outputs/clear_lactyl_preparation/preparation_audit.json`.

## Locked Split

MMseqs2 `d0185a2d3478af8e8dccfeeadd420072474f2138` searched the retained
full proteins against themselves in both directions with at least 30% sequence
identity and 80% coverage of both proteins (`--cov-mode 0`, sensitivity 7,
E-value 0.001). Qualifying hits define connected homology groups. Groups, not
site rows, were assigned to train, validation, or held-out internal test using
seed 42. The assignment targeted 70/10/20 by row count while also matching
the positive distribution. Exact windows and accession IDs are verified not
to cross partitions. No post-split class resampling is applied.

| Partition | Positive | Putative negative | Total | Proteins | Homology groups |
|---|---:|---:|---:|---:|---:|
| Train | 10,191 | 10,408 | 20,599 | 3,037 | 2,480 |
| Validation | 1,459 | 1,485 | 2,944 | 442 | 359 |
| Held-out internal test | 2,911 | 2,977 | 5,888 | 923 | 729 |

`outputs/clear_lactyl_locked/full_manifest.csv` and `model_manifest.csv`
contain the same rows because no training sampling was applied. Their SHA-256
is `f3fbb6dddf346fd0086d693f783ec476ec802eb56586ed3225a2eda82f72551b`.
`split_audit.json` stores the source checksums, split counts, and homology
screen summary. These files are not model results.

## Reproduction

From `experiment/pcbert_kla_clean/`:

```bash
python3 scripts/prepare_clear_lactyl.py \
  --source-csv downloaded-dataset/CLEAR-Lactyl.csv \
  --proteome-fasta downloaded-dataset/UP000005640_9606.fasta.gz \
  --kladb-human-csv downloaded-dataset/kladb_homo_sapiens_2026-09-20.csv.gz \
  --output-dir outputs/clear_lactyl_preparation

mmseqs easy-search \
  outputs/clear_lactyl_preparation/candidate_proteins.fasta \
  outputs/clear_lactyl_preparation/candidate_proteins.fasta \
  outputs/clear_lactyl_preparation/all_vs_all.tsv \
  /tmp/clear_lactyl_mmseqs_tmp \
  --min-seq-id 0.3 -c 0.8 --cov-mode 0 -s 7 \
  --max-seqs 5000 --threads 4 \
  --format-output query,target,pident,qcov,tcov

python3 scripts/lock_clear_lactyl.py \
  --candidate-csv outputs/clear_lactyl_preparation/candidate_sites.csv \
  --preparation-audit outputs/clear_lactyl_preparation/preparation_audit.json \
  --homology-hits-tsv outputs/clear_lactyl_preparation/all_vs_all.tsv \
  --output-dir outputs/clear_lactyl_locked --seed 42
```

The raw data are available from the authors' repository; the reference
proteome and KlaDB human export are the same pinned inputs used by the
[KlaDB human benchmark](../kladb_human_v1/README.md). Hold out the internal
test until model selection is complete. Compare all models using identical
partitions, preprocessing, selection criteria, and evaluation metrics.

## Scope And Limitations

- The study provenance of 4,378 retained positives could not be matched to a
  KlaDB accession-position record. The released authors' CSV does not assign
  study IDs, so a study-disjoint split cannot be asserted from this release.
- The positive evidence comes from the source authors' curation; this audit
  verifies sequence and accession-position agreement, not each original
  mass-spectrometry record.
- A qualifying full-length homology screen does not exclude every shared
  shorter domain or motif.
- This split is approximately balanced. Its AUPRC and accuracy are not directly
  comparable to the earlier imbalanced KlaDB human benchmark. Changing to it
  cannot by itself establish an architectural improvement.
- Any separate KlaDB external evaluation requires a fresh **cross-dataset**
  protein/homology and window-overlap audit before training. A preliminary
  exact audit found 134 overlapping windows and six shared accessions between
  the unsplit retained CLEAR-Lactyl sites and the existing KlaDB external set.
