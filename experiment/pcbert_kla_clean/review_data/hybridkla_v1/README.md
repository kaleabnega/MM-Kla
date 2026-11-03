# HybridKla Reproduction Boundary

## Provisional HybridKla-Source Benchmark

The [second-benchmark Kaggle notebook](../../kaggle_hybridkla_source.ipynb)
prepares the official-archive train/train_data.xlsx with
scripts/lock_hybridkla_benchmark.py. The workbook is pinned by SHA-256 and
is not redistributed. Of 47,004 source rows, 41,468 canonical 51-residue
windows remain after excluding 5,536 noncanonical windows; no conflicting
site labels or exact-window labels were found. The eligible set has 20,107
positive and 21,361 negative-labelled rows. Exact sites/windows are deduplicated
before the split. The deterministic accession-grouped partitions are:

| Partition | Rows | Positive labels | Accession strings |
| --- | ---: | ---: | ---: |
| Train | 29,023 | 14,083 | 4,856 |
| Validation | 4,148 | 2,000 | 749 |
| Internal test | 8,297 | 4,024 | 1,385 |

The local manifest SHA-256 is
9628cd756316dc5d6234b259d5a5f0a2364460e300ac3e6ba9d654239599a955.
The lock checks accession and exact-window disjointness. It does **not**
verify that accession-like strings refer to the correct original proteins,
separate homologous proteins, validate the provenance of every positive site,
or supply a study-held-out/external test. Label 0 means a **putative negative**,
not experimentally confirmed absence of lactylation. This is a provisional
internal comparison, not a replacement for the primary KlaDB human benchmark
or a complete resolution of the reviewers' dataset concerns.

The notebook can train PCBert-Kla and MM-Kla on the same locked data and
invoke scripts/run_hybridkla_architecture.py for an eight-branch HybridKla
comparison. That runner fine-tunes ESM2 and uses five **protein-grouped**
out-of-fold training splits, in contrast to the authors' row-wise folds.
Its ESM2 pooling and binary-logit formulation also differ from the published
script. It is therefore **architecture-aligned retraining**, not an exact
reproduction. A one-fold, one-epoch ESM2 timing run is provided before the
full run. No result is claimed until all three models have been trained and
evaluated on this lock. The official top10.txt AAindex selection resource
has not been traced to its original fitting rows; if it was selected using
the whole published workbook, that selection would constitute another
possible source of test information. This must be audited before a strict
leakage-free claim for the HybridKla comparator.

## Purpose

HybridKla is a reviewer-named comparator for the accession-labelled KlaDB
human benchmark. A head-to-head result requires the same locked training,
validation, internal-test, and study-held-out partitions used by PCBert-Kla
and MM-Kla. Published HybridKla scores and Kla-trained checkpoints are not
substitutes for this comparison: their training data may include sites in our
locked tests.

## Upstream Audit

The [authors' source repository](https://github.com/kongxianzw/HybridKla),
inspected at commit `22d90f5f964cf16f28f087cfbb75049c865c55a2`, contains
separate scripts for ESM2, LSTM, six handcrafted feature classifiers, and a
meta-model. The ESM2, LSTM, and six-feature scripts read a single
`train_data.xlsx` or derived feature files and create their own five-fold
splits. The meta-model reads predictions from those folds. The published
training scripts do not accept a pre-existing train/validation/test split or
produce a locked external-test evaluation without adaptation. The repository
does not include a license file; upstream source is linked, not redistributed
here.

The full architecture has eight base predictors: ESM2, LSTM, ACF, AAINDEX,
CKSAAP, GPS, OBC, and PSEAAC, followed by a learned meta-model. Replacing
this with only ESM2 and a small classifier would be a different baseline,
not a reproduction of HybridKla.

### Training Workbook Overlap

The GitHub `train/train_data.xlsx` (SHA-256
`c5196c40a3ee0c3044f332a25b47f175cc69b88eaa909a5c3b82cd1fc50d5b29`)
has 47,004 unique 51-residue windows, balanced between positive and negative
labels. All `UniProt_ID` entries are generic `trainN` identifiers, so they
cannot support a protein-level overlap check. Exact-window comparison with
our locked export found:

| Locked partition | Rows | Exact matches | Matches with opposite labels |
| --- | ---: | ---: | ---: |
| Training | 39,880 | 9,519 | 644 |
| Validation | 17,713 | 2,134 | 223 |
| Internal test | 35,455 | 4,656 | 555 |
| Study-held-out test | 5,095 | 66 | 2 |

The authors' [official code archive](https://transkla.zzu.edu.cn/download/)
(`HybridKla.zip`, SHA-256
`73ddb128c441d7b213c4636e6bb6ce34501e82143c2b9bc06ec329d68a422993`)
contains the same 47,004 windows and labels but uses accession-position strings
instead of generic `trainN` IDs. It also contains `BLOSUM62R.txt` (SHA-256
`7f6f4a1ba4c9df90b402898e9ed7baaf9e515037b4236f063d9fcf991e044526`),
which is absent from GitHub. The official workbook's SHA-256 is
`e813ed5e495b63539a409bf66d70f5279d1308dfda65a1da181c39eabb4aa449`.
Its IDs are accession-like strings and have not been independently validated
against UniProt. Comparing those strings with the locked export finds:

| Locked partition | Exact site IDs | Rows sharing an accession | Distinct shared accessions |
| --- | ---: | ---: | ---: |
| Training | 9,171 | 30,500 | 2,009 |
| Validation | 2,062 | 12,406 | 281 |
| Internal test | 4,438 | 25,233 | 556 |
| Study-held-out test | 64 | 665 | 3 |

The opposing labels reflect a difference between the two dataset definitions;
they do not establish which label is biologically correct. The upstream ESM2
training script reads this workbook directly. Thus, a Kla-trained model built
from those records cannot be presented as independent on either locked test.
This audit covers only exact windows, not homologous proteins or all possible
upstream training sources. It is reproducible without redistributing upstream
data:

```bash
cd experiment/pcbert_kla_clean
python3 scripts/audit_hybridkla_overlap.py \
  --workbook /path/to/official/HybridKla/train/train_data.xlsx \
  --split-dir outputs/hybridkla_locked \
  --output outputs/hybridkla_locked/upstream_overlap_audit.json
```

## Locked Input Export

`scripts/export_hybridkla_splits.py` verifies the adjacent locked manifest
against `split_audit.json` and exports four CSV files with site-level keys:

```bash
cd experiment/pcbert_kla_clean
python3 scripts/export_hybridkla_splits.py \
  --human-lock outputs/kladb_human_locked/split_audit.json \
  --output-dir outputs/hybridkla_locked
```

Each row retains its 51-residue lysine-centered window and binary label.
`Site_ID` contains the unique `accession_position` site key; `UniProt_ID` and
`ACC_ID` retain the actual parent protein accession. Upstream feature-score
joins that currently use `UniProt_ID` must be adapted to `Site_ID` because a
protein can have multiple candidate lysines. The exporter rejects repeated sites/windows and
cross-partition proteins or homology groups. It does not train a model or
touch the test partitions for tuning.

The verified export contains 39,880 training rows (9,970 positives), 17,713
validation rows (1,436 positives), 35,455 internal-test rows (2,864 positives),
and 5,095 study-held-out rows (254 positives). The exported files remain
untracked local artifacts; checksums and counts are in `export_audit.json`.

## Required Training Adaptation

A valid HybridKla comparison still requires training every Kla-specific base
predictor on the locked training data only. Meta-model training needs
out-of-fold base predictions generated **within training**; neither locked
test partition may participate in base training, epoch selection, or feature
normalization. Validation alone selects checkpoints and any method-specific
hyperparameters. Only then may the two tests be scored at the predeclared
threshold, using the same reporting metrics and seeds as the other models.

Kla-trained HybridKla weights or predictions derived from the upstream
training workbook must not be used on the locked tests. Generic protein-language
model pretraining is a separate matter and can be used if documented. No
HybridKla score on this locked benchmark has been produced yet.

`scripts/run_hybridkla_adapted.py` implements a compute-bounded exploratory
eight-branch comparison using the six feature encoders, a six-layer LSTM,
frozen ESM2 embeddings with a trained head, and a learned meta-model. The
GPS reference is fitted only on the base-training subset. Base predictors use
a homology-group-disjoint 80% training subset; the remaining training groups fit
the meta-model. Checkpoints are selected by validation AUPRC, then the two
locked tests are scored at threshold 0.5. The official `top10.txt` and
`BLOSUM62R.txt` are loaded at runtime and verified by SHA-256. This is an
**adaptation**, not an exact reproduction of the authors' five-fold, fine-tuned
ESM2 training protocol; any resulting score must be labeled accordingly.
The Kaggle notebook's section 10 invokes this runner only when explicitly
enabled. No score is claimed until that run completes.

On two synthetic 51-residue windows, the ACF, AAINDEX, CKSAAP, GPS, and
PSEAAC encoders were compared with the authors' official archive functions;
their maximum absolute differences were below `2e-6`. The archive's OBC
function calls `tf.keras` without importing `tf`, so it could not be executed
unchanged in that check. The adaptation uses a direct 22-category one-hot
encoding with the published 1,122-value dimension. GPS equivalence was checked
with the same reference windows; in the benchmark runner, its reference is
restricted to base-training windows to prevent test contamination.
