# PCBert-Kla Reproducibility Framework

This directory contains the active implementation for PCBert-Kla replication
and the proposed token-gated Kla prediction framework. The code is organized as
a reproducible command-line and notebook workflow for local GPU environments
and Google Colab.

The [Kaggle source-study notebook](kaggle_source_study.ipynb) is a separate,
minimal runner for the locked Gao/rice evaluation. It clones the private
repository using a Kaggle Secret, records the environment and commit, checks
the locked inputs, and executes matched baseline and token-gated runs. It
does not replace the historical `main.ipynb` workbook.

The [Kaggle weighted-development notebook](kaggle_weighted_development.ipynb)
compares unweighted and training-set-ratio-weighted BCE for both architectures
on the locked Gao training and validation partitions. It does not evaluate Gao
internal test or rice. The weighting is computed separately from each run's
training indices, while checkpoint selection uses unweighted validation AUPRC.
The original model outputs are probabilities, so this implementation weights
the existing probability-based BCE rather than applying a logits loss to them.

The [Kaggle matched-seed development notebook](kaggle_multiseed_development.ipynb)
runs unweighted PCBert-Kla and token-gated models on the locked Gao development
partition with five matched seeds. It also evaluates a deterministic logistic
regression control on the same 27 physicochemical features. The notebook saves
per-seed validation results, paired differences, mean and sample standard
deviation, and prediction files. It neither evaluates Gao internal test nor
rice. The feature-only control has less input information than the two
ProtBert-based models; it is not a matched PLM-plus-features baseline.

The [Kaggle human benchmark notebook](kaggle_human_study.ipynb) uses the locked
KlaDB human manifests. It runs matched baseline and token-gated models on two
visible GPUs as separate processes, with pilot and timing modes that do not
evaluate the test partitions. Final test evaluation is disabled by default.
Each final Kaggle version runs one predeclared seed (`42`, `123`, or `777`) with
both architectures. Set `RUN_CAPPED_PILOT = False`,
`RUN_FULL_EPOCH_TIMING = False`, `RUN_FINAL_EVALUATION = True`, and select
`FINAL_SEED`; keep all other settings unchanged. Save a separate **Save & Run
All** version for each seed, retaining its complete notebook output. Attach
the prior versions' outputs as Kaggle inputs for subsequent runs. The notebook
verifies the per-seed artifacts and combines the results only when all three
seeds are available. Duplicate seeds and mismatched code commits or settings
are rejected. The notebook output includes `final_seed_<seed>.json`, raw
predictions, model checkpoints, logs, and the combined summary after seed 777.

The [HybridKla-source Kaggle notebook](kaggle_hybridkla_source.ipynb) is a
**second, provisional internal benchmark** from the official HybridKla
workbook. It locks accession-grouped train/validation/internal-test partitions
and offers matched PCBert-Kla/MM-Kla runs plus an architecture-aligned
eight-branch HybridKla retraining with a fine-tuned ESM2. Preparation and
integrity checks run by default; all training switches are initially off.
RUN_HYBRIDKLA_TIMING measures one ESM2 training fold for one epoch without
test inference. RUN_PAIRED_MODELS and RUN_HYBRIDKLA_FULL enable the
respective full evaluations. The official archive is downloaded and
checksum-checked at runtime, or can be supplied through LOCAL_ARCHIVE.
The workbook lacks verified parent-protein provenance and an external mouse
test, so these internal scores cannot establish generalization to a new study.
The [source audit](review_data/hybridkla_v1/README.md) records exact counts,
hashes, and further limitations.

The [human-to-mouse HybridKla-derived notebook](kaggle_hybridkla_study.ipynb)
builds a separate reference-verified and homology-screened benchmark from
supplementary Tables S1 and S7. Its locked manifest supports matched
PCBert-Kla/MM-Kla runs and architecture-aligned HybridKla retraining, with the
mouse test withheld from selection. After screening, the mouse test has only
64 reported positive sites; S1/S7 lack per-site study identifiers. The
[benchmark audit](review_data/hybridkla_study_v1/README.md) gives counts,
eligibility rules, and interpretation limits. This does not replace the
provisional workbook analysis above or establish exact reproduction of the
authors' published scores.

The [revised original-data notebook](kaggle_pcbert_revised.ipynb) freezes a
deduplicated, candidate-homology-screened repartition of the released
PCBert-Kla rows. Its [lock and audit](review_data/pcbert_revised_v1/README.md)
retain 35 positive internal-test windows, versus seven positives if exact
training overlaps are simply discarded from the old test. It remains an
internal diagnostic because the released rows lack verified parent-protein
identities and an independent study. The notebook can retrain PCBert-Kla,
MM-Kla, and architecture-aligned HybridKla on the same locked rows. The
HybridKla run requires the pinned official feature-resource archive, uses its
own sequence-derived descriptors rather than the released 27 features, and
has a substantially larger training budget. Its one-fold ESM2 timing pilot
does not score the test. All model runs are disabled by default; completed
scores and group-aware intervals are generated from saved predictions.
In the notebook configuration, set `RUN_PAIRED_MODELS = True` for the two
ProtBert models and `RUN_HYBRIDKLA_FULL = True` for HybridKla. Set
`RUN_HYBRIDKLA_PILOT = True` separately to estimate the ESM2 branch runtime.
Set `RUN_SEED` to one of the predeclared seeds (`42`, `123`, `777`); each
Kaggle saved version trains only that seed. Saved versions have separate
working directories, so keep each version's output for later aggregation.
If the official archive is attached to Kaggle rather than downloaded, set
`LOCAL_HYBRIDKLA_ARCHIVE` to its path.

### Original-training to external-study experiment

The [Kaggle original-training/external-study notebook](kaggle_original_train_kladb_external.ipynb)
compares PCBert-Kla, MM-Kla, and architecture-aligned HybridKla using the
released, unrevised training source and locked KlaDB human study holdout. The
original training file is cloned with the private repository; **only the
external-study export needs a Kaggle dataset upload**. Generate the two files
locally with:

```bash
python experiment/pcbert_kla_clean/scripts/export_kladb_external.py
```

Create a **private** Kaggle dataset from both files in
`experiment/pcbert_kla_clean/outputs/kladb_external_upload/`:
`kladb_human_external_test.csv` and `external_test_manifest.json`. These files
are ignored by Git. In Kaggle, attach that dataset, enable Internet and two
T4 GPUs, and provide a `GITHUB_TOKEN` Kaggle Secret with read access to this
repository. Run the notebook preflight, select `RUN_SEEDS` (predeclared: 42,
123, 777), and set `RUN_TRAINING = True` for the two ProtBert models. Each
uses one GPU. A candidate-homology-grouped validation subset selects the
epoch; the two models are refit on **all 3,487 original rows** before external
inference. Set `RUN_HYBRIDKLA_PILOT = True` for a one-fold ESM2 timing check
without test inference. Set `RUN_HYBRIDKLA_FULL = True` for a three-model
matched-subset comparison. This also runs a *separate* PCBert-Kla/MM-Kla pair
on exactly the grouped original-source training rows used by HybridKla;
HybridKla's grouped out-of-fold stacking and training budget remain distinct.
The official HybridKla resource archive is downloaded and checksum-checked,
or can be supplied through `LOCAL_HYBRIDKLA_ARCHIVE` if download fails.
The external study is scored only after model selection. The original
training duplicates remain, and its candidate parent-protein identities are
not verified. The independent-study negatives are putative. This same KlaDB
holdout was examined in earlier project experiments, so it is not a fresh
blinded confirmatory test.

The original train/test benchmark has exact sequence overlap. Previous metrics
remain available for historical comparison, but they do not establish
performance on unseen proteins. The current revision protocol and audit are
documented in [REVIEW_REVISION.md](REVIEW_REVISION.md).
The 27 descriptors and their verified computation are documented in
[FEATURES.md](FEATURES.md).
The source-derived, locked Gao/rice evaluation is documented in
[locked_source_study_v1](review_data/locked_source_study_v1/README.md).
The current human study-held-out revision, based on accession-labelled KlaDB
sites and a UniProt reference proteome, is documented in
[kladb_human_v1](review_data/kladb_human_v1/README.md). The same locked manifest
is used for both PCBert-Kla and token-gated comparisons. Source downloads and
generated manifests are local artifacts and are not committed.
The [HybridKla reproduction audit](review_data/hybridkla_v1/README.md) records
the upstream training dependencies, exact overlap with the locked tests, and
a split-preserving export. Section 9 of the Kaggle human notebook can run the
export and overlap audit without training either existing architecture: set
`RUN_CAPPED_PILOT = False`, `RUN_FULL_EPOCH_TIMING = False`,
`RUN_FINAL_EVALUATION = False`, and `RUN_HYBRIDKLA_PREPARATION = True`. It uses
a separate `main` checkout and records its commit. This section
does not train HybridKla or produce a HybridKla score. Section 10 runs an
exploratory eight-branch **HybridKla-adapted** model. First set
`RUN_HYBRIDKLA_PILOT = True` for a one-epoch training-only smoke test that
does not score tests. In a separate Kaggle version, set it to `False` and set
`RUN_HYBRIDKLA_ADAPTED = True` for the full seed-42 run. Keep
`RUN_HYBRIDKLA_PREPARATION = True` for both. It uses frozen ESM2
representations and a homology-group-disjoint training holdout for stacking, not the
authors' fine-tuned ESM2 and five-fold stacking. Its outputs must not be
reported as an exact HybridKla reproduction. The benchmark branch is kept
separate from `main` while the predeclared PCBert-Kla/MM-Kla seeds remain
unfinished.

## Directory Structure

```text
experiment/pcbert_kla_clean/
├── main.ipynb
├── REVIEW_REVISION.md
├── FEATURES.md
├── requirements.txt
├── scripts/
│   ├── run_replication.py
│   └── audit_features.py
└── src/pcbert_kla_clean/
    ├── backbones.py
    ├── data.py
    ├── datasets.py
    ├── experiments.py
    ├── metrics.py
    ├── model.py
    ├── review_dataset.py
    ├── simple_baselines.py
    ├── training.py
    └── utils.py
```

## Implemented Functionality

- Parsing of the original PCBert-Kla `train.csv` and `test.csv` FASTA-like
  benchmark files.
- Loading of the original `feature_train.csv` and `feature_test.csv`
  physicochemical feature tables.
- Dataset audit reporting, including sample counts, duplicate sequence counts,
  feature dimensionality, and train/test sequence overlap.
- Historical five-fold cross-validation and nominal-test evaluation.
- Reviewer audit and exact-window-disjoint diagnostic evaluation.
- Baseline PCBert-Kla-style classifier.
- Proposed token-gated architecture with central-lysine-guided token pooling,
  physicochemical feature projection, gated fusion, and residual MLP
  classification.
- Backbone comparison with ProtBert, ESM-2, ProtT5, and Ankh.
- Component ablations for the token-gated architecture.
- Accession-labelled Gao train/validation/internal-test and rice
  study-held-out external evaluation, with source-file hash checks.
- KlaDB human site preparation, full-protein homology and exact-window split
  locking, and matched baseline/token-gated study-held-out evaluation.

## Data

Historical replication experiments use the original benchmark files under:

```text
baselines/PCBert-Kla-original/data/
```

The benchmark consists of 51-amino-acid windows centered on a candidate lysine
and 27 physicochemical or residue-composition features. The original split is
retained for historical replication; publication claims require a corrected
evaluation on matched partitions.

The corrected source-study workflow instead uses the locked manifests under
`review_data/gao_2020_source_split/` and
`review_data/locked_source_study_v1/`. It computes the same 27 features from
the source windows at load time. Run
`python3 scripts/run_replication.py --run source-study-check` before training.
The rice set has putative negatives and must not be described as a set of
experimentally verified non-Kla sites.

The first source-study test results have already been inspected. Subsequent
architecture or loss changes must be developed on Gao training/validation;
reusing the same test sets afterward is a follow-up evaluation, not a new
untouched blind test. For a local validation-only weighting comparison:

```bash
python3 scripts/run_replication.py \
  --run source-study-development \
  --architecture token_gated \
  --positive-weighting train_ratio \
  --epochs 30 --batch-size 4 --device cuda \
  --optimizer sgd --learning-rate 0.003 \
  --weight-decay 0 --scheduler none
```

This saves `validation_metrics.json`, `validation_predictions.csv`, and a run
configuration containing the training class counts and applied positive weight.
Omit `--positive-weighting` for the unchanged unweighted control.

## Setup

```bash
cd experiment/pcbert_kla_clean
pip install -r requirements.txt
```

## Dataset Audit

```bash
python3 scripts/run_replication.py --run data-check
```

The audit does not download any protein language model checkpoint. It validates
the benchmark files and reports dataset statistics.

The reviewer audit generates a row-level inventory and a deterministic
exact-window-disjoint diagnostic split:

```bash
python3 scripts/run_replication.py --run review-audit
```

The manifest and JSON report are written under `outputs/review_exact_window/`.
This split is not protein- or homology-disjoint and is not a replacement for the
reviewers' requested independent evaluation.

An exploratory model run on that fixed diagnostic split is available with
`--run window-diagnostic`. It uses the manifest's validation subset for model
selection and fits feature scaling on the training subset only.

The stricter candidate-protein audit joins all exact proteome matches for a
window before partitioning. Ambiguous matches are kept together, so no mapped
protein accession crosses the resulting split:

```bash
python3 scripts/run_replication.py --run candidate-group-audit
```

Its report and manifest are under `outputs/review_candidate_groups/`. A
cluster-screened seed 42 manifest and its supporting protein search are
preserved in [review_data](review_data/README.md). Training on that fixed
split uses `--run candidate-group-diagnostic` with the cluster and search
files. These remain provisional results: original parent-protein IDs are not
verified, and an external study-sourced test is still pending. The
[revision protocol](REVIEW_REVISION.md) documents the full-protein clustering
workflow and remaining evaluation requirements.

## Baseline Replication

Five-fold cross-validation:

```bash
python3 scripts/run_replication.py \
  --run cv \
  --epochs 30 \
  --batch-size 4 \
  --device cuda
```

Original nominal-test evaluation (historical):

```bash
python3 scripts/run_replication.py \
  --run independent \
  --epochs 30 \
  --batch-size 4 \
  --device cuda
```

## Proposed Token-Gated Model

Primary single-model configuration:

```bash
python3 scripts/run_replication.py \
  --run independent \
  --architecture token_gated \
  --epochs 30 \
  --batch-size 4 \
  --device cuda \
  --optimizer sgd \
  --learning-rate 0.003 \
  --weight-decay 0.0 \
  --scheduler none
```

This configuration uses `Rostlab/prot_bert` by default, retains the first four
ProtBert encoder layers, and fine-tunes the retained encoder layers together
with the token-gated prediction head.

## Optimizer Comparison

AdamW configuration:

```bash
python3 scripts/run_replication.py \
  --run independent \
  --architecture token_gated \
  --epochs 30 \
  --batch-size 4 \
  --device cuda \
  --optimizer adamw \
  --learning-rate 2e-5 \
  --weight-decay 0.01 \
  --scheduler linear \
  --warmup-ratio 0.1
```

## Backbone Comparison

ESM-2 35M:

```bash
python3 scripts/run_replication.py \
  --run independent \
  --architecture token_gated \
  --model-name facebook/esm2_t12_35M_UR50D \
  --epochs 30 \
  --batch-size 4 \
  --device cuda \
  --optimizer adamw \
  --learning-rate 2e-5 \
  --weight-decay 0.01 \
  --scheduler linear \
  --warmup-ratio 0.1
```

ProtT5 frozen encoder:

```bash
python3 scripts/run_replication.py \
  --run independent \
  --architecture token_gated \
  --model-name Rostlab/prot_t5_xl_half_uniref50-enc \
  --encoder-layers 4 \
  --freeze-encoder \
  --epochs 30 \
  --batch-size 2 \
  --device cuda \
  --optimizer adamw \
  --learning-rate 1e-4 \
  --weight-decay 0.01 \
  --scheduler linear \
  --warmup-ratio 0.1
```

Ankh-base frozen encoder:

```bash
python3 scripts/run_replication.py \
  --run independent \
  --architecture token_gated \
  --model-name ElnaggarLab/ankh-base \
  --encoder-layers 4 \
  --freeze-encoder \
  --epochs 30 \
  --batch-size 2 \
  --device cuda \
  --optimizer adamw \
  --learning-rate 1e-4 \
  --weight-decay 0.01 \
  --scheduler linear \
  --warmup-ratio 0.1
```

## Token-Gated Ablation Study

The ablation study keeps the ProtBert token-gated SGD configuration fixed and
removes one component at a time.

No gated fusion:

```bash
python3 scripts/run_replication.py \
  --run independent \
  --architecture token_gated \
  --ablation no_gated_fusion \
  --epochs 30 \
  --batch-size 4 \
  --device cuda \
  --optimizer sgd \
  --learning-rate 0.003 \
  --weight-decay 0.0 \
  --scheduler none
```

No physicochemical feature branch:

```bash
python3 scripts/run_replication.py \
  --run independent \
  --architecture token_gated \
  --ablation no_physicochemical \
  --epochs 30 \
  --batch-size 4 \
  --device cuda \
  --optimizer sgd \
  --learning-rate 0.003 \
  --weight-decay 0.0 \
  --scheduler none
```

No sequence branch:

```bash
python3 scripts/run_replication.py \
  --run independent \
  --architecture token_gated \
  --ablation no_sequence \
  --epochs 30 \
  --batch-size 4 \
  --device cuda \
  --optimizer sgd \
  --learning-rate 0.003 \
  --weight-decay 0.0 \
  --scheduler none
```

## Reproducibility Notes

- Default checkpoint: `Rostlab/prot_bert`.
- Default retained encoder depth: 4 layers.
- Default feature dimension: 27.
- Default batch size: 4.
- Default epoch count: 30.
- Default decision threshold: 0.5.
- Validation is performed after every epoch for independent-test runs that use
  the default validation split.
- Protein language model embeddings are computed during each training and
  evaluation batch; they are not precomputed.
- Mixed precision, gradient accumulation, LoRA, and gradient checkpointing are
  not enabled in the current implementation.

## Benchmark Caveat

The original benchmark contains train/test sequence overlap. The audit command
reports the overlap explicitly. Results should therefore be interpreted as
same-benchmark comparisons against PCBert-Kla rather than as evidence of
generalization to a fully independent homology-filtered benchmark.
