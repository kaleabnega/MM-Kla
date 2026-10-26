# PCBert-Kla Reproducibility Framework

This directory contains the active implementation for PCBert-Kla replication
and the proposed token-gated Kla prediction framework. The code is organized as
a reproducible command-line and notebook workflow for local GPU environments
and Google Colab.

The original train/test benchmark has exact sequence overlap. Previous metrics
remain available for historical comparison, but they do not establish
performance on unseen proteins. The current revision protocol and audit are
documented in [REVIEW_REVISION.md](REVIEW_REVISION.md).
The 27 descriptors and their verified computation are documented in
[FEATURES.md](FEATURES.md).

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

## Data

All active experiments use the original benchmark files under:

```text
baselines/PCBert-Kla-original/data/
```

The benchmark consists of 51-amino-acid windows centered on a candidate lysine
and 27 physicochemical or residue-composition features. The original split is
retained for historical replication; publication claims require a corrected
evaluation on matched partitions.

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
