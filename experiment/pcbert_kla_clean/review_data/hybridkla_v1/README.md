# HybridKla Reproduction Boundary

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

The released Kla-trained HybridKla weights must not initialize any
task-specific component without an overlap audit. Generic protein-language
model pretraining is a separate matter and can be used if documented. No
HybridKla score on this locked benchmark has been produced yet.
