# Evaluation Revision Protocol

## Status

The PCBert-Kla/DeepKla benchmark supplied with this repository has 3,487
training rows and 354 nominal test rows. Exact 51-residue window comparison
finds 255 test rows also present in training, including 170 of 177 positive
test rows. Only 172 distinct positive windows occur among the 1,720 positive
training rows. The nominal test set is therefore unsuitable as an independent
generalization estimate.

The audit also finds five unique windows with both labels: four cross the
original train/test boundary and all five have contradictory rows within the
original test file. They require source-level investigation; no label is
inferred from majority vote. The diagnostic protocol excludes all 50 rows
belonging to these five windows.

## Reproducible Audit

From `experiment/pcbert_kla_clean/`:

```bash
python3 scripts/run_replication.py --run review-audit
```

The command writes `outputs/review_exact_window/dataset_audit.json` and
`outputs/review_exact_window/window_inventory.csv`. The inventory records
each unique sequence, its observed labels, original split counts, label counts,
feature consistency, and diagnostic assignment. It does not download a PLM or
train a model.

## Diagnostic Partition

The audit combines the original files, excludes contradictory labels and
inconsistent feature vectors, and keeps one record per exact window before
partitioning. It uses a fixed seed (42) and class-stratified 70/10/20
train/validation/test proportions. The resulting 1,932 distinct windows are
assigned as follows:

| Partition | Negative | Positive | Total |
|---|---:|---:|---:|
| Train | 1,230 | 122 | 1,352 |
| Validation | 176 | 17 | 193 |
| Test | 352 | 35 | 387 |

The diagnostic test prevalence is approximately 9%, compared with 50% in the
original nominal test set. AUPRC and threshold-dependent metrics must be read
in that context. Any resampling for training must happen after this split and
must be confined to its training partition.

To train and evaluate the existing architecture on the fixed diagnostic split:

```bash
python3 scripts/run_replication.py \
  --run window-diagnostic \
  --architecture token_gated \
  --epochs 30 \
  --batch-size 4 \
  --device cuda \
  --optimizer sgd \
  --learning-rate 0.003 \
  --weight-decay 0.0 \
  --scheduler none \
  --save-models
```

Training feature normalization is fitted on training rows alone. The fixed
validation subset controls checkpoint selection (validation AUPRC by default)
and, if requested, decision threshold selection; the diagnostic test subset is
evaluated afterward. The default decision threshold remains 0.5.
Each distinct run configuration writes predictions, metrics, and a run record
under `outputs/review_exact_window/runs/<configuration-hash>/`; `--save-models`
also saves model weights and the fitted scaler there.
Changing `--seed` changes model initialization without changing the diagnostic
partition. The split can be regenerated with `--review-split-seed` and the
fractions with `--review-test-fraction` and
`--review-validation-fraction`; comparisons require the same manifest.

## Limits and Publication Evaluation

Exact-window disjointness does not prevent homologous proteins or overlapping
windows from related proteins appearing on opposite sides of the split. The
public FASTA-like files identify rows as `Protein N`, without verified parent
accessions. Earlier exploratory mapping to a *Botrytis cinerea* proteome is
suggestive but cannot substitute for verified source identities. This
diagnostic partition is also not newly blinded: the source rows were used in
earlier model development.

The manuscript's corrected main evaluation requires verified full-protein
identities, site positions, and source labels, followed by protein- or
homology-cluster-disjoint partitioning before oversampling. All principal
baselines and architectural ablations must use the same partitions and
preprocessing. At least five independent training seeds, saved per-sample
predictions, and paired statistical comparisons are needed before revising the
performance claims. The original overlap-affected metrics should be labeled
as historical benchmark results in any revised manuscript.

For background on sequence leakage and non-overlapping group evaluation, see
[Hou et al., PLOS Computational Biology (2022)](https://doi.org/10.1371/journal.pcbi.1010669)
and the [scikit-learn StratifiedGroupKFold documentation](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedGroupKFold.html).
