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

## Candidate Protein Mapping

An earlier exact-sequence search against a *Botrytis cinerea* B05.10 proteome
produced the mapping under
`experiment-1/structure_aware_kla/results/botrytis_b0510_window_mapping/`.
The record-level audit of that mapping covers all 3,841 benchmark rows. Of
these, 3,464 have one candidate protein/site and 377 have multiple candidate
sites. All 1,937 distinct 51-aa windows have at least one exact proteome match.

The original train and test partitions share 139 accessions among uniquely
mapped records. All 319 uniquely mapped test rows have an accession that also
appears among uniquely mapped training rows. Even after excluding the 255 test
rows whose exact windows occur in training, 81 of the remaining 99 test rows
uniquely map to an accession represented in training. The other 18 have
ambiguous site mappings.

A follow-up audit using a dated UniProt B05.10 proteome, the public DeepKla
train/test files, and Gao et al. 2020 Supplementary Table 1 is recorded in
`experiment-1/structure_aware_kla/results/source_site_audit_2026_10_27/`.
The DeepKla and PCBert-Kla sequence-label files are identical as multisets;
the dated proteome reproduces all 3,841 earlier candidate-site assignments.
Of the 172 distinct positive training windows, 169 exactly match a Gao
accession and Kla position and three match reported modified peptides with
different current proteome accessions. In the test positives, 174 are exact
Gao site matches and three are peptide-only matches. Of the exact positive
site assignments, 167 appear in both original partitions and only seven are
test-only. Four test negatives exactly match Gao-reported Kla sites, with
one additional negative matching a reported modified peptide only.

These findings strongly connect the benchmark positives to the published
*B. cinerea* lactylome, including the nominal training positives. They do not
establish the original parent protein of every benchmark record, particularly
the negatives, or resolve the discrepancy with the publications' description
of rice-derived training data. The dataset authors should be asked for the
original accession/site/label inventory before a verified protein-level
partition is claimed.

These are **candidate sequence matches**, not verified source annotations.
The original *DeepKla* and PCBert-Kla publications describe rice-derived
training and *B. cinerea* testing, whereas the mapping artifact matches the
released training and test windows to the *B. cinerea* proteome. The benchmark
files expose only generic `Protein N` names. The initial proteome FASTA was
not retained, but the mapping has now been reproduced using the dated FASTA
and checksums in the follow-up audit. Experimental source proteins,
negative-label construction, and the species discrepancy still require
confirmation from the original dataset authors. No biological label should
be inferred merely from a sequence match.

From the repository root, the audit can be rerun without a GPU:

```bash
python3 experiment-1/structure_aware_kla/scripts/audit_mapped_sites.py \
  --mapping-csv experiment-1/structure_aware_kla/results/botrytis_b0510_window_mapping/all_window_matches.csv \
  --train-windows baselines/PCBert-Kla-original/data/train.csv \
  --test-windows baselines/PCBert-Kla-original/data/test.csv \
  --output-dir experiment-1/structure_aware_kla/results/botrytis_b0510_window_mapping
```

It writes `candidate_protein_mapping_audit.json` and a per-record
`candidate_protein_sites.csv`. See the [DeepKla source paper](https://doi.org/10.1002/imt2.11),
[PCBert-Kla source paper](https://doi.org/10.1093/bib/bbaf615), and the
[published *B. cinerea* lactylome](https://doi.org/10.3389/fmicb.2020.594743).

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

Exact-window disjointness does not prevent sites from the same or homologous
proteins appearing on opposite sides of the split. The public FASTA-like files
identify rows as `Protein N`, without verified parent accessions. Candidate
proteome matches cannot substitute for verified source identities. This
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

The versioned full-protein FASTA is available. The remaining dependency is an
authoritative accession/site/label table for all benchmark rows, including
negatives. Until the records are reconciled, no protein-disjoint test set
should be described as verified.

For background on sequence leakage and non-overlapping group evaluation, see
[Hou et al., PLOS Computational Biology (2022)](https://doi.org/10.1371/journal.pcbi.1010669)
and the [scikit-learn StratifiedGroupKFold documentation](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedGroupKFold.html).
