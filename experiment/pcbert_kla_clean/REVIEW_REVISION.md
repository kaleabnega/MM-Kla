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
of rice-derived training data. Author clarification would help, but these
records can be audited and evaluated without waiting for it. A verified
original protein-level partition must not be claimed.

These are **candidate sequence matches**, not verified source annotations.
The original *DeepKla* and PCBert-Kla publications describe rice-derived
training and *B. cinerea* testing, whereas the mapping artifact matches the
released training and test windows to the *B. cinerea* proteome. The benchmark
files expose only generic `Protein N` names. The initial proteome FASTA was
not retained, but the mapping has now been reproduced using the dated FASTA
and checksums in the follow-up audit. Experimental source proteins,
negative-label construction, and the species discrepancy remain unresolved.
No biological label should be inferred merely from a sequence match.

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

## Source-Derived Dataset Reconstruction

Gao et al. 2020 Supplementary Table 1 supplies documented *B. cinerea*
accession-position pairs for positive Kla sites. A separate reconstruction
starts from those records, rather than assigning inferred accessions to
the original benchmark rows. From the repository root, with the source
supplement and dated proteome at the paths below:

```bash
python3 experiment-1/structure_aware_kla/scripts/reconstruct_gao_sites.py \
  --gao-xlsx experiment-1/structure_aware_kla/source_data/gao_2020_botrytis_lactylome/Table_1.XLSX \
  --proteome-fasta experiment-1/structure_aware_kla/source_data/botrytis_b0510_proteome/uniprot_UP000001798_2026-10-26.fasta.gz \
  --benchmark-train baselines/PCBert-Kla-original/data/train.csv \
  --benchmark-test baselines/PCBert-Kla-original/data/test.csv \
  --output-dir experiment/pcbert_kla_clean/outputs/gao_2020_reconstruction
```

The source files are local inputs and are not committed. The Gao supplement
is linked from the [source article](https://doi.org/10.3389/fmicb.2020.594743);
input checksums are written to `reconstruction_audit.json`. `reported_sites.csv`
retains every source site and explains why a 51-residue window can or cannot
be reconstructed. `candidate_sites.csv` lists eligible positives and
unannotated lysines from the same source proteins. Potential controls whose
site matches a reported modified peptide or whose window matches a known
benchmark/source positive are excluded and, if present, listed separately.

With the currently audited files, 273 reported Kla sites from 166 proteins
produce 208 full 51-residue positive windows. Sixty sites are too close to a
protein terminus; five have no accession in the dated proteome. The same
proteins yield 4,945 unannotated lysines with full windows. These are
**putative negatives**, not experimentally confirmed absence of Kla. No
class balancing or train/validation/test assignment has been applied.

This reconstruction does not yet resolve the reviewers' dataset concerns.
Before model training, remaining candidate controls need independent
positive-site screening and a fixed negative-sampling policy; proteins and
qualifying homologues then need a locked split. Because Gao's sites overlap
the original benchmark, a separate non-overlapping study is still needed
for external testing. Baseline and proposed models must be rerun on the same
locked partitions; historical scores cannot be compared directly to the
new split.

## Candidate-Protein Group Diagnostic

The candidate-mapping audit supports a stricter provisional split without
assuming that each window has only one possible source protein. The command
below first removes duplicate and contradictory windows, then joins all
candidate proteins connected by any window, including excluded contradictory
windows. Each connected group goes to one partition. The candidate-only seed
42 run contains 1,348 training, 194 validation, and 390 test windows; no
candidate accession appears in two partitions. With the documented MMseqs2
full-protein clusters added, the split contains 1,351 training (122 positive),
194 validation (17 positive), and 387 test windows (35 positive). The latter
has no cross-partition qualifying hits in the recorded all-versus-all search.
Neither split verifies original parent-protein identities or constitutes a
previously unseen experimental study.

From `experiment/pcbert_kla_clean/`:

```bash
python3 scripts/run_replication.py --run candidate-group-audit
```

The report and per-window manifest are written to
`outputs/review_candidate_groups/`. The cluster-screened seed 42 manifest is
preserved under `review_data/` as described below. To train on this
provisional split, use `--run candidate-group-diagnostic` with the cluster
and search files and the same model arguments used for `window-diagnostic`.
All models in a comparison must use the same manifest. The output must be
reported as a diagnostic, not as an independent test estimate.

On a CUDA-enabled Colab runtime, the matched baseline and proposed-model
runs use this command, first with `baseline` and then with `token_gated`:

```bash
python3 scripts/run_replication.py \
  --run candidate-group-diagnostic \
  --architecture baseline \
  --homology-clusters-csv review_data/homology_clusters_mmseqs2.csv \
  --homology-hits-tsv review_data/all_vs_all_mmseqs2.tsv \
  --epochs 30 --batch-size 4 --device cuda \
  --optimizer sgd --learning-rate 0.003 --weight-decay 0 --scheduler none \
  --save-models
```

Model comparisons require the same fixed split, training recipe, checkpoint
selection rule, and independently repeated seeds. The originally published
test scores are historical context, not valid comparators on this new split.

### Full-Protein Homology Clustering

The dated B05.10 proteome file is a local audit input and is not committed to
the repository. Once available at the path below, prepare a candidate-only
FASTA from the repository root:

```bash
python3 experiment/pcbert_kla_clean/scripts/prepare_homology_clusters.py export-fasta \
  --mapping-csv experiment-1/structure_aware_kla/results/botrytis_b0510_window_mapping/all_window_matches.csv \
  --proteome-fasta experiment-1/structure_aware_kla/source_data/botrytis_b0510_proteome/uniprot_UP000001798_2026-10-26.fasta.gz \
  --output-fasta experiment/pcbert_kla_clean/outputs/review_candidate_groups/candidate_proteins.fasta
```

With [MMseqs2](https://github.com/soedinglab/MMseqs2) installed, one
explicit full-protein clustering criterion is 30% sequence identity and 80%
coverage of both proteins. Record the MMseqs2 version and exact parameters:

```bash
mmseqs easy-cluster \
  experiment/pcbert_kla_clean/outputs/review_candidate_groups/candidate_proteins.fasta \
  experiment/pcbert_kla_clean/outputs/review_candidate_groups/mmseqs_clusters \
  experiment/pcbert_kla_clean/outputs/review_candidate_groups/mmseqs_tmp \
  --min-seq-id 0.3 -c 0.8 --cov-mode 0 --cluster-mode 0 \
  --cluster-reassign -s 7 --threads 4
python3 experiment/pcbert_kla_clean/scripts/prepare_homology_clusters.py convert-tsv \
  --mapping-csv experiment-1/structure_aware_kla/results/botrytis_b0510_window_mapping/all_window_matches.csv \
  --cluster-tsv experiment/pcbert_kla_clean/outputs/review_candidate_groups/mmseqs_clusters_cluster.tsv \
  --output-csv experiment/pcbert_kla_clean/review_data/homology_clusters_mmseqs2.csv
mmseqs easy-search \
  experiment/pcbert_kla_clean/outputs/review_candidate_groups/candidate_proteins.fasta \
  experiment/pcbert_kla_clean/outputs/review_candidate_groups/candidate_proteins.fasta \
  experiment/pcbert_kla_clean/review_data/all_vs_all_mmseqs2.tsv \
  experiment/pcbert_kla_clean/outputs/review_candidate_groups/search_tmp \
  --min-seq-id 0.3 -c 0.8 --cov-mode 0 -s 7 --max-seqs 1000 \
  --threads 4 --format-output query,target,pident,qcov,tcov
```

Then, from `experiment/pcbert_kla_clean/`, reproduce the saved audit:

```bash
python3 scripts/run_replication.py --run candidate-group-audit \
  --homology-clusters-csv review_data/homology_clusters_mmseqs2.csv \
  --homology-hits-tsv review_data/all_vs_all_mmseqs2.tsv \
  --output-dir review_data
```

The command rejects incomplete cluster files and fails if a supplied cluster
or qualifying all-versus-all search hit spans partitions. The search file has
206 hits, including 180 self-hits, and zero cross-partition hits. Exact tool
version and input checksums are in [the artifact record](review_data/README.md).
This criterion may miss remote or partial-domain similarity; describe it
precisely rather than claiming that all homology is absent.

## Remaining Evaluation

Retrain MM-Kla and the principal PCBert-Kla baseline on the **same** revised
training/validation/test manifest. Do not compare a revised-split score
directly with the published overlap-affected score as if the tests were
equivalent. An additional source-annotated external dataset, screened against
the final training set, is needed to test transfer beyond the reused
benchmark. [KlaDB](https://kladb.wuxl-group.com/download) is a candidate
source, not yet a validated independent test for this project. Putative
negative labels and any study/protein overlap must be documented. If the
original benchmark cannot support a defensible grouped evaluation, move the
main analysis to an accession-annotated dataset rather than implying that
candidate mappings verify original provenance.

For background on sequence leakage and non-overlapping group evaluation, see
[Hou et al., PLOS Computational Biology (2022)](https://doi.org/10.1371/journal.pcbi.1010669)
and the [scikit-learn StratifiedGroupKFold documentation](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedGroupKFold.html).
