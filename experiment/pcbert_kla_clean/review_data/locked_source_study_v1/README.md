# Locked Source-Study Evaluation v1

This dataset is a **replacement evaluation protocol**, not the original PCBert-Kla/DeepKla benchmark. The original nominal test shares 255 exact windows with training and is unsuitable as an independent estimate. Both PCBert-Kla and MM-Kla must be retrained on this same source-derived protocol before any corrected comparison is reported.

## Partitions and Labels

| Partition | Study | Reported Kla | Putative negative | Total |
|---|---|---:|---:|---:|
| Train | Gao 2020, *B. cinerea* | 145 | 3,454 | 3,599 |
| Validation | Gao 2020, *B. cinerea* | 21 | 498 | 519 |
| Internal test | Gao 2020, *B. cinerea* | 42 | 993 | 1,035 |
| External test | Meng 2021, rice | 409 | 4,364 | 4,773 |

The Gao split is accession- and MMseqs homology-cluster-disjoint. The external rice set is from a different study and species, after excluding proteins with Gao hits at the documented 30% identity/40% reciprocal-coverage threshold. It has no exact 51-aa window shared with Gao. These screens do **not** exclude every possible remote or short-domain homologue.

The 20 repeated rice-window rows were reduced to one representative per exact window: lexicographically first accession, then lowest 1-based position. This does not alter labels; no contradictory repeated windows were present. The full pre-deduplication inventory remains in `../source_study_audit_2026_11_01/rice_homology_screened_candidates.csv`. The external class prevalence is about 8.6%; no balancing was applied.

Reported positive sites have primary-study evidence. Negative rows are **unannotated lysines, not experimentally verified non-lactylated sites**. Screening removed known sites and matching modified peptides where possible, but incomplete Kla annotation can produce false negatives. Binary specificity, precision, and AUPRC therefore measure agreement with these *operational* labels, not definitive biochemical absence of Kla. The rice result also measures cross-species transfer, which is harder to interpret than a same-species independent cohort.

`protocol_lock.json` stores SHA-256 digests of the source split, cross-study homology audit, screened rice candidates, and final external manifest. The loader refuses changed inputs. Recreating the lock is explicit:

```bash
python3 scripts/lock_source_study.py
python3 scripts/run_replication.py --run source-study-check
```

From `experiment/pcbert_kla_clean/`, train each architecture with the same backbone, seed, optimizer, 30 epochs, feature normalization, validation-AUPRC checkpoint rule, and fixed threshold. The 27 physicochemical features are regenerated from each window with the audited PCBert-Kla/Biopython definition. The scaler is fitted only on Gao training rows. Neither Gao internal test nor rice external test is used for model selection.

```bash
python3 scripts/run_replication.py --run source-study \
  --architecture baseline --model-name Rostlab/prot_bert \
  --epochs 30 --batch-size 4 --device cuda \
  --optimizer sgd --learning-rate 0.003 --weight-decay 0 \
  --scheduler none --seed 42 --save-models

python3 scripts/run_replication.py --run source-study \
  --architecture token_gated --model-name Rostlab/prot_bert \
  --epochs 30 --batch-size 4 --device cuda \
  --optimizer sgd --learning-rate 0.003 --weight-decay 0 \
  --scheduler none --seed 42 --save-models
```

Each command fits one model on Gao train, selects its checkpoint using Gao validation AUPRC, then evaluates **that same checkpoint** on Gao internal and rice external tests. Outputs are under `outputs/source_study/runs/<configuration-hash>/`: run configuration, test metrics, predictions, model weights, and feature scaler. Do not select a backbone, threshold, or final architecture based on rice results and then report the same rice set as untouched external validation. Changes to the dataset or evaluation rule require a new protocol version.

No trained-model results for this corrected protocol have been produced or claimed yet.
