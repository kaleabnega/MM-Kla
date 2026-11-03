# Revised PCBert-Kla Released-Data Benchmark, v1

This benchmark reuses the [PCBert-Kla released train and test files](../../../../baselines/PCBert-Kla-original/data) and their 27 released physicochemical features. It is a **new internal partition**, not the original published train/test evaluation or a new external study. The scripts freeze the same split for PCBert-Kla and the token-gated model.

## Why Repartition

The original test has 354 rows, including 177 positives. Of those, 255 rows exactly match a training sequence; 170 are positive test rows. Dropping overlapping test rows while retaining the old training set would leave **99 test rows and only 7 positives**, before further duplicate or label-conflict handling. A seven-positive test is too unstable for a credible primary comparison. Keeping the test and removing matching training rows would instead leave very few distinct positive training windows. Therefore, all released rows are audited together and repartitioned after deduplication.

## Lock Protocol

1. Verify the SHA-256 hashes of the four released files, the candidate-protein mapping, and the precomputed full-protein MMseqs2 clusters and hits.
2. Remove windows with contradictory labels. The original audit found five such windows, representing 50 rows. Confirm duplicate windows have consistent released feature vectors, then retain one example per eligible window.
3. Group windows sharing any candidate accession or a qualifying full-protein similarity component. The precomputed search uses at least 30% identity and 80% coverage of both proteins; no qualifying hit crosses the resulting partitions.
4. Assign groups deterministically to training, validation, and internal test (seed 42; target fractions 70/10/20). Sample at most three released negative-labelled training windows per positive **after** splitting. Keep the full negative-labelled validation and test pools.

| Partition | Positive | Released negative label | Total |
| --- | ---: | ---: | ---: |
| Training before negative sampling | 122 | 1,229 | 1,351 |
| Training used by models | 122 | 366 | 488 |
| Validation | 17 | 177 | 194 |
| Internal test | 35 | 352 | 387 |

The reproducible lock consists of `full_manifest.csv`, `model_manifest.csv`, and `split_audit.json`. All three are committed. Rebuild them with:

```bash
cd experiment/pcbert_kla_clean
python scripts/lock_pcbert_revised.py --output-dir review_data/pcbert_revised_v1
python scripts/run_replication.py --run pcbert-revised-check
```

The [Kaggle notebook](../../kaggle_pcbert_revised.ipynb) checks byte-for-byte regeneration and runs matched PCBert-Kla/MM-Kla experiments across predeclared seeds. It reports descriptive candidate-group bootstrap intervals for AUPRC and AUC. The internal test must not determine hyperparameters, thresholds, architecture changes, or seed selection. The released negative labels are not experimentally confirmed non-lactylated sites.

## Remaining Limits

The released rows use generic names, not verified source-protein IDs. In the retained unique windows, 172 of 174 positive windows match an accession and lysine position in [Gao et al.'s published Botrytis site table](https://doi.org/10.3389/fmicb.2020.594743); the other two have modified-peptide matches only. None of the 1,758 retained negative-labelled windows has an exact Gao site match. These observations corroborate positive *sites*, but do not establish which protein supplied every original benchmark row or prove that unmatched lysines are non-lactylated. The [record-level source audit](../../../../experiment-1/structure_aware_kla/results/source_site_audit_2026_10_27/source_site_audit.md) documents the matching evidence.

Candidate accessions are exact proteome sequence matches, and their homology groups are a conservative screen, **not proof of the original parent-protein assignments**. Neither [DeepKla's released FASTA files](https://github.com/linDing-group/DeepKla/tree/main/DeepKla/data) nor [PCBert-Kla's released CSV files](https://github.com/ZhangHongqi215/PCBert-Kla/tree/main/data) provide a row-level source-accession crosswalk. Without those original records, this limitation cannot be removed by assigning the most likely match. An accession-labelled *new* benchmark can instead be reconstructed from Gao's source table; this is already represented by the separate [Gao source-derived split](../gao_2020_source_split/source_split_audit.json), but it is not the original PCBert-Kla dataset.

This project has already used the released rows for model development, so the new internal test is retrospective and not freshly blinded. No independent-study or verified protein-disjoint claim is possible from this benchmark alone. Results should be reported alongside the separate accession-labelled KlaDB evaluation, with the two datasets and their different protocols clearly distinguished.
