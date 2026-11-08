# Revised PCBert-Kla vs KlaDB Human: Cross-Benchmark Overlap Audit

This audit compares the locked revised PCBert-Kla benchmark with the locked
KlaDB human benchmark. It tests whether KlaDB can serve as a separate-source
inference set for models trained on the revised PCBert-Kla partition. It does
not establish the true parent proteins of the released PCBert-Kla windows.

## Inputs

| Input | SHA-256 |
| --- | --- |
| Revised `full_manifest.csv` | `bb57684bb09349149ab38b84ff5af21096eaa084b3b45ba437fa88f5db8a2596` |
| KlaDB human `full_manifest.csv` | `2a46fd62b1d4d67ded1c31d8a9956f48eab6b7c33aba146f9d9f24c267b34748` |
| Revised candidate proteins FASTA (180 sequences) | `f5598a85640ab50a574fb5144c8a5bcaf8d7662d3ccc2939d0ae59963d2575ac` |
| KlaDB candidate proteins FASTA (4,180 sequences) | `6ccb7f01fec221980eaa82dd5020cb38fb724759f8d4ede18628354c9fa01c80` |

The revised manifest is in `review_data/pcbert_revised_v1/`; its candidate FASTA
is in `outputs/review_candidate_groups/`. The KlaDB manifest is in
`outputs/kladb_human_locked/`; its candidate FASTA is in
`outputs/kladb_human_preparation/`. The generated KlaDB inputs are not tracked
in Git and must be regenerated or supplied with their recorded hashes.

## Search

MMseqs2 `18-8cc5c+ds-2` searched the complete candidate-protein FASTAs in
both directions with `--min-seq-id 0.3 -c 0.8 --cov-mode 0 -s 7
--max-seqs 5000 --threads 4 --format-output query,target,pident,qcov,tcov`.
The default E-value threshold was `0.001`. A pair qualifies only when the
reported identity is at least 30% and **both** query and target coverage are
at least 80%. The directional search outputs are retained here as
`revised_to_kladb.tsv` and `kladb_to_revised.tsv`; their SHA-256 hashes are
`fe17d4dc1a9a098c5ad8faed7a7062f2aeae74013079df58e23eae2b3396e034`
and `7c3c39807690097f1119fba19a66fd1eb8e7da0f841c424638135b60fe4c0412`.
They contain 186 and 182 qualifying directional hits, respectively, or 188
distinct accession pairs after reversing the second search and taking the
union.

## Findings

Across every revised partition and every KlaDB partition there are **zero
identical 51-residue windows**, **zero shared accession IDs**, and **zero
identical complete candidate-protein sequences**. All 124 candidate accessions
associated with the full revised training partition, and all 81 accessions in
the KlaDB external-study test, occur in the searched FASTAs.

The table counts KlaDB proteins with a qualifying similarity hit to **any
candidate protein in the full revised training partition** (1,351 windows,
before negative sampling). Rows and positives are from the locked KlaDB
`full_manifest.csv`.

| KlaDB partition | Total rows | Homologous proteins | Affected rows | Affected positives |
| --- | ---: | ---: | ---: | ---: |
| Training | 123,567 | 42 | 883 | 195 |
| Validation | 17,713 | 2 | 39 | 9 |
| Internal test | 35,455 | 13 | 318 | 62 |
| External-study test | 5,095 | **0** | **0** | **0** |

The 488-row training subset actually used by the revised PCBert-Kla and MM-Kla
models also has zero qualifying hits to the KlaDB external-study test. The
external-study partition has 254 positives and 4,841 putative negatives from
81 proteins. In contrast, the **whole KlaDB benchmark is not protein-disjoint
from the full revised training partition at this search threshold**.

## Interpretation and limits

The KlaDB external-study test is the appropriate predeclared candidate for
inference-only evaluation of the revised-data models. No qualifying cross-
benchmark hit was detected for that partition under the stated search. This
is a thresholded homology screen, not proof that no shorter shared domains,
remote homologues, or unknown source-protein relationships exist. The revised
PCBert-Kla accessions are sequence-matched candidates, **not verified original
parent-protein IDs**. The KlaDB data have also been examined previously during
this project, so this is not a pristine, never-inspected test.

External inference must use the previously selected checkpoints, fixed
decision rules, and training-fitted feature preprocessing. Training or
threshold selection on KlaDB would answer a different question.
