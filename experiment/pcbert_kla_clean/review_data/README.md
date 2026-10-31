# Cluster-Screened Review Split

This directory records the seed 42 candidate-protein-group diagnostic based on
the original PCBert-Kla/DeepKla windows. It is not a verified reconstruction
of the original source-protein IDs or an independent-study test.

## Inputs and Method

- Candidate mapping: `experiment-1/structure_aware_kla/results/botrytis_b0510_window_mapping/all_window_matches.csv` (SHA-256 `e6cc23a8355864fb8c426fbbd57ab1201f23e3e52c0e3f2ef29cbfebcd3ef341`).
- Dated UniProt B05.10 proteome: `uniprot_UP000001798_2026-10-26.fasta.gz` (SHA-256 `d32997a41cf3bd3839ef55d404c839fdcf47d3065ceda780bbcc736aa1277258`); this local source file is not committed.
- Extracted 180 candidate full-protein sequences: SHA-256 `f5598a85640ab50a574fb5144c8a5bcaf8d7662d3ccc2939d0ae59963d2575ac`.
- MMseqs2 version `503743960fc57733a36951f5fa803cb3389af7e2`; official AVX2 binary downloaded from [mmseqs.com](https://mmseqs.com/latest/mmseqs-linux-avx2.tar.gz) on 2026-10-31.
- Full-protein `easy-cluster`: `--min-seq-id 0.3 -c 0.8 --cov-mode 0 --cluster-mode 0 --cluster-reassign -s 7 --threads 4`.
- All-versus-all `easy-search`: `--min-seq-id 0.3 -c 0.8 --cov-mode 0 -s 7 --max-seqs 1000 --threads 4 --format-output query,target,pident,qcov,tcov`.
- Split seed 42; target fractions 70% training, 10% validation, 20% testing, with no cross-partition candidate accession or qualifying search hit.

MMseqs2 produced 168 clusters from 180 candidates. Joining ambiguous-window
candidate links reduced these to 163 assignment groups. Five contradictory
windows (50 original rows) are excluded before model training or testing.

## Artifacts

| File | Contents |
|---|---|
| `homology_clusters_mmseqs2.csv` | One cluster assignment per candidate protein. |
| `all_vs_all_mmseqs2.tsv` | 206 qualifying search hits, including 180 self-hits. |
| `review_candidate_groups/dataset_audit.json` | Counts, input checksums, and overlap checks. |
| `review_candidate_groups/window_inventory.csv` | Per-window exclusions, candidate matches, group, and final assignment. |

The final manifest contains 1,351 training windows (122 positive), 194
validation windows (17 positive), and 387 test windows (35 positive). The
search hit audit found zero cross-partition hits at the stated criteria.
The original benchmark has generic row names, so its true parent proteins
remain unverified; negatives are putative non-lactylated sites. The reused
benchmark is also not a newly blinded dataset.

See [the evaluation protocol](../REVIEW_REVISION.md) for regeneration and
training commands. The external-study evaluation is separate and pending.
