# DeepKla Source-Site Audit

## Evidence

- The original DeepKla and PCBert-Kla train/test files have identical sequence-label multisets.
- DeepKla headers contain only a class label; PCBert-Kla headers use generic row names. Neither provides a verified parent protein accession.
- Gao et al. report 273 distinct Botrytis Kla sites in their supplementary table.
- Exact matches below require both a full-protein accession and a lysine position from that table.

| Benchmark subset | Exact Gao site | Modified peptide only | No Gao site/peptide match |
|---|---:|---:|---:|
| Training positives (distinct windows) | 169 | 3 | 0 |
| Test positives | 174 | 3 | 0 |
| Test negatives | 4 | 1 | 172 |

Of the exact Gao-positive site assignments, 167 occur in both original partitions; only 7 exact Gao sites occur in the test partition without appearing in training.
The four test negatives with an exact Gao site match are label-discordant. One further test negative shares a modified peptide with a Gao site but lacks a current-accession match.

Three distinct positive windows have peptide-only evidence. Their current proteome accessions differ from the Gao table accessions; this is not an independently verified accession crosswalk.

## Mapping Reproduction

- Dated proteome records searched: 13036.
- Candidate-site assignments unchanged from the earlier mapping: 3841/3841.
- Input paths, SHA-256 checksums, and the DeepKla repository commit are recorded in `source_site_audit.json`.
- Record-level candidate and evidence fields are in `benchmark_site_evidence.csv`.

## Interpretation

The positive benchmark windows are strongly consistent with the Gao Botrytis lactylome, including the nominal training positives. This conflicts with the published description of rice-derived training data and requires clarification from the dataset authors. An exact window or peptide match does not prove that every benchmark row originated from the matched protein. In particular, negative rows have no source accession or experimentally verified non-lactylated label. This audit is not a protein-disjoint split.

[DeepKla source](https://github.com/linDing-group/DeepKla) | [Gao et al. 2020](https://doi.org/10.3389/fmicb.2020.594743) | [PCBert-Kla source](https://github.com/ZhangHongqi215/PCBert-Kla)
