# HybridKla-derived human-to-mouse benchmark, v1

This benchmark uses human sites from [HybridKla supplementary Table S1](https://pmc.ncbi.nlm.nih.gov/articles/PMC12309240/) for development and the paper-designated mouse Table S7 as an external source. It is **not** the authors' original training workbook split. The same locked `model_manifest.csv` is used for PCBert-Kla, MM-Kla, and architecture-aligned HybridKla retraining. Original published scores are not directly comparable to this protocol.

## Source and eligibility

`prepare_hybridkla_study.py` pins the SHA-256 hashes of S1, S7, and the UniProt human and mouse reference proteomes. An accession and site are retained only when the reported 51-residue window exactly matches the reference sequence, has a central lysine, and contains canonical residues. The preparation also excludes proteins over 8,192 residues because the pinned MMseqs2 search omitted their self-hits. Known S1 positive windows from all species and all S7 sites are protected against negative assignment. Other canonical lysines on positive-bearing proteins are **putative negatives**, not experimentally confirmed non-lactylated sites.

All-vs-all full-protein MMseqs2 hits use at least 30% sequence identity and 80% coverage of both sequences. Exact duplicate windows are joined before partitioning. Any homology component containing both human S1 and mouse S7 candidates is excluded from both development and external evaluation. The remaining human groups are deterministically split into training, validation, and internal test. Training negatives are sampled at no more than three per positive **after** partitioning; the validation and test pools retain all eligible putative negatives.

## Audited counts

The local preparation retained 14,183 reference-matched S1 human positives and 1,088 S7 mouse positives after source-level exclusions. The lock removed 204 cross-source homology groups, containing 11,971 human candidate rows and 7,736 mouse candidate rows. It also removed 523 repeated windows within retained partitions.

| Partition | Positives | Putative negatives | Proteins |
| --- | ---: | ---: | ---: |
| Training, before negative sampling | 7,912 | 90,193 | 2,265 |
| Validation | 1,140 | 12,905 | 332 |
| Internal test | 2,251 | 25,985 | 646 |
| Mouse external test | 64 | 848 | 42 |

The model training manifest contains 31,648 training rows after negative sampling. Generated manifests and audits are under the ignored `outputs/hybridkla_study_preparation/` and `outputs/hybridkla_study_locked/` directories. The notebook [kaggle_hybridkla_study.ipynb](../../kaggle_hybridkla_study.ipynb) rebuilds them from pinned source inputs and records the software version and Git commit.

## Interpretation limits

S1 and S7 do not provide per-site study identifiers. The paper designates S7 as mouse evaluation data, but site-level study independence cannot be independently verified from these tables. The external set's 64 positive sites make estimates sensitive to a few predictions; confidence intervals and the positive count should accompany results. This protocol addresses exact-window and qualifying full-protein similarity leakage for retained records; it does not prove that every homologous relationship is absent or that putative negatives are true negatives. No test partition may be used for hyperparameter, threshold, or architecture selection.
