# KlaDB Human Study-Held-Out Benchmark

## Scope

This protocol uses KlaDB release 2026.09.20 site-and-evidence records for *Homo
sapiens* and the UniProt human reference proteome (UP000005640). It replaces
the original PCBert-Kla/DeepKla train/test files for the revised, accession-
labelled comparison. Published PCBert-Kla scores on the original files are
historical context, not directly comparable estimates on this benchmark.

The development set contains proteins with no evidence from PMID 41192556.
The external test contains proteins whose usable reported Kla evidence comes
only from PMID 41192556, a 2025 human asthenozoospermia study. Proteins with
evidence from both this study and another study are excluded. Complete-protein
homology links and identical 51-residue windows also prevent cross-partition
sharing. This holdout was selected before training: it retains a substantial
development cohort while preserving a separate-study test with 254 positive
windows. No model score informed its selection.

## Source Data

| Source | Local file | SHA-256 |
|---|---|---|
| [KlaDB human site/evidence export](https://kladb.wuxl-group.com/download) | `downloaded-dataset/kladb_homo_sapiens_2026-09-20.csv.gz` | `6d39eda5c7ac2ac4231ba05e77c68065d9cd726a14fed147f0f601a5ae802fce` |
| [UniProt human reference proteome](https://www.uniprot.org/proteomes/UP000005640) | `downloaded-dataset/UP000005640_9606.fasta.gz` | `64299d01e8f41fb6ef03cdf7ec051d3d4e5f041003b6b0fa75b09d439af79dce` |

The first checksum agrees with the [KlaDB release checksums](https://kladb.wuxl-group.com/downloads/checksums.txt). The FASTA was downloaded from the [UniProt reference-proteome file](https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/reference_proteomes/Eukaryota/UP000005640/UP000005640_9606.fasta.gz) on 2026-11-02. Both source files and generated manifests remain untracked locally. KlaDB distributes the site records under CC BY 4.0; the corresponding source publication should also be cited in the manuscript.

## Preparation And Split

From `experiment/pcbert_kla_clean/`, with the two source files at the paths
above:

```bash
python3 scripts/prepare_kladb_human.py \
  --site-csv downloaded-dataset/kladb_homo_sapiens_2026-09-20.csv.gz \
  --proteome-fasta downloaded-dataset/UP000005640_9606.fasta.gz \
  --output-dir outputs/kladb_human_preparation

mmseqs easy-search \
  outputs/kladb_human_preparation/candidate_proteins.fasta \
  outputs/kladb_human_preparation/candidate_proteins.fasta \
  outputs/kladb_human_preparation/all_vs_all.tsv \
  outputs/kladb_human_preparation/mmseqs_search_tmp \
  --min-seq-id 0.3 -c 0.8 --cov-mode 0 -s 7 \
  --max-seqs 5000 --threads 4 \
  --format-output query,target,pident,qcov,tcov

python3 scripts/lock_kladb_human.py \
  --candidate-csv outputs/kladb_human_preparation/candidate_sites.csv \
  --preparation-audit outputs/kladb_human_preparation/preparation_audit.json \
  --homology-hits-tsv outputs/kladb_human_preparation/all_vs_all.tsv \
  --output-dir outputs/kladb_human_locked \
  --holdout-pmid 41192556 --seed 42 --train-negative-ratio 3

python3 scripts/run_replication.py --run human-study-check
```

The homology search used MMseqs2 build
`21fbeafa3eaed712b82b5743b78f1a36690e7d51`. The split joins proteins
connected by any qualifying full-protein hit at >=30% identity and >=80%
coverage of both proteins, then joins proteins sharing any exact candidate
window. The development component assignment targets 70% training, 10%
validation, and 20% internal test by candidate row count, with seed 42.
Identical windows are deduplicated within each partition. Only training
negatives are sampled, at most three per training positive; validation and
both test sets retain the full putative-negative pool. All reported Kla sites
in KlaDB, including sites excluded for incomplete windows, are withheld from
negative generation on the same accession.

## Locked Counts

| Partition | Positive | Putative negative | Total | Proteins |
|---|---:|---:|---:|---:|
| Training before negative sampling | 9,970 | 113,597 | 123,567 | 2,642 |
| Training after negative sampling | 9,970 | 29,910 | 39,880 | 2,642 |
| Validation | 1,436 | 16,277 | 17,713 | 380 |
| Internal test | 2,864 | 32,591 | 35,455 | 751 |
| External study test | 254 | 4,841 | 5,095 | 81 |

The initial 33,978 evidence rows represent 18,908 distinct reported sites.
Among them, 42 accession-position sites are absent from the downloaded
reference proteome, three have an invalid central residue, and 2,292 have
terminal padding or noncanonical residues. This leaves 16,571 usable positive
site candidates before study partitioning. The candidate inventory contains
176,569 putative negative sites after excluding 242 negative windows identical
to any reported positive window. The locked split excludes mixed-study and
homology-linked groups, then removes 525 repeated windows within partitions.

## Evaluation Boundary

The two architectures, PCBert-Kla baseline and MM-Kla token-gated, must use
the **same `model_manifest.csv`**, validation selection rule, fixed test
threshold, training budget, and seeds. Training-only decisions may use the
validation partition. The internal and external tests remain unused until the
model settings are frozen. AUPRC, AUC, recall, precision, MCC, and confusion
counts should be reported; high accuracy alone is misleading at this
prevalence.

```bash
python3 scripts/run_replication.py --run human-study-development \
  --architecture baseline --seed 42 --epochs 30 --batch-size 4 --device cuda

python3 scripts/run_replication.py --run human-study-development \
  --architecture token_gated --seed 42 --epochs 30 --batch-size 4 --device cuda
```

The commands above are development runs only. `--run human-study` performs
the locked internal and external evaluations after settings are fixed. Full
30-epoch runs over 39,880 training examples may be expensive on free-tier
GPUs. A separate pilot manifest can limit only the training subset while
preserving the locked validation and test partitions:

```bash
python3 scripts/lock_kladb_human.py \
  --candidate-csv outputs/kladb_human_preparation/candidate_sites.csv \
  --preparation-audit outputs/kladb_human_preparation/preparation_audit.json \
  --homology-hits-tsv outputs/kladb_human_preparation/all_vs_all.tsv \
  --output-dir outputs/kladb_human_pilot \
  --holdout-pmid 41192556 --seed 42 --train-negative-ratio 3 \
  --train-positive-cap 500

python3 scripts/run_replication.py --run human-study-development \
  --human-lock outputs/kladb_human_pilot/split_audit.json \
  --architecture baseline --seed 42 --epochs 1 --batch-size 4 --device cuda
```

Pilot results are for runtime and pipeline validation only. The primary
comparison must use the uncapped `kladb_human_locked` manifest for both models,
or state and justify a new common training budget before testing.

## Limitations

- Negative sites are unreported lysines, not experimentally verified
  non-lactylated lysines. The study-specific detection coverage is unknown.
- UniProt reference sequences may differ from some experimental isoforms;
  exact accession, position, and window matches are required for inclusion.
- The homology screen controls qualifying full-length similarity but does not
  prove absence of shorter shared domains or local sequence motifs.
- The external test is study-held-out and protein-disjoint by this protocol,
  but its 254 positive windows arise from only 81 proteins. Uncertainty should
  be reported with protein-group-aware intervals.
- A single human study is not evidence of cross-species generalization.

The audit files in `outputs/kladb_human_preparation/` and
`outputs/kladb_human_locked/` record the input hashes, exclusion counts,
partition counts, and output hashes. They are local artifacts, not trained
model results.

For Kaggle execution, the locked `model_manifest.csv` and `split_audit.json`
must be made available together as private input data, or regenerated in the
Kaggle environment from the pinned public sources. `--human-lock` accepts the
mounted path to `split_audit.json`; its adjacent `model_manifest.csv` is
verified by checksum at load time. No GitHub access token is needed when only
the public source files and a private Kaggle dataset are used.

The [Kaggle human notebook](../../kaggle_human_study.ipynb) additionally clones
this private code repository through a Kaggle Secret named `GITHUB_TOKEN`. It
expects exactly the v1 locked pair listed above, runs the two architectures on
separate GPUs when available, and leaves test evaluation disabled until the
training settings are fixed. Each final saved version evaluates one predeclared
seed and writes a verified result bundle. Earlier notebook outputs can be
attached as Kaggle inputs for the cross-seed summary. Notebook output is
written to `/kaggle/working`.
