# Candidate Study Audit (2026-11-01)

This audit evaluates three independently published lactylation studies as potential sources for a corrected, accession-labelled benchmark. It does **not** establish a finished external test set.

## Sources and Reproduction

| Input | Origin | Recorded identifier |
|---|---|---|
| Rice primary site table | [Meng et al. 2021, Table S1](https://doi.org/10.1021/acs.jafc.1c00760); parsed CSV in this repository | Source XLSX SHA256 `50a05560aedea32b29c99986e2ff21c2aeeaa005ad094dc1ef39914833470573` |
| Curated sites | [KlaDB](https://github.com/wuxl-group/KlaDB), `Database/KlaDB_full_curated_sites.csv.gz` | Git commit `160751dee1619c8bc0a03fa60aa4dcd1d1a64f7b`; file SHA256 in `audit.json` |
| Rice proteins | [UniProtKB](https://www.uniprot.org/) query `organism_id:39947`, FASTA, downloaded 2026-11-01 | UniProt release `2026_03`; compressed FASTA SHA256 `562f03b3650d3a9b6e39b5770bd107a394c49537ee3bc7e9a13a145c9f3f05fc` |
| Gao candidates | [Gao source reconstruction](../../outputs/gao_2020_reconstruction/) | SHA256 in `audit.json` |
| PCBert-Kla original windows | `baselines/PCBert-Kla-original/data/{train,test}.csv` | SHA256 values in `audit.json` |

The primary XLSX and parsed rice CSV are already in the repository. The KlaDB checkout, rice FASTA, and Gao reconstruction are local audit inputs; their checksums are recorded to detect changed source versions. To obtain the rice FASTA again:

```bash
git clone https://github.com/wuxl-group/KlaDB.git /tmp/kladb-audit
git -C /tmp/kladb-audit checkout 160751dee1619c8bc0a03fa60aa4dcd1d1a64f7b
curl -L --fail -o /tmp/uniprot_taxon39947_2026-11-01.fasta.gz \
  'https://rest.uniprot.org/uniprotkb/stream?compressed=true&format=fasta&query=organism_id%3A39947'
sha256sum /tmp/uniprot_taxon39947_2026-11-01.fasta.gz
```

From the repository root, with the KlaDB checkout at `/tmp/kladb-audit` and the downloaded rice FASTA at `/tmp/uniprot_taxon39947_2026-11-01.fasta.gz`:

```bash
python3 experiment-1/structure_aware_kla/scripts/audit_candidate_studies.py \
  --kladb-sites /tmp/kladb-audit/Database/KlaDB_full_curated_sites.csv.gz \
  --rice-table experiment-1/structure_aware_kla/results/meng_2021_table_s1_rice_lactylated_sites.clean.csv \
  --rice-proteome /tmp/uniprot_taxon39947_2026-11-01.fasta.gz \
  --gao-candidates experiment/pcbert_kla_clean/outputs/gao_2020_reconstruction/candidate_sites.csv \
  --benchmark-train baselines/PCBert-Kla-original/data/train.csv \
  --benchmark-test baselines/PCBert-Kla-original/data/test.csv \
  --output-json experiment/pcbert_kla_clean/review_data/source_study_audit_2026_11_01/audit.json \
  --rice-positive-csv experiment/pcbert_kla_clean/review_data/source_study_audit_2026_11_01/rice_verified_positive_sites.csv
```

The command writes `audit.json` and `rice_verified_positive_sites.csv`. A changed UniProt release requires a new audit, not reuse of these counts.

## Findings

- Meng Table S1 contains 638 unique accession-position sites across 342 accessions. All 626 KlaDB rice records match that primary table by accession and position; 12 primary records are absent from KlaDB. In the dated rice FASTA, 11 source-site records have no matching accession and one reported position is not lysine. Another 97 valid sites lack 25 residues on one side of the central lysine. The remaining **529** sites have exact, unpadded, 51-residue windows in both the primary-site/UniProt reconstruction and KlaDB. These 529 experimentally reported positives are exported, with accession, one-based position, and window.
- KlaDB contains 895 wheat sites (376 accessions), with 720 unpadded windows; the [wheat article](https://doi.org/10.3390/ijms242216195) reports 927 sites in 394 proteins. KlaDB contains 783 soybean sites (327 accessions), with 686 unpadded rows but only 550 distinct windows; the [soybean article](https://doi.org/10.1021/acs.jafc.5c02106) reports 783 sites in 327 proteins. Wheat and soybean records have **not** yet been compared against their original supplementary tables or versioned full-protein FASTA files.
- No full curated rice, wheat, or soybean 51-aa window exactly matches a Gao reconstruction candidate or an original PCBert-Kla window. Across the three plant studies, exact full-window overlaps remain: rice/wheat 3, rice/soybean 4, and wheat/soybean 1. These would need to be grouped or removed in any combined split. Exact-window screening alone cannot exclude homologous full proteins.

## Decision Gate

The 529 rice positives provide an auditable starting point for a **candidate** study-held-out evaluation. An external binary test cannot yet be claimed: rice controls would be unobserved rather than experimentally verified negatives, and they need screening against all known rice positives before sampling. The Gao development proteins and rice test proteins also need a documented full-protein homology screen. The external study must remain unused for architecture, hyperparameter, threshold, and checkpoint selection. MM-Kla and PCBert-Kla must be trained and evaluated under the same final protocol.

The wheat and soybean studies remain secondary options until their primary supplements and full-protein sequences have been reconciled. The attempted automated access to the wheat publisher archive returned HTTP 403; the soybean ACS XLSX returned a publisher challenge. Curated counts alone are insufficient to label either study verified.

## Rice Control Candidates and Cross-Study Homology

The screened site candidates in `rice_candidate_sites.csv` are drawn only from the 291 rice proteins with full, verified positive windows. All 638 primary rice sites are excluded from the control pool by accession and position. Additional exclusions use modified-peptide matches in the primary table, complete-window KlaDB positives across species, and the Gao/original benchmark windows. No class balancing or random sampling was applied. The result is **529 reported positives and 5,607 putative negatives**; 35 potential controls matched reported modified peptides, six matched known positive windows, and four matched development windows. Another 842 unreported lysines were too close to a terminus for a full window. The candidate inventory has 39 repeated-window rows. Input/output checksums and the exact counts are in `rice_candidate_audit.json`.

The Gao and rice candidate proteins were screened with [MMseqs2](https://github.com/soedinglab/MMseqs2) `easy-search`, version `18-8cc5c+ds-2`, in **both directions**, using at least 30% sequence identity and 40% coverage of both full proteins (`--cov-mode 0`); the default E-value cutoff was 0.001. A second 80%-coverage search serves as a sensitivity check. The 40%-coverage union contains 70 Gao/rice pairs, involving 36 Gao and **56 rice proteins**. The 80%-coverage union contains 57 pairs involving 45 rice proteins. All 56 rice proteins found under the more conservative 40%-coverage screen were removed from `rice_homology_screened_candidates.csv`. That file retains **409 positives and 4,384 putative negatives on 235 proteins**. It has 20 repeated-window rows and no conflicting labels for an identical window. Pair-level evidence is in `gao_rice_30id_40cov_hits.csv`; input FASTA/search hashes and output hashes are in `cross_study_homology_audit.json`.

The source-data FASTA files and intermediate search outputs are local inputs, not versioned. The commands below reproduce the screen after the Gao reconstruction and rice FASTA described above are available:

```bash
python3 experiment-1/structure_aware_kla/scripts/prepare_rice_candidates.py \
  --rice-table experiment-1/structure_aware_kla/results/meng_2021_table_s1_rice_lactylated_sites.clean.csv \
  --rice-proteome /tmp/uniprot_taxon39947_2026-11-01.fasta.gz \
  --kladb-sites /tmp/kladb-audit/Database/KlaDB_full_curated_sites.csv.gz \
  --verified-positives experiment/pcbert_kla_clean/review_data/source_study_audit_2026_11_01/rice_verified_positive_sites.csv \
  --gao-candidates experiment/pcbert_kla_clean/outputs/gao_2020_reconstruction/candidate_sites.csv \
  --benchmark-train baselines/PCBert-Kla-original/data/train.csv \
  --benchmark-test baselines/PCBert-Kla-original/data/test.csv \
  --output-csv experiment/pcbert_kla_clean/review_data/source_study_audit_2026_11_01/rice_candidate_sites.csv \
  --output-json experiment/pcbert_kla_clean/review_data/source_study_audit_2026_11_01/rice_candidate_audit.json
```

The full-protein FASTA files are generated with `prepare_homology_clusters.py export-fasta`, using the Gao candidate CSV and the dated *B. cinerea* FASTA for one call, and the rice candidate CSV and dated rice FASTA for the other. Both outputs must exactly cover their candidate accession sets; the audit script checks this. With `G` and `R` set to the resulting FASTA paths, the following four searches produce the inputs to `audit_cross_study_homology.py`:

```bash
H=experiment/pcbert_kla_clean/outputs/source_study_homology
G="$H/gao_proteins.fasta"
R="$H/rice_proteins.fasta"
python3 experiment/pcbert_kla_clean/scripts/prepare_homology_clusters.py export-fasta \
  --mapping-csv experiment/pcbert_kla_clean/outputs/gao_2020_reconstruction/candidate_sites.csv \
  --proteome-fasta experiment-1/structure_aware_kla/source_data/botrytis_b0510_proteome/uniprot_UP000001798_2026-10-26.fasta.gz \
  --output-fasta "$G"
python3 experiment/pcbert_kla_clean/scripts/prepare_homology_clusters.py export-fasta \
  --mapping-csv experiment/pcbert_kla_clean/review_data/source_study_audit_2026_11_01/rice_candidate_sites.csv \
  --proteome-fasta /tmp/uniprot_taxon39947_2026-11-01.fasta.gz \
  --output-fasta "$R"
search() {
  mmseqs easy-search "$1" "$2" "$3" "$4" \
    --min-seq-id 0.3 -c "$5" --cov-mode 0 -s 7 --max-seqs 10000 \
    --threads 4 --format-output query,target,pident,qcov,tcov,alnlen,qlen,tlen -v 1
}
search "$G" "$R" "$H/gao_vs_rice_30id_40cov.tsv" "$H/tmp_40" 0.4
search "$R" "$G" "$H/rice_vs_gao_30id_40cov.tsv" "$H/tmp_40_reverse" 0.4
search "$G" "$R" "$H/gao_vs_rice_30id_80cov.tsv" "$H/tmp_80" 0.8
search "$R" "$G" "$H/rice_vs_gao_30id_80cov.tsv" "$H/tmp_80_reverse" 0.8
```

```bash
python3 experiment-1/structure_aware_kla/scripts/audit_cross_study_homology.py \
  --gao-candidates experiment/pcbert_kla_clean/outputs/gao_2020_reconstruction/candidate_sites.csv \
  --rice-candidates experiment/pcbert_kla_clean/review_data/source_study_audit_2026_11_01/rice_candidate_sites.csv \
  --gao-fasta "$G" --rice-fasta "$R" \
  --forward-40 "$H/gao_vs_rice_30id_40cov.tsv" \
  --reverse-40 "$H/rice_vs_gao_30id_40cov.tsv" \
  --forward-80 "$H/gao_vs_rice_30id_80cov.tsv" \
  --reverse-80 "$H/rice_vs_gao_30id_80cov.tsv" \
  --output-dir experiment/pcbert_kla_clean/review_data/source_study_audit_2026_11_01
```

The saved `cross_study_homology_audit.json` records checksums for each input and output. The screen rules out only similarities meeting its explicit thresholds, not remote or shorter domain homology. Rice negatives remain *putative*. The 20 repeated windows, negative-label limitations, fixed test prevalence, and feature regeneration must be addressed before locking an external evaluation or reporting model performance.
