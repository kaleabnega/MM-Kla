# Dataset Provenance and Structure-Aware Feasibility Audit

This directory contains a supplementary audit of the public PCBert-Kla/DeepKla
benchmark provenance. It is not part of the active token-gated model training
pipeline in `experiment/pcbert_kla_clean/`, but it documents why a full
structure-aware extension was not used as the main proposed method.

## Purpose

The audit examined whether the public 51-amino-acid benchmark windows could be
traced back to full-length protein sequences and experimentally reported
lactylation sites. Full-protein identifiers and residue positions would be
required for a rigorous graph- or structure-aware Kla model.

## Required Information for Full Structure Modeling

A native full-protein structure workflow requires, for each candidate site:

```text
record_id
protein_accession
full_protein_sequence
full_protein_lysine_position
label
51-aa_window
```

Given these fields, structures could be obtained from AlphaFold DB, ESMFold,
ColabFold, or experimental PDB entries. Local residue graphs could then encode
spatial contacts, solvent accessibility, local charge, secondary structure, and
distance from the central lysine.

## Audit Scripts

```text
scripts/map_windows_to_proteome.py
scripts/extract_xlsx_table.py
scripts/validate_windows_against_modified_peptides.py
scripts/audit_dataset_provenance.py
scripts/audit_mapped_sites.py
scripts/audit_source_site_evidence.py
```

These scripts support:

- exact window-to-proteome mapping,
- extraction of supplementary lactylome tables,
- peptide-level validation against reported modified peptides,
- reproducible provenance summaries.

## Source Materials

The recovered Meng et al. rice lactylome supplementary file is stored under:

```text
source_data/meng_2021_rice_lactylome/jf1c00760_si_001.xlsx
```

The source-site audit also uses Gao et al. 2020 Supplementary Table 1 and a
dated UniProt *B. cinerea* B05.10 proteome under `source_data/`. These source
files are kept separate from the benchmark and identified by SHA-256 checksums
in the audit report.

The extracted and cleaned audit tables are stored under:

```text
results/
```

## Main Finding

The recovered Meng et al. 2021 rice lactylome table is internally consistent
and matches the reported scale of 638 Kla sites across 342 proteins. However,
exact sequence reconstruction from this table does not explain the public
PCBert-Kla/DeepKla training windows. In contrast, the public benchmark windows
all have exact matches in a *Botrytis cinerea* B05.10 proteome mapping. This is
an unresolved discrepancy with the published description of rice-derived
training data; sequence matching alone does not verify the experimental source
of a label.

This provenance mismatch prevents a defensible claim that the current public
benchmark can support a native full-protein rice structure branch without
additional source recovery.

## Reproducible Audit Report

```text
results/dataset_provenance_audit.md
```

The report summarizes:

- DeepKla versus PCBert-Kla sequence-label agreement,
- Meng et al. 2021 Table S1 integrity,
- peptide evidence against benchmark windows,
- reconstructed-window overlap,
- Botrytis proteome mapping.

The per-record mapping audit is generated with:

```bash
python3 experiment-1/structure_aware_kla/scripts/audit_mapped_sites.py \
  --mapping-csv experiment-1/structure_aware_kla/results/botrytis_b0510_window_mapping/all_window_matches.csv \
  --train-windows baselines/PCBert-Kla-original/data/train.csv \
  --test-windows baselines/PCBert-Kla-original/data/test.csv \
  --output-dir experiment-1/structure_aware_kla/results/botrytis_b0510_window_mapping
```

Run the command from the repository root. It produces
`candidate_protein_mapping_audit.json` and `candidate_protein_sites.csv`.
The dated proteome can now be used to rebuild the mapping. The source-site
comparison is recorded in
`results/source_site_audit_2026_10_27/source_site_audit.md` and its JSON/CSV
companions. The audit requires the public DeepKla train/test FASTA files, the
original PCBert-Kla files, the proteome FASTA, and Gao's supplementary XLSX
table. From the repository root, using a checkout of DeepKla at commit
`d9f0f5ef987cd65f1562f1ba312e5608ddd44f9e`:

```bash
python3 experiment-1/structure_aware_kla/scripts/audit_source_site_evidence.py \
  --pcbert-train baselines/PCBert-Kla-original/data/train.csv \
  --pcbert-test baselines/PCBert-Kla-original/data/test.csv \
  --deepkla-train /tmp/deepkla-source-audit/DeepKla/data/upTrain.fa \
  --deepkla-test /tmp/deepkla-source-audit/DeepKla/data/fungiForTest.fa \
  --deepkla-commit d9f0f5ef987cd65f1562f1ba312e5608ddd44f9e \
  --proteome-fasta experiment-1/structure_aware_kla/source_data/botrytis_b0510_proteome/uniprot_UP000001798_2026-10-26.fasta.gz \
  --gao-xlsx experiment-1/structure_aware_kla/source_data/gao_2020_botrytis_lactylome/Table_1.XLSX \
  --previous-mapping-csv experiment-1/structure_aware_kla/results/botrytis_b0510_window_mapping/candidate_protein_sites.csv \
  --output-dir experiment-1/structure_aware_kla/results/source_site_audit_2026_10_27
```

Replace the two `/tmp/deepkla-source-audit/` paths if the public source is
checked out elsewhere. The report records SHA-256 checksums for each input.
It reproduces all 3,841 prior candidate-site assignments and separately
identifies exact reported-site matches, modified-peptide-only matches, and
label-discordant test negatives.

The public DeepKla files do not include source protein IDs. Even source-site
matches cannot assign an authoritative parent protein to every benchmark
negative or every repeated window. A publication split must not use inferred
protein identities without documenting that limitation.

## Interpretation

The structure-aware branch remains a possible future extension if full protein
identifiers and exact full-site positions can be recovered. The active
publishable method in this repository therefore focuses on same-dataset neural
architecture changes using the existing sequence windows and physicochemical
features.
