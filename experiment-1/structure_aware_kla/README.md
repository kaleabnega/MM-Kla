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

The extracted and cleaned audit tables are stored under:

```text
results/
```

## Main Finding

The recovered Meng et al. 2021 rice lactylome table is internally consistent
and matches the reported scale of 638 Kla sites across 342 proteins. However,
exact sequence reconstruction from this table does not explain the public
PCBert-Kla/DeepKla training windows. In contrast, the public benchmark windows
map strongly to the Botrytis cinerea B05.10 proteome.

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

## Interpretation

The structure-aware branch remains a possible future extension if full protein
identifiers and exact full-site positions can be recovered. The active
publishable method in this repository therefore focuses on same-dataset neural
architecture changes using the existing sequence windows and physicochemical
features.
