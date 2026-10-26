# Physicochemical Feature Specification

The supplied `feature_train.csv` and `feature_test.csv` contain 27 unscaled
descriptors per candidate site. The original `PCBert-Kla.ipynb` computes them
with `Bio.SeqUtils.ProtParam.ProteinAnalysis` on the **51-residue input window**,
after removing any spaces. They describe the window, not the complete parent
protein. Column order is retained exactly in the cleaned runner.

| CSV column | Definition on the 51-residue window | Biopython operation |
|---|---|---|
| Molecular Weight Calculation | Calculated molecular mass of the unmodified peptide, in Da. | `molecular_weight()` |
| Isoelectric Point Calculation | Calculated pH at which modeled net charge is zero. | `isoelectric_point()` |
| A | Alanine count / 51. | Amino-acid composition |
| C | Cysteine count / 51. | Amino-acid composition |
| D | Aspartate count / 51. | Amino-acid composition |
| E | Glutamate count / 51. | Amino-acid composition |
| F | Phenylalanine count / 51. | Amino-acid composition |
| G | Glycine count / 51. | Amino-acid composition |
| H | Histidine count / 51. | Amino-acid composition |
| I | Isoleucine count / 51. | Amino-acid composition |
| K | Lysine count / 51, including the central candidate lysine. | Amino-acid composition |
| L | Leucine count / 51. | Amino-acid composition |
| M | Methionine count / 51. | Amino-acid composition |
| N | Asparagine count / 51. | Amino-acid composition |
| P | Proline count / 51. | Amino-acid composition |
| Q | Glutamine count / 51. | Amino-acid composition |
| R | Arginine count / 51. | Amino-acid composition |
| S | Serine count / 51. | Amino-acid composition |
| T | Threonine count / 51. | Amino-acid composition |
| V | Valine count / 51. | Amino-acid composition |
| W | Tryptophan count / 51. | Amino-acid composition |
| Y | Tyrosine count / 51. | Amino-acid composition |
| Helix | Fraction of residues in the helix-associated set E, M, A, L, K. | `secondary_structure_fraction()[0]` |
| Turn | Fraction in the turn-associated set N, P, G, S, D. | `secondary_structure_fraction()[1]` |
| Sheet | Fraction in the sheet-associated set V, I, Y, F, W, L. | `secondary_structure_fraction()[2]` |
| Hydrophobicity Calculation | Grand average of Kyte-Doolittle hydropathy scores (GRAVY). | `gravy()` |
| Net Charge Calculation | Modeled peptide net charge at pH 7.0. | `charge_at_pH(7.0)` |

The Helix, Turn, and Sheet values are **composition-based propensities**, not
secondary-structure assignments from a 3D model or experiment. Their residue
sets can overlap, so these three values need not sum to one. Molecular mass,
charge, and isoelectric point describe the unmodified amino-acid window; they
are not measured after lactylation.

The original notebook calls `get_amino_acids_percent()`, which returned
fractions between zero and one in the Biopython version used for those CSVs.
In newer Biopython releases, `amino_acids_percent` returns percentages between
zero and 100, so reproduction code divides those values by 100. The exact
Biopython version used to create the original CSVs was not pinned in the
upstream notebook.

The CSV values are loaded without normalization. During training, the cleaned
implementation fits `sklearn.preprocessing.MinMaxScaler` on the training
partition only and applies that fitted transform to validation and test rows.
For five-fold cross-validation it fits a new scaler inside each fold. The
scaler uses per-feature training minimum and maximum, mapping observed
training values to the 0-1 range; held-out values can fall outside that range.

Feature reproduction can be checked with:

```bash
python3 scripts/audit_features.py
```

The audit recomputes all 27 columns and writes a machine-readable report to
`outputs/review_exact_window/feature_audit.json`. With Biopython 1.88, all
3,841 supplied rows and 27 columns match within a relative tolerance of
`1e-5` and absolute tolerance of `1e-3`; the largest absolute difference is
approximately `0.000244` Da from float32 conversion. Method definitions follow the
[Biopython `ProteinAnalysis` source](https://github.com/biopython/biopython/blob/master/Bio/SeqUtils/ProtParam.py).
