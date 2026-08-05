# PCBert-Kla Original Baseline

This directory stores the cloned PCBert-Kla baseline materials used as the
reference benchmark for this project. The baseline directory is retained to
preserve the original data files and notebook-style implementation separately
from the cleaned experimental framework.

## Contents

```text
PCBert-Kla.ipynb
data/
├── train.csv
├── test.csv
├── feature_train.csv
└── feature_test.csv
```

The active implementation does not modify the baseline data files. The cleaned
runner in `experiment/pcbert_kla_clean/` reads these files directly for
replication, proposed-method evaluation, backbone comparison, and ablation
experiments.

## Benchmark Role

The baseline provides:

- 51-amino-acid lysine-centered sequence windows,
- binary Kla/non-Kla labels,
- physicochemical and residue-composition feature tables,
- the original independent test split used for fair comparison.

The original notebook is retained for reference. Reproducible experiments for
this repository are conducted through:

```text
experiment/pcbert_kla_clean/scripts/run_replication.py
experiment/pcbert_kla_clean/main.ipynb
```
