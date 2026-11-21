# Held-out result reconstruction

`input/` contains the saved seed-42 per-site predictions from the locked
CLEAR-Lactyl internal test. The files originated from the HPC result bundles
`hpc_clear_lactyl_final/saved`, `hpc_clear_lactyl_final/stacked`, and
`hpc_clear_lactyl_final/transkla`. They are model outputs, not new training
data. Input SHA-256 values and the locked manifest hash are recorded in
`../revision/figures_generated/analysis_summary.json`.

`generate_results.py` verifies the accession, position, and label of every
model score against `experiment/pcbert_kla_clean/outputs/clear_lactyl_locked/
model_manifest.csv`. It rejects missing, duplicated, nonfinite, and out-of-range
scores. The script then regenerates threshold-0.5 metrics, ROC/PR curves,
confusion matrices, predictor comparison, sequence-context heat map, and
paired homology-group bootstrap intervals. All generated manuscript artifacts
are written to `../revision/figures_generated/`.

Dependencies: Python 3.10+, NumPy, pandas, SciPy, scikit-learn, and
Matplotlib. Reproduction command from repository root:

```bash
python3 experiment/manuscript/analysis/generate_results.py
```

This analysis is conditional on a single trained seed and one locked source
release. Resampling held-out homology groups does not quantify variation
from retraining or experimental-study differences.
