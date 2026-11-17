# Auto-Kla and DeepKla source audit

The upstream repositories are retrieved into the ignored
`baselines/upstream_sources/` directory by `fetch_upstream_models.sh`. They
are kept separate from the project implementation and are not redistributed
in this repository.

| Model | Upstream repository | Pinned commit | Available artifacts |
| --- | --- | --- | --- |
| DeepKla | <https://github.com/linDing-group/DeepKla> | `d9f0f5ef987cd65f1562f1ba312e5608ddd44f9e` | Training script and rice-trained `51bp_rusuan.h5` checkpoint |
| Auto-Kla | <https://github.com/tubic/Auto-Kla> | `429f10ea9449dccc4630b670c3a6e734ce33ba56` | Inference scripts and the authors' original datasets; no training script or model weights in GitHub |

DeepKla's author code targets Python 2.7, Keras 2.0.6, and TensorFlow. Its
training script defines a sequence embedding, two 1D convolutions, a
bidirectional GRU, attention, and a sigmoid output. The repository provides
a rice-trained checkpoint. Running that checkpoint on CLEAR-Lactyl would be
an external transfer test, not a retrained, same-split comparison. DeepKla
does not include a license file, so its code and weights are not copied into
this repository.

Auto-Kla's public inference script loads `model_9` with AutoGluon TextPredictor.
Its README links the trained artifact on OneDrive rather than storing it in
GitHub. The link returned HTTP 403 during this audit (2026-11-17). The README
specifies AutoGluon 0.5.2 and Python 3.7-3.9, but no training entry point or
full run configuration is released in the repository. The paper describes an
AutoGluon-selected ELECTRA-based classifier. A newly trained modern AutoGluon
model would need to be labelled **Auto-Kla-inspired**, not the original
Auto-Kla implementation.

For the intended comparison, each model must train on the same locked
CLEAR-Lactyl training partition and be evaluated on the same validation
partition. Published checkpoints or metrics cannot substitute for that
experiment, and the held-out test should remain untouched. Any legacy-code
port, environment change, or replacement training pipeline must be disclosed
before its result is presented as a comparison.

## CLEAR-Lactyl DeepKla-aligned experiment

The project-owned PyTorch port is in
`experiment/pcbert_kla_clean/src/pcbert_kla_clean/deepkla.py`; its
validation-only runner is `scripts/run_deepkla_clear.py`. It retains the
author script's residue-to-ID preprocessing (including the `Y`-to-OOV
behavior), 200-position left padding, 128-dimensional embedding, dropout,
10- and 5-wide 1D convolutions with max-pooling, 50-unit bidirectional GRU,
context attention, sigmoid output, MSE loss, and RMSprop optimizer. Epoch
selection uses validation AUPRC to match this project's benchmark protocol,
not the author's validation-loss stopping rule. PyTorch GRU and RMSprop are
not numerically identical to Keras 2.0.6. Results must be labelled
**DeepKla-aligned PyTorch**, not exact DeepKla reproduction.

The runner reads the locked CLEAR-Lactyl training and validation partitions
and produces `report.json`, `history.json`, `validation_predictions.csv`, and
`best_model.pt`. The internal test is not evaluated. The HPC entry point is
`experiment/pcbert_kla_clean/hpc/project_deepkla_clear.sbatch`; it requires a
Slurm GPU allocation and the pinned project environment.
