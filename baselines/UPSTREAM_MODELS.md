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
