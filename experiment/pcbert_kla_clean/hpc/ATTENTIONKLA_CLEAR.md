# AttentionKla on the locked CLEAR-Lactyl development split

This run trains the AttentionKla ESM-2 650M sequence-classification architecture
with fresh LoRA adapters. It does **not** load the authors' released Kla-trained
adapters: their training data overlap the locked validation set.

The run uses five protein-grouped folds of the 20,599 training sites. Each fold
selects its checkpoint by AUPRC on its held-out training fold; the five models'
logits are averaged on the 2,944-site validation partition. The 5,888-site
held-out test partition is not scored. The model and tokenizer are pinned to
`facebook/esm2_t33_650M_UR50D` revision
`08e4846e537177426273712802403f7ba8261b6c`. LoRA targets query, key,
and value projections (rank 8, alpha 32, dropout 0.3). Training uses AdamW,
learning rate 0.001, cosine decay, effective batch size 64, up to 15 epochs,
and patience 2. This is a reproducible architecture-level adaptation, not a
claim to reproduce every hyperparameter of the authors' released model.

Run from the `hpc` directory on the HPC checkout after pulling the committed
code. Set `EXPECTED_COMMIT` to that checkout's full commit hash and submit
`project_attentionkla_clear_development.sbatch` with `sbatch`. The script
refuses to train outside a Slurm compute allocation. Outputs, including fold
histories, adapters, per-site validation scores, and final metrics, go to
`outputs/hpc_clear_lactyl_final/attentionkla/seed_42/development/`.
