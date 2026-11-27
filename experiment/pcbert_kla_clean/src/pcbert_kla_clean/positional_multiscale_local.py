"""Multiscale local motif encoder for the single positional predictor."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from pcbert_kla_clean.positional_order_context_development import OrderContextPositionalBranch


class MultiscaleLocalOrderContextPositionalBranch(OrderContextPositionalBranch):
    """Add short-range motif filters before the existing local BiLSTM."""

    def __init__(self) -> None:
        super().__init__()
        self.local_scales = nn.ModuleList(
            nn.Conv1d(64, 64, width, padding=width // 2, groups=64)
            for width in (3, 5, 9)
        )
        self.local_projection = nn.Conv1d(192, 64, 1)
        self.local_norm = nn.LayerNorm(64)
        self.local_dropout = nn.Dropout(0.2)

    def encode_local(self, local_tokens: torch.Tensor) -> torch.Tensor:
        embedded = super().encode_local(local_tokens)
        channels_first = embedded.transpose(1, 2)
        motifs = torch.cat([layer(channels_first) for layer in self.local_scales], dim=1)
        refinement = self.local_projection(F.gelu(motifs)).transpose(1, 2)
        return self.local_norm(embedded + self.local_dropout(refinement))
