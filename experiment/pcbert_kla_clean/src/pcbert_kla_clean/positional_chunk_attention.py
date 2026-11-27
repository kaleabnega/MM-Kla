"""Lysine-conditioned residue pooling within ordered full-protein chunks."""

from __future__ import annotations

import torch

from pcbert_kla_clean.positional_global_development import CHUNKS
from pcbert_kla_clean.positional_order_context_development import OrderContextPositionalBranch


def attentive_chunk_pool(states: torch.Tensor, query: torch.Tensor,
                         lengths: torch.Tensor) -> torch.Tensor:
    if (states.ndim != 3 or query.shape != (states.shape[0], states.shape[2]) or
            lengths.shape != (states.shape[0],) or
            torch.any(lengths < 1) or torch.any(lengths > states.shape[1])):
        raise ValueError("Invalid protein states, query, or lengths")
    batch_size, padded_length, width = states.shape
    positions = torch.arange(padded_length, device=states.device)[None, :]
    valid = positions < lengths[:, None]
    bins = torch.clamp(positions * CHUNKS // lengths[:, None], max=CHUNKS - 1)
    bins = bins.expand(batch_size, -1)
    logits = (states * query[:, None, :]).sum(dim=-1) * (width ** -0.5)
    logits = logits.masked_fill(~valid, -torch.inf)
    maximum = states.new_full((batch_size, CHUNKS), -torch.inf)
    maximum.scatter_reduce_(1, bins, logits, reduce="amax", include_self=True)
    maximum = torch.where(torch.isfinite(maximum), maximum, 0)
    weights = torch.exp(logits - maximum.gather(1, bins))
    totals = states.new_zeros((batch_size, CHUNKS))
    totals.scatter_add_(1, bins, weights)
    pooled = states.new_zeros((batch_size, CHUNKS, width))
    pooled.scatter_add_(1, bins[:, :, None].expand(-1, -1, width),
                        states * weights[:, :, None])
    return pooled / totals.clamp_min(1e-12)[:, :, None]


class ChunkAttentiveOrderContextPositionalBranch(OrderContextPositionalBranch):
    """Retain chunk order while letting the target site select residues in each chunk."""

    def encode_context(self, proteins: torch.Tensor, lengths: torch.Tensor,
                       site_positions: torch.Tensor, center: torch.Tensor) -> torch.Tensor:
        if (torch.any(site_positions < 0) or torch.any(site_positions >= lengths) or
                torch.any(lengths > proteins.shape[1])):
            raise ValueError("Invalid site position or protein length")
        embedded = self.protein_residue(proteins).transpose(1, 2)
        states = torch.nn.functional.gelu(
            self.protein_pointwise(self.protein_depthwise(embedded))
        ).transpose(1, 2)
        pooled = attentive_chunk_pool(states, self.global_query(center), lengths)
        chunk_ids = torch.arange(CHUNKS, device=proteins.device)
        site_bin = torch.clamp(site_positions * CHUNKS // lengths, max=CHUNKS - 1)
        marker = (chunk_ids[None, :] == site_bin[:, None]).long()
        return pooled + self.global_position(chunk_ids) + self.site_marker(marker)
