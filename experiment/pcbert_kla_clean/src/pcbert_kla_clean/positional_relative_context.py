"""Site-relative multiscale full-protein context for the positional predictor."""

from __future__ import annotations

import torch
from torch import nn

from pcbert_kla_clean.positional_order_context_development import OrderContextPositionalBranch


RELATIVE_BOUNDARIES = (-200, -50, -25, 0, 1, 26, 51, 201)
RELATIVE_BINS = len(RELATIVE_BOUNDARIES) + 1


def relative_bin_ids(proteins: torch.Tensor, lengths: torch.Tensor,
                     site_positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if (torch.any(lengths < 1) or torch.any(lengths > proteins.shape[1]) or
            torch.any(site_positions < 0) or torch.any(site_positions >= lengths)):
        raise ValueError("Invalid protein length or site position")
    positions = torch.arange(proteins.shape[1], device=proteins.device)[None, :]
    distance = positions - site_positions[:, None]
    boundaries = torch.tensor(RELATIVE_BOUNDARIES, device=proteins.device)
    bins = torch.bucketize(distance.contiguous(), boundaries, right=True)
    return bins, positions < lengths[:, None]


class RelativeContextPositionalBranch(OrderContextPositionalBranch):
    """Use signed distance bands around K instead of equal-length protein chunks."""

    def __init__(self) -> None:
        super().__init__()
        self.global_position = nn.Embedding(RELATIVE_BINS, 64)

    def global_padding_mask(self, proteins: torch.Tensor, lengths: torch.Tensor,
                            site_positions: torch.Tensor) -> torch.Tensor:
        bins, valid = relative_bin_ids(proteins, lengths, site_positions)
        counts = torch.zeros((len(proteins), RELATIVE_BINS), device=proteins.device,
                             dtype=torch.long)
        counts.scatter_add_(1, bins, valid.long())
        return counts == 0

    def encode_protein(self, proteins: torch.Tensor, lengths: torch.Tensor,
                       site_positions: torch.Tensor) -> torch.Tensor:
        bins, valid = relative_bin_ids(proteins, lengths, site_positions)
        embedded = self.protein_residue(proteins).transpose(1, 2)
        states = torch.nn.functional.gelu(
            self.protein_pointwise(self.protein_depthwise(embedded))
        ).transpose(1, 2)
        pooled = states.new_zeros((len(proteins), RELATIVE_BINS, 64))
        pooled.scatter_add_(1, bins[:, :, None].expand(-1, -1, 64),
                            states * valid[:, :, None])
        counts = states.new_zeros((len(proteins), RELATIVE_BINS, 1))
        counts.scatter_add_(1, bins[:, :, None], valid[:, :, None].to(states.dtype))
        bin_ids = torch.arange(RELATIVE_BINS, device=proteins.device)
        marker = (bin_ids == 4).long()
        return pooled / counts.clamp_min(1) + self.global_position(bin_ids) + self.site_marker(marker)
