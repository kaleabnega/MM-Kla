"""Architecture-aligned PyTorch implementation of the released DeepKla model."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


# The author script maps residues to 1..26, prefixes a start token (1), adds
# three to residue IDs, then maps IDs >= 23 to the OOV token (2).
RESIDUE_IDS = {
    "A": 1, "C": 2, "D": 3, "E": 4, "F": 5, "G": 6, "H": 7,
    "I": 8, "K": 9, "L": 10, "M": 11, "N": 12, "P": 13, "Q": 14,
    "R": 15, "S": 16, "T": 17, "V": 18, "W": 19, "Y": 20,
}
INPUT_LENGTH = 200


def encode_windows(windows: list[str]) -> torch.Tensor:
    encoded = torch.zeros((len(windows), INPUT_LENGTH), dtype=torch.long)
    for row, window in enumerate(windows):
        if len(window) != 51 or window[25] != "K" or set(window) - RESIDUE_IDS.keys():
            raise ValueError("DeepKla requires canonical 51-residue, K-centered windows")
        tokens = [1] + [2 if RESIDUE_IDS[residue] + 3 >= 23
                        else RESIDUE_IDS[residue] + 3 for residue in window]
        encoded[row, -len(tokens):] = torch.tensor(tokens)
    return encoded


class ContextAttention(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.projection = nn.Linear(channels, channels)
        self.context = nn.Parameter(torch.empty(channels))
        nn.init.xavier_uniform_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)
        nn.init.xavier_uniform_(self.context.unsqueeze(0))

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        scores = torch.tanh(self.projection(tokens)) @ self.context
        weights = torch.softmax(scores, dim=1)
        return torch.sum(tokens * weights.unsqueeze(-1), dim=1)


class DeepKlaAligned(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(23, 128)
        self.dropout = nn.Dropout(0.25)
        self.conv1 = nn.Conv1d(128, 64, kernel_size=10)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=5)
        self.gru = nn.GRU(64, 50, batch_first=True, bidirectional=True)
        self.attention = ContextAttention(100)
        self.output = nn.Linear(100, 1)
        nn.init.uniform_(self.embedding.weight, -0.05, 0.05)
        for convolution in (self.conv1, self.conv2):
            nn.init.xavier_uniform_(convolution.weight)
            nn.init.zeros_(convolution.bias)
        for name, parameter in self.gru.named_parameters():
            if name.startswith("weight_ih"):
                nn.init.xavier_uniform_(parameter)
            elif name.startswith("weight_hh"):
                nn.init.orthogonal_(parameter)
            else:
                nn.init.zeros_(parameter)
        nn.init.xavier_uniform_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        x = self.dropout(self.embedding(token_ids)).transpose(1, 2)
        x = F.max_pool1d(F.relu(self.conv1(x)), kernel_size=2)
        x = F.max_pool1d(F.relu(self.conv2(x)), kernel_size=2)
        x, _ = self.gru(x.transpose(1, 2))
        return torch.sigmoid(self.output(self.attention(x)).squeeze(-1))
