from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F


class ResidualUpdate(nn.Module):
    # simple residual node update
    def __init__(self,
                 in_dim: int,
                 out_dim: int,
                 dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm_out = nn.LayerNorm(out_dim)

    def forward(self, feat: torch.Tensor, update: torch.Tensor) -> torch.Tensor:
        return self.norm_out(feat + self.dropout(update))


class TransformerUpdate(nn.Module):
    # transformer-style update (residual connection around high-dimensional feed-forward)
    def __init__(self,
                 in_dim: int,
                 out_dim: int,
                 dropout: float,
                 feed_forward_dim: Optional[int] = None) -> None:
        super().__init__()
        if feed_forward_dim is None:
            feed_forward_dim = 2 * in_dim

        self.dropout_in = nn.Dropout(dropout)
        self.norm_in = nn.Linear(in_dim, in_dim)

        self.linear_ff_in = nn.Linear(in_dim, feed_forward_dim)
        self.dropout_ff = nn.Dropout(dropout)
        self.linear_ff_out = nn.Linear(feed_forward_dim, out_dim)

        self.dropout_out = nn.Dropout(dropout)
        self.norm_out = nn.LayerNorm(out_dim)

    def forward(self, feat: torch.Tensor, update: torch.Tensor) -> torch.Tensor:
        feat = self.norm_in(feat + self.dropout_in(update))
        feat = self.norm_out(feat + self.dropout_out(self._ff(feat)))
        return feat

    def _ff(self, feat: torch.Tensor) -> torch.Tensor:
        return self.linear_ff_out(self.dropout_ff(F.gelu(self.linear_ff_in(feat))))


class RNNUpdate(nn.Module):
    def __init__(self,
                 in_dim: int,
                 out_dim: int,
                 dropout: float,
                 rnn_type: str = "gru") -> None:
        super().__init__()
        self.dropout_update = nn.Dropout(dropout)

        if rnn_type == "gru":
            self.gru = nn.GRUCell(in_dim, out_dim)
        else:
            raise ValueError(f"Unknown rnn type {rnn_type}")

        self.norm_out = nn.LayerNorm(out_dim)

    def forward(self, feat: torch.Tensor, update: torch.Tensor) -> torch.Tensor:
        return self.norm_out(self.gru(self.dropout_update(update), feat))
