import enum
from typing import Any, Optional

import einops
import torch
from torch import nn

from nsc.modules import utils


class Encoders(enum.IntEnum):
    MLP = 1
    BI_GRU = 2
    CNN = 3
    TRANSFORMER = 4


class Encoder(nn.Module):
    def __init__(self,
                 in_dim: int,
                 hidden_dim: int,
                 dropout: float,
                 num_layers: int = 2) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.num_layers = num_layers

    def forward(self, feat: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        raise NotImplementedError


class MLP(Encoder):
    def __init__(self,
                 in_dim: int,
                 hidden_dim: int,
                 dropout: float,
                 num_layers: int = 2,
                 feed_forward_dim: Optional[int] = None) -> None:
        super().__init__(in_dim, hidden_dim, dropout, num_layers)
        self.layers = nn.ModuleList()
        if feed_forward_dim is None:
            feed_forward_dim = hidden_dim
        in_features = in_dim
        for i in range(num_layers):
            self.layers.append(nn.Linear(in_features, hidden_dim if i == num_layers - 1 else feed_forward_dim))
            if i < num_layers - 1:
                self.layers.append(nn.GELU())
                self.layers.append(nn.Dropout(dropout))
                in_features = feed_forward_dim
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, inputs: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        x = inputs
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)


class CNN(nn.Module):
    def __init__(self,
                 in_dim: int,
                 hidden_dim: int,
                 dropout: float,
                 num_layers: int = 2) -> None:
        super().__init__()
        self.layers = nn.ModuleList()
        in_channels = in_dim
        for i in range(num_layers):
            self.layers.append(nn.Conv1d(in_channels=in_channels,
                                         out_channels=hidden_dim,
                                         kernel_size=(3, 3),
                                         groups=4,
                                         padding=1))
            if i < num_layers - 1:
                self.layers.append(nn.GELU())
                self.layers.append(nn.Dropout(dropout))
                in_channels = hidden_dim
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, inputs: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        # x: [N, Lmax, H]
        x = einops.rearrange(inputs, "n l h -> n h l")
        for layer in self.layers:
            x = layer(x)
        x = einops.rearrange(x, "n h l -> n l h")
        return self.norm(x)


class BiGRU(nn.Module):
    def __init__(self,
                 in_dim: int,
                 hidden_dim: int,
                 dropout: float,
                 num_layers: int = 1) -> None:
        super().__init__()
        self.gru = nn.GRU(input_size=in_dim,
                          hidden_size=hidden_dim,
                          batch_first=True,
                          bidirectional=True,
                          dropout=dropout,
                          num_layers=num_layers)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, inputs: torch.Tensor, lengths: Optional[torch.Tensor] = None, **kwargs: Any) -> torch.Tensor:
        # inputs: N * [L, H]
        x = utils.pack(inputs, lengths)
        # x: [N, Lmax, H]
        x = self.gru(x)[0]
        x, _ = utils.unpack(x)
        # sum the two directions
        x = einops.reduce(x, "n l (d h) -> n l h", d=2, reduction="sum")
        return self.norm(x)


class Transformer(nn.Module):
    def __init__(self,
                 in_dim: int,
                 hidden_dim: int,
                 dropout: float,
                 num_layers: int = 1,
                 feed_forward_dim: Optional[int] = None,
                 num_heads: Optional[int] = None) -> None:
        super().__init__()
        if in_dim != hidden_dim:
            self.proj = nn.Linear(in_dim, hidden_dim)
        else:
            self.proj = nn.Identity()
        feed_forward_dim = feed_forward_dim if feed_forward_dim is not None else hidden_dim * 2
        num_heads = num_heads if num_heads is not None else max(1, hidden_dim // 64)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            dim_feedforward=feed_forward_dim,
            nhead=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, inputs: torch.Tensor, padding_mask: Optional[torch.Tensor] = None, **kwargs: Any) -> torch.Tensor:
        x = self.proj(inputs)
        x = self.encoder(x, src_key_padding_mask=padding_mask)
        return self.norm(x)


def get_feature_encoder(
        encoder: Encoders,
        **kwargs: Any
) -> nn.Module:
    if encoder == Encoders.MLP:
        encoder = MLP(**kwargs)
    elif encoder == Encoders.CNN:
        encoder = CNN(**kwargs)
    elif encoder == Encoders.BI_GRU:
        encoder = BiGRU(**kwargs)
    elif encoder == Encoders.TRANSFORMER:
        encoder = Transformer(**kwargs)
    else:
        raise ValueError(f"Unknown feature encoder {encoder}")
    return encoder
