from dataclasses import dataclass
from typing import Optional

import dgl
import torch
from torch import nn

from gnn_lib import models
from gnn_lib.models import GNNs
from gnn_lib.modules import encoders, utils, gnn


@dataclass
class TransformerEncoderGNNConfig(models.GNNConfig):
    type = GNNs.TRANSFORMER_ENCODER_GNN
    feedforward_dim: Optional[int] = None
    num_layers: int = 3
    num_heads: Optional[int] = None
    dropout: float = 0.1


class TransformerEncoderGNN(models.GNN):
    """
    Model to verify that a Transformer encoder is the
    same as a GNN with attention
    based aggregation applied on a homogeneous
    fully connected graph
    """

    def __init__(self,
                 node_hidden_dim: int,
                 hidden_feature: str,
                 cfg: TransformerEncoderGNNConfig,
                 sample_g: dgl.DGLHeteroGraph,
                 edge_hidden_dim: Optional[int] = None) -> None:
        super().__init__(cfg, node_hidden_dim, hidden_feature)
        self.cfg: TransformerEncoderGNNConfig
        assert sample_g.is_homogeneous, "transformer encoder gnn can only be used with homogeneous graphs"

        node_attrs = sample_g.node_attr_schemes()
        if "features" in sample_g.node_attr_schemes():
            num_additional_node_features = torch.prod(
                torch.tensor(node_attrs["features"].shape)
            ).item()
            self.feature_encoders = nn.ModuleList()
        else:
            num_additional_node_features = 0
            self.feature_encoders = None

        num_heads = max(1, self.node_hidden_dim // 64) if self.cfg.num_heads is None else self.cfg.num_heads
        feed_forward_dim = self.node_hidden_dim * 2 if self.cfg.feedforward_dim is None else self.cfg.feedforward_dim

        self.layers = nn.ModuleList()
        for i in range(self.cfg.num_layers):
            self.layers.append(
                gnn.TransformerEncoderConv(
                    num_features=self.node_hidden_dim,
                    num_heads=num_heads,
                    feedforward_dim=feed_forward_dim,
                    dropout=self.cfg.dropout
                )
            )
            if self.feature_encoders is not None:
                self.feature_encoders.append(
                    encoders.MLP(
                        in_dim=self.node_hidden_dim + num_additional_node_features,
                        hidden_dim=self.node_hidden_dim,
                        dropout=self.cfg.dropout,
                        num_layers=2
                    )
                )
        self.norm = nn.LayerNorm(self.node_hidden_dim)

    def forward(self, g: dgl.DGLHeteroGraph) -> dgl.DGLHeteroGraph:
        assert g.is_homogeneous and torch.all(torch.pow(g.batch_num_nodes(), 2) == g.batch_num_edges()), \
            f"{self.name} only works with fully connected homogeneous graphs"
        h = g.ndata[self.hidden_feature]

        for i in range(self.cfg.num_layers):
            # add additional node features at each layer
            if self.feature_encoders is not None:
                h = self.feature_encoders[i](
                    torch.cat([h, utils.flatten_and_pad(g.ndata["features"])], dim=1)
                )
            h = self.layers[i](g, h)

        h = self.norm(h)

        g.ndata[self.hidden_feature] = h
        return g


@dataclass
class TransformerEncoderConfig(models.GNNConfig):
    type = GNNs.TRANSFORMER_ENCODER
    feedforward_dim: Optional[int] = None
    num_layers: int = 3
    num_heads: Optional[int] = None
    dropout: float = 0.1


class TransformerEncoder(models.GNN):
    # transformer implemented with the pytorch transformer module for comparison with and
    # verification of gnn implementation above
    def __init__(self,
                 node_hidden_dim: int,
                 hidden_feature: str,
                 cfg: TransformerEncoderGNNConfig,
                 sample_g: dgl.DGLHeteroGraph,
                 edge_hidden_dim: Optional[int] = None) -> None:
        super().__init__(cfg, node_hidden_dim, hidden_feature)
        self.cfg: TransformerEncoderConfig

        self.encoder = encoders.Transformer(
            in_dim=self.node_hidden_dim,
            hidden_dim=self.node_hidden_dim,
            dropout=self.cfg.dropout,
            num_layers=self.cfg.num_layers,
            feed_forward_dim=self.cfg.feedforward_dim,
            num_heads=self.cfg.num_heads
        )

    def forward(self, g: dgl.DGLHeteroGraph) -> dgl.DGLHeteroGraph:
        assert g.is_homogeneous and torch.all(torch.pow(g.batch_num_nodes(), 2) == g.batch_num_edges()), \
            f"{self.name} only works with fully connected homogeneous graphs"
        # inputs: N * [L, H]
        x, lengths = utils.split_and_pad(g.ndata[self.hidden_feature], g.batch_num_nodes())
        # x: [N, Lmax, H]
        enc = self.encoder(x, padding_mask=utils.padding_mask(x, lengths))
        g.ndata[self.hidden_feature] = utils.join(enc, lengths)
        return g
