from dataclasses import dataclass
from typing import Optional

import dgl
import torch
from dgl.nn import pytorch as gnn
from torch import nn
from torch.nn import functional as F

from gnn_lib import models
from gnn_lib.models import GNNs
from gnn_lib.modules import normalization, utils, encoders


@dataclass
class SimpleGNNConfig(models.GNNConfig):
    type: GNNs = GNNs.SIMPLE_GNN
    num_layers: int = 3
    dropout: float = 0.2
    norm: str = "layer"
    recurrent_node_update: bool = False


class SimpleGNN(models.GNN):
    def __init__(self,
                 node_hidden_dim: int,
                 hidden_feature: str,
                 cfg: SimpleGNNConfig,
                 sample_g: dgl.DGLHeteroGraph,
                 edge_hidden_dim: Optional[int] = None) -> None:
        super().__init__(cfg, node_hidden_dim, hidden_feature)
        self.cfg: SimpleGNNConfig

        self.additional_node_feature_encoders = nn.ModuleDict()
        for node_type in sample_g.ntypes:
            node_attrs = sample_g.node_attr_schemes(node_type)

            if "features" not in node_attrs:
                continue

            num_features = torch.prod(
                torch.tensor(node_attrs["features"].shape)
            ).item()
            self.additional_node_feature_encoders[node_type] = encoders.MLP(
                in_dim=self.node_hidden_dim + num_features,
                hidden_dim=self.node_hidden_dim,
                dropout=self.cfg.dropout,
                num_layers=2
            )

        self.drops = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.layers = nn.ModuleList()
        for i in range(1 if self.cfg.recurrent_node_update else self.cfg.num_layers):
            self.drops.append(
                nn.Dropout(self.cfg.dropout)
            )
            self.norms.append(normalization.get_norm(
                norm=self.cfg.norm,
                node_types=sample_g.ntypes,
                edge_types=[],
                node_hidden_dim=self.node_hidden_dim,
                edge_hidden_dim=self.edge_hidden_dim
            ))
            self.layers.append(
                gnn.HeteroGraphConv({
                    edge_type: gnn.GraphConv(
                        in_feats=self.node_hidden_dim,
                        out_feats=self.node_hidden_dim,
                    )
                    for edge_type in sample_g.etypes
                })
            )

        if self.cfg.recurrent_node_update:
            self.grus = nn.ModuleDict({
                node_type: nn.GRUCell(self.node_hidden_dim, self.node_hidden_dim)
                for node_type in sample_g.ntypes
            })

    def forward(self, g: dgl.DGLHeteroGraph) -> dgl.DGLHeteroGraph:
        h = {g.ntypes[0]: g.ndata[self.hidden_feature]} if g.is_homogeneous else g.ndata[self.hidden_feature]

        for i in range(self.cfg.num_layers):
            # each layer has the structure:
            # 1. graph convolution
            # 2. one of:
            #   2.1 recurrent update (gru)
            #   2.2 skip + activation
            # 3. normalization
            i = 0 if self.cfg.recurrent_node_update else i

            # add additional node features at each step
            h_in = {
                node_type: self.additional_node_feature_encoders[node_type](
                    torch.cat([feat, utils.flatten_and_pad(g.nodes[node_type].data["features"])], dim=1)
                )
                if node_type in self.additional_node_feature_encoders else feat
                for node_type, feat in h.items()
            }

            u = self.layers[i](g, h_in)

            # for recurrent updates use gru
            if self.cfg.recurrent_node_update:
                h = {k: self.drops[i](self.grus[k](u[k], h[k]))
                     for k, v in h.items()}
            # else just use a residual connection
            else:
                h = {k: v + u[k]
                     for k, v in h.items()}
                h = {k: self.drops[i](F.gelu(v))
                     for k, v in h.items()}

            h, _ = self.norms[i](g, (h, {}))

        h = h[g.ntypes[0]] if g.is_homogeneous else h
        g.ndata[self.hidden_feature] = h

        return g
