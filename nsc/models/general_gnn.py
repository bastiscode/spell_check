from copy import deepcopy
from dataclasses import dataclass
from typing import Optional, Dict

import dgl
import torch
from torch import nn

from nsc import models
from nsc.models import GNNs
from nsc.modules import utils, gnn, node_update


@dataclass
class GeneralGNNConfig(models.GNNConfig):
    type: GNNs = GNNs.GENERAL_GNN
    num_layers: int = 3
    dropout: float = 0.2
    message_scheme: str = "convolution"
    message_gating: bool = False
    node_update: str = "residual"
    share_parameters: bool = False
    norm: str = "layer"


class GeneralGNN(models.GNN):
    def __init__(self,
                 node_hidden_dim: int,
                 hidden_feature: str,
                 cfg: GeneralGNNConfig,
                 sample_g: dgl.DGLHeteroGraph,
                 edge_hidden_dim: Optional[int] = None) -> None:
        super().__init__(cfg, node_hidden_dim, hidden_feature, edge_hidden_dim)
        self.cfg: GeneralGNNConfig

        self.additional_node_features: Dict[str, int] = {}
        for node_type in sample_g.ntypes:
            node_attrs = sample_g.node_attr_schemes(node_type)

            if "features" not in node_attrs:
                continue

            self.additional_node_features[node_type] = torch.prod(torch.tensor(node_attrs["features"].shape)).item()

        self.additional_edge_features: Dict[str, int] = {}
        for edge_type in sample_g.canonical_etypes:
            edge_attrs = sample_g.edge_attr_schemes(edge_type)

            if "features" not in edge_attrs:
                continue

            self.additional_edge_features[edge_type[1]] = max(
                self.additional_edge_features.get(edge_type[1], 0),
                torch.prod(torch.tensor(edge_attrs["features"].shape)).item()
            )

        self.gnn_layers = nn.ModuleList()
        self.node_update_layers = nn.ModuleList()

        if self.cfg.message_scheme == "convolution":
            layer = gnn.ConvolutionLayer(
                sample_g=sample_g,
                node_input_dims={node_type: self.node_hidden_dim + self.additional_node_features.get(node_type, 0)
                                 for node_type in sample_g.ntypes},
                node_hidden_dim=self.node_hidden_dim,
                dropout=self.cfg.dropout,
                message_gating=self.cfg.message_gating,
                edge_feat_dims=self.additional_edge_features
            )
        elif self.cfg.message_scheme == "attention":
            layer = gnn.AttentionLayer(
                sample_g=sample_g,
                node_input_dims={node_type: self.node_hidden_dim + self.additional_node_features.get(node_type, 0)
                                 for node_type in sample_g.ntypes},
                node_hidden_dim=self.node_hidden_dim,
                dropout=self.cfg.dropout,
                message_gating=self.cfg.message_gating,
                edge_feat_dims=self.additional_edge_features
            )
        elif self.cfg.message_scheme == "message_passing":
            layer = gnn.MessagePassingLayer(
                sample_g=sample_g,
                node_input_dims={node_type: self.node_hidden_dim + self.additional_node_features.get(node_type, 0)
                                 for node_type in sample_g.ntypes},
                node_hidden_dim=self.node_hidden_dim,
                edge_hidden_dim=self.edge_hidden_dim,
                dropout=self.cfg.dropout,
                message_gating=self.cfg.message_gating,
                edge_feat_dims=self.additional_edge_features
            )
        else:
            raise ValueError(f"Unknown messaging scheme {self.cfg.message_scheme}")

        if self.cfg.node_update == "recurrent":
            assert self.cfg.share_parameters, "recurrent update can only be used together with share_parameters=true"
            node_update_layer = node_update.RNNUpdate(
                self.node_hidden_dim, self.node_hidden_dim, self.cfg.dropout, rnn_type="gru"
            )
        elif self.cfg.node_update == "transformer":
            node_update_layer = node_update.TransformerUpdate(
                self.node_hidden_dim, self.node_hidden_dim, self.cfg.dropout
            )
        elif self.cfg.node_update == "residual":
            node_update_layer = node_update.ResidualUpdate(self.node_hidden_dim, self.node_hidden_dim, self.cfg.dropout)
        else:
            raise ValueError(f"Unknown node update type {self.cfg.node_update}")

        dst_types = set(e_type[2] for e_type in sample_g.canonical_etypes)
        for i in range(1 if self.cfg.share_parameters else self.cfg.num_layers):
            self.gnn_layers.append(deepcopy(layer))
            self.node_update_layers.append(
                nn.ModuleDict({
                    node_type: deepcopy(node_update_layer)
                    for node_type in dst_types
                })
            )

    def forward(self, g: dgl.DGLHeteroGraph) -> dgl.DGLHeteroGraph:
        self.cfg: GeneralGNNConfig
        h = {g.ntypes[0]: g.ndata[self.hidden_feature]} if g.is_homogeneous else g.ndata[self.hidden_feature]

        node_feat = {
            n_type: utils.flatten_and_pad(g.nodes[n_type].data["features"])
            if n_type in self.additional_node_features
            else torch.zeros((g.num_nodes(n_type), 0), dtype=torch.float, device=g.device)
            for n_type in g.ntypes
        }
        edge_feat = {
            e_type: utils.flatten_and_pad(g.edges[e_type].data["features"], self.additional_edge_features[e_type[1]])
            if e_type[1] in self.additional_edge_features
            else torch.zeros((g.num_edges(e_type), 0), dtype=torch.float, device=g.device)
            for e_type in g.canonical_etypes
        }

        for i in range(self.cfg.num_layers):
            i = 0 if self.cfg.share_parameters else i

            h_in = {k: torch.cat([v, node_feat[k]], dim=1) for k, v in h.items()}

            u = self.gnn_layers[i](g, h_in, edge_feat)

            h = {
                k: self.node_update_layers[i][k](v, u[k])
                if k in self.node_update_layers[i] and k in u else v
                for k, v in h.items()
            }

        g.ndata[self.hidden_feature] = h[g.ntypes[0]] if g.is_homogeneous else h
        return g
