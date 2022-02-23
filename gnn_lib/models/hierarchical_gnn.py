from dataclasses import dataclass
from typing import Optional, List

import dgl
import torch
from dgl import function as dfn
from dgl.nn import pytorch as dgl_gnn
from omegaconf import MISSING
from torch import nn
from torch.nn import functional as F

from gnn_lib import models
from gnn_lib.models import GNNs
from gnn_lib.modules import normalization, encoders, utils, gnn


@dataclass
class HierarchicalGNNConfig(models.GNNConfig):
    type: GNNs = GNNs.HIERARCHICAL_GNN
    num_layers: int = 3
    num_heads: Optional[int] = None
    dropout: float = 0.2
    attention_dropout: float = 0.1
    use_v2: bool = True
    norm: str = "layer"
    recurrent_node_update: bool = False
    node_hierarchy: List[str] = MISSING


class HierarchicalGNN(models.GNN):
    def __init__(self,
                 node_hidden_dim: int,
                 hidden_feature: str,
                 cfg: HierarchicalGNNConfig,
                 sample_g: dgl.DGLHeteroGraph,
                 edge_hidden_dim: Optional[int] = None) -> None:
        super().__init__(cfg, node_hidden_dim, hidden_feature)
        self.cfg: HierarchicalGNNConfig

        assert set(self.cfg.node_hierarchy) == set(sample_g.ntypes)

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

        gat_cls = gnn.GATv2Conv if self.cfg.use_v2 else dgl_gnn.GATConv

        num_heads = self.cfg.num_heads if self.cfg.num_heads is not None else max(1, self.node_hidden_dim // 64)

        self.drops = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.layers = nn.ModuleList()
        for i in range(self.cfg.num_layers):
            self.drops.append(nn.Dropout(self.cfg.dropout))
            self.norms.append(normalization.get_norm(
                norm=self.cfg.norm,
                node_types=sample_g.ntypes,
                edge_types=[],
                hidden_dim=self.node_hidden_dim
            ))
            self.layers.append(
                dgl_gnn.HeteroGraphConv({
                    edge_type: gat_cls(
                        in_feats=self.node_hidden_dim,
                        out_feats=self.node_hidden_dim,
                        num_heads=num_heads,
                        feat_drop=self.cfg.dropout,
                        attn_drop=self.cfg.attention_dropout
                    )
                    for edge_type in sample_g.etypes
                })
            )

        self.node_update_encoders = nn.ModuleDict()
        for node_type in sample_g.ntypes:
            self.node_update_encoders[node_type] = encoders.MLP(
                in_dim=self.node_hidden_dim * num_heads,
                hidden_dim=self.node_hidden_dim,
                dropout=self.cfg.dropout,
                num_layers=2
            )

        if self.cfg.recurrent_node_update:
            self.grus = nn.ModuleDict({
                node_type: nn.GRUCell(self.node_hidden_dim, self.node_hidden_dim)
                for node_type in sample_g.ntypes
            })

    def forward(self, g: dgl.DGLHeteroGraph) -> dgl.DGLHeteroGraph:
        self.cfg: HierarchicalGNNConfig
        h = {g.ntypes[0]: g.ndata[self.hidden_feature]} if g.is_homogeneous else g.ndata[self.hidden_feature]

        for i in range(len(self.cfg.node_hierarchy)):
            filtered_edge_types = [
                edge_type for edge_type in g.canonical_etypes
                if edge_type[2] == self.cfg.node_hierarchy[i]
            ]
            filtered_g = dgl.edge_type_subgraph(g, filtered_edge_types)
            filtered_g.set_batch_num_nodes({node_type: g.batch_num_nodes(node_type)
                                            for node_type in filtered_g.ntypes})
            filtered_g.set_batch_num_edges({edge_type: g.batch_num_edges(edge_type)
                                            for edge_type in filtered_g.canonical_etypes})
            filtered_h = {node_type: h[node_type] for node_type in filtered_g.ntypes}

            for j in range(self.cfg.num_layers):
                j = 0 if self.cfg.recurrent_node_update else j

                filtered_h_in = {
                    node_type: self.additional_node_feature_encoders[node_type](
                        torch.cat([feat, utils.flatten_and_pad(filtered_g.nodes[node_type].data["features"])], dim=1)
                    )
                    if node_type in self.additional_node_feature_encoders else feat
                    for node_type, feat in filtered_h.items()
                }

                u = self.layers[j](filtered_g, filtered_h_in)
                u = {k: self.node_update_encoders[k](torch.flatten(v, 1))
                     for k, v in u.items()}

                # for recurrent updates use gru
                if self.cfg.recurrent_node_update:
                    filtered_h = {k: self.drops[i](self.grus[k](u[k], filtered_h[k]))
                                  for k, v in filtered_h.items()}
                # else just use a residual connection
                else:
                    filtered_h = {k: v + u[k]
                         for k, v in filtered_h.items()}
                    filtered_h = {k: self.drops[i](F.gelu(v))
                         for k, v in filtered_h.items()}

                filtered_h, _ = self.norms[i](filtered_g, (filtered_h, {}))

            h[self.cfg.node_hierarchy[i]] = filtered_h[self.cfg.node_hierarchy[i]]

            # initialize next nodes with the mean of their incoming connections plus potential embeddings
            if i < len(self.cfg.node_hierarchy) - 1:
                # find corresponding edge type between current and next node in hierarchy
                e_type = None
                for from_, e_type_, to_ in g.canonical_etypes:
                    if from_ == self.cfg.node_hierarchy[i] and to_ == self.cfg.node_hierarchy[i + 1]:
                        e_type = (from_, e_type_, to_)
                        break
                assert e_type is not None

                with g.local_scope():
                    g.nodes[self.cfg.node_hierarchy[i]].data[self.hidden_feature] = \
                        filtered_h[self.cfg.node_hierarchy[i]]

                    g.update_all(
                        message_func=dfn.copy_u(self.hidden_feature, "x"),
                        reduce_func=dfn.mean("x", self.hidden_feature),
                        etype=e_type
                    )

                    h[self.cfg.node_hierarchy[i + 1]] += \
                        g.nodes[self.cfg.node_hierarchy[i + 1]].data[self.hidden_feature]

        h = h[g.ntypes[0]] if g.is_homogeneous else h
        g.ndata[self.hidden_feature] = h

        return g
