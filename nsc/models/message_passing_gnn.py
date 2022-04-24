from dataclasses import dataclass
from typing import Dict, Callable, Optional, Tuple

import dgl
import torch
from dgl import function as dfn
from dgl.udf import EdgeBatch, NodeBatch
from torch import nn

from nsc import models
from nsc.models import GNNs
from nsc.modules import encoders, normalization, utils


@dataclass
class MessagePassingGNNConfig(models.GNNConfig):
    type: GNNs = GNNs.MESSAGE_PASSING_GNN
    num_layers: int = 3
    edge_num_layers: int = 2
    edge_feed_forward_dim: Optional[int] = None
    node_num_layers: int = 2
    node_feed_forward_dim: Optional[int] = None
    dropout: float = 0.2
    message_gating: bool = False
    message_aggregation: str = "sum"
    recurrent_update: bool = False
    norm: str = "layer"


class MessagePassingGNN(models.GNN):
    def __init__(self,
                 node_hidden_dim: int,
                 hidden_feature: str,
                 cfg: MessagePassingGNNConfig,
                 sample_g: dgl.DGLHeteroGraph,
                 edge_hidden_dim: Optional[int] = None) -> None:
        assert edge_hidden_dim is not None, "general gnn requires edge_hidden_dim to be specified"
        super().__init__(cfg, node_hidden_dim, hidden_feature, edge_hidden_dim)
        self.cfg: MessagePassingGNNConfig

        self.max_num_node_features = self.max_num_edge_features = 0
        for node_type in sample_g.ntypes:
            node_attrs = sample_g.node_attr_schemes(node_type)

            if "features" not in node_attrs:
                continue

            self.max_num_node_features = max(
                self.max_num_node_features,
                torch.prod(torch.tensor(node_attrs["features"].shape)).item()
            )

        for e_type in sample_g.canonical_etypes:
            edge_attrs = sample_g.edge_attr_schemes(e_type)

            if "features" not in edge_attrs:
                continue

            self.max_num_edge_features = max(
                self.max_num_edge_features,
                torch.prod(torch.tensor(edge_attrs["features"].shape)).item()
            )

        self.norms = nn.ModuleList()
        self.edge_layers = nn.ModuleList()
        self.edge_gates = nn.ModuleList()
        self.node_layers = nn.ModuleList()
        for i in range(1 if self.cfg.recurrent_update else self.cfg.num_layers):
            self.norms.append(normalization.get_norm(
                norm=self.cfg.norm,
                node_types=sample_g.ntypes,
                edge_types=sample_g.etypes,
                node_hidden_dim=self.node_hidden_dim,
                edge_hidden_dim=self.edge_hidden_dim
            ))
            self.edge_layers.append(
                encoders.MLP(
                    2 * self.node_hidden_dim + self.edge_hidden_dim + self.max_num_edge_features,
                    self.edge_hidden_dim,
                    self.cfg.dropout,
                    self.cfg.edge_num_layers,
                    self.cfg.edge_feed_forward_dim
                )
            )
            if self.cfg.message_gating:
                self.edge_gates.append(
                    nn.Sequential(
                        nn.Linear(self.edge_hidden_dim, self.edge_hidden_dim),
                        nn.GELU(),
                        nn.Dropout(self.cfg.dropout),
                        nn.Linear(self.edge_hidden_dim, 1),
                        nn.Sigmoid()
                    )
                )
            self.node_layers.append(
                encoders.MLP(
                    self.node_hidden_dim + self.edge_hidden_dim + self.max_num_node_features,
                    self.node_hidden_dim,
                    self.cfg.dropout,
                    self.cfg.node_num_layers,
                    self.cfg.node_feed_forward_dim
                )
            )

        if self.cfg.recurrent_update:
            self.edge_gru = nn.GRUCell(self.edge_hidden_dim, self.edge_hidden_dim)
            self.node_gru = nn.GRUCell(self.node_hidden_dim, self.node_hidden_dim)

        if self.cfg.message_aggregation == "mean":
            self.reduce_fn = dfn.mean("messages", "message")
        elif self.cfg.message_aggregation == "sum":
            self.reduce_fn = dfn.sum("messages", "message")
        else:
            raise ValueError(f"Unknown message aggregation method '{self.cfg.message_aggregation}")

    def compute_messages(self, idx: int) -> Callable:
        ff = self.edge_layers[idx]

        def udf(edges: EdgeBatch) -> Dict[str, torch.Tensor]:
            h = edges.data[self.hidden_feature]
            if self.max_num_edge_features > 0:
                if "features" in edges.data:
                    feats = utils.flatten_and_pad(edges.data["features"], self.max_num_edge_features).float()
                else:
                    feats = torch.zeros((h.shape[0], self.max_num_edge_features), dtype=torch.float, device=h.device)
            else:
                feats = torch.zeros((h.shape[0], 0), dtype=torch.float, device=h.device)

            update = ff(
                torch.cat([
                    h,
                    feats,
                    edges.src[self.hidden_feature],
                    edges.dst[self.hidden_feature]
                ], dim=1)
            )

            if self.cfg.message_gating:
                update = update * self.edge_gates[idx](update)

            if self.cfg.recurrent_update:
                return {"messages": update,
                        self.hidden_feature: self.edge_gru(update, h)}
            else:
                return {"messages": update,
                        self.hidden_feature: update}

        return udf

    def update_nodes(self, idx: int) -> Callable:
        ff = self.node_layers[idx]

        def udf(nodes: NodeBatch) -> Dict[str, torch.Tensor]:
            h = nodes.data[self.hidden_feature]
            if self.max_num_node_features > 0:
                if "features" in nodes.data:
                    feats = utils.flatten_and_pad(nodes.data["features"], self.max_num_node_features).float()
                else:
                    feats = torch.zeros((h.shape[0], self.max_num_node_features), dtype=torch.float, device=h.device)
            else:
                feats = torch.zeros((h.shape[0], 0), dtype=torch.float, device=h.device)

            update = ff(
                torch.cat([
                    h,
                    feats,
                    nodes.data["message"]
                ], dim=1)
            )

            if self.cfg.recurrent_update:
                return {self.hidden_feature: self.node_gru(update, h)}
            else:
                return {self.hidden_feature: update}

        return udf

    def _get_node_and_edge_feat(self, g: dgl.DGLHeteroGraph) \
            -> Tuple[Dict[str, torch.Tensor], Dict[Tuple[str, str, str], torch.Tensor]]:
        h_n = {g.ntypes[0]: g.ndata[self.hidden_feature]} \
            if g.is_homogeneous else g.ndata[self.hidden_feature]
        h_e = {g.canonical_etypes[0]: g.edata[self.hidden_feature]} \
            if g.is_homogeneous else g.edata[self.hidden_feature]
        return h_n, h_e

    def forward(self, g: dgl.DGLHeteroGraph) -> dgl.DGLHeteroGraph:
        with g.local_scope():
            for i in range(self.cfg.num_layers):
                i = 0 if self.cfg.recurrent_update else i

                for e_type in g.canonical_etypes:
                    if g.num_edges(e_type) == 0:
                        continue

                    # compute messages and update edge hidden representations
                    g.apply_edges(
                        func=self.compute_messages(i),
                        etype=e_type
                    )

                g.multi_update_all(
                    {e_type: (
                        dfn.copy_e("messages", "messages"),
                        self.reduce_fn
                    ) for e_type in g.canonical_etypes if g.num_edges(e_type) > 0},
                    cross_reducer="sum",
                    apply_node_func=self.update_nodes(i)
                )

                # normalize node and edge features
                h_n, h_e = self._get_node_and_edge_feat(g)
                h_n, h_e = self.norms[i](g, (h_n, h_e))

                g.ndata[self.hidden_feature] = h_n[g.ntypes[0]] if g.is_homogeneous else h_n
                g.edata[self.hidden_feature] = h_e[g.canonical_etypes[0]] if g.is_homogeneous else h_e

            h_n, h_e = self._get_node_and_edge_feat(g)

        g.ndata[self.hidden_feature] = h_n[g.ntypes[0]] if g.is_homogeneous else h_n
        g.edata[self.hidden_feature] = h_e[g.canonical_etypes[0]] if g.is_homogeneous else h_e
        return g
