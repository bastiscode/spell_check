from typing import Dict, Any, List, Tuple

import dgl
import torch
from torch import nn

from nsc.modules import utils

NODE_AND_EDGE_FEAT = Tuple[Dict[str, torch.Tensor], Dict[Tuple[str, str, str], torch.Tensor]]


class Normalization(nn.Module):
    def forward(self,
                g: dgl.DGLHeteroGraph,
                feat: NODE_AND_EDGE_FEAT
                ) -> NODE_AND_EDGE_FEAT:
        raise NotImplementedError


class LayerNormalization(Normalization):
    def __init__(self,
                 node_types: List[str],
                 edge_types: List[str],
                 node_hidden_dim: int,
                 edge_hidden_dim: int,
                 affine: bool = True) -> None:
        super().__init__()
        self.node_norms = nn.ModuleDict({
            node_type: nn.LayerNorm(node_hidden_dim, elementwise_affine=affine)
            for node_type in node_types
        })
        self.edge_norms = nn.ModuleDict({
            edge_type: nn.LayerNorm(edge_hidden_dim, elementwise_affine=affine)
            for edge_type in edge_types
        })

    def forward(self,
                _: dgl.DGLHeteroGraph,
                feat: NODE_AND_EDGE_FEAT
                ) -> NODE_AND_EDGE_FEAT:
        node_feat = {k: self.node_norms[k](v)
                     for k, v in feat[0].items()}
        edge_feat = {k: self.edge_norms[k[1]](v)
                     for k, v in feat[1].items()}
        return node_feat, edge_feat


class BatchNormalization(Normalization):
    def __init__(self,
                 node_types: List[str],
                 edge_types: List[str],
                 node_hidden_dim: int,
                 edge_hidden_dim: int,
                 affine: bool = True) -> None:
        super().__init__()
        self.node_norms = nn.ModuleDict({
            node_type: nn.BatchNorm1d(node_hidden_dim, affine=affine)
            for node_type in node_types
        })
        self.edge_norms = nn.ModuleDict({
            edge_type: nn.BatchNorm1d(edge_hidden_dim, affine=affine)
            for edge_type in edge_types
        })

    def forward(self,
                _: dgl.DGLHeteroGraph,
                feat: NODE_AND_EDGE_FEAT
                ) -> NODE_AND_EDGE_FEAT:
        node_feat = {k: self.node_norms[k](v)
                     for k, v in feat[0].items()}
        edge_feat = {k: self.edge_norms[k[1]](v)
                     for k, v in feat[1].items()}
        return node_feat, edge_feat


# modified original implementation from
# https://github.com/cyh1112/GraphNormalization/blob/d6f423130a99d3803ad8147e4df18fc747f9fa05/norm/graph_norm.py
class GraphNormalization(Normalization):
    def __init__(self,
                 node_types: List[str],
                 edge_types: List[str],
                 node_hidden_dim: int,
                 edge_hidden_dim: int,
                 affine: bool = True) -> None:
        super().__init__()
        self.eps = 1e-5
        self.momentum = 0.1
        self.affine = affine
        if self.affine:
            self.node_gamma = nn.ParameterDict({
                node_type: nn.Parameter(torch.ones(node_hidden_dim))
                for node_type in node_types
            })
            self.node_beta = nn.ParameterDict({
                node_type: nn.Parameter(torch.zeros(node_hidden_dim))
                for node_type in node_types
            })
            self.edge_gamma = nn.ParameterDict({
                edge_type: nn.Parameter(torch.ones(edge_hidden_dim))
                for edge_type in edge_types
            })
            self.edge_beta = nn.ParameterDict({
                edge_type: nn.Parameter(torch.zeros(edge_hidden_dim))
                for edge_type in edge_types
            })

    def normalize(self, features: torch.Tensor) -> torch.Tensor:
        if features.shape[0] > 1:
            std, mean = torch.std_mean(features, dim=0, unbiased=False, keepdim=True)
            return (features - mean) / (std + self.eps)
        else:
            return features

    def forward(self,
                g: dgl.DGLHeteroGraph,
                feat: NODE_AND_EDGE_FEAT
                ) -> NODE_AND_EDGE_FEAT:
        node_output = {}
        for node_type, features in feat[0].items():
            batch_features = torch.split(
                features,
                utils.tensor_to_python(g.batch_num_nodes(node_type), force_list=True)
            )
            normalized_features = []
            for bf in batch_features:
                normalized_features.append(self.normalize(bf))
            node_output[node_type] = torch.cat(normalized_features, dim=0)
            if self.affine:
                node_output[node_type] = node_output[node_type] * self.node_gamma[node_type] + self.node_beta[node_type]
        edge_output = {}
        for edge_type, features in feat[1].items():
            batch_features = torch.split(
                features,
                utils.tensor_to_python(g.batch_num_edges(edge_type), force_list=True)
            )
            normalized_features = []
            for bf in batch_features:
                normalized_features.append(self.normalize(bf))
            edge_output[edge_type] = torch.cat(normalized_features, dim=0)
            if self.affine:
                edge_output[edge_type] = (
                        edge_output[edge_type] * self.edge_gamma[edge_type[1]] + self.edge_beta[edge_type[1]]
                )
        return node_output, edge_output


def get_norm(norm: str, **kwargs: Any) -> Normalization:
    if norm == "layer":
        return LayerNormalization(**kwargs)
    elif norm == "batch":
        return BatchNormalization(**kwargs)
    elif norm == "graph":
        return GraphNormalization(**kwargs)
    else:
        raise ValueError(f"Unknown normalization type {norm}")
