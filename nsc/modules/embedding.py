import math
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Callable, Set, Union, Any

import dgl
import einops
import omegaconf
import torch
from dgl import function as dfn
from dgl.udf import EdgeBatch, NodeBatch
from torch import nn

from nsc.utils import DataInput


class TokenEmbedding(nn.Module):
    def __init__(self,
                 embedding_dim: int,
                 num_embeddings: int,
                 padding_idx: Optional[int] = None) -> None:
        super().__init__()
        self.padding_idx = padding_idx
        self.emb = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
        self.embedding_dim = embedding_dim
        self.scale = self.embedding_dim ** 0.5
        self.init_parameters()

    def init_parameters(self) -> None:
        nn.init.normal_(self.emb.weight, mean=0, std=self.embedding_dim ** -0.5)
        if self.padding_idx is not None:
            with torch.no_grad():
                self.emb.weight[self.padding_idx].fill_(0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.emb(x) * self.scale


class LearnedPositionalEmbedding(TokenEmbedding):
    def __init__(self, embedding_dim: int, max_len: int) -> None:
        super().__init__(embedding_dim, max_len)
        self.embedding_dim = embedding_dim
        self.max_len = max_len

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(embedding_dim={self.embedding_dim}, max_len={self.max_len})"


class SinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, embedding_dim: int, max_len: int):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.max_len = max_len
        # Compute the positional encodings once in log space.
        pe = torch.zeros(self.max_len, self.embedding_dim, dtype=torch.float, requires_grad=False)
        position = torch.arange(0, self.max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.embedding_dim, 2, dtype=torch.float) *
                             -(math.log(10000.0) / self.embedding_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, pos: torch.Tensor) -> torch.Tensor:
        pos_emb = torch.index_select(self.pe, 0, pos.reshape(-1))
        return pos_emb.view((*pos.shape, -1))

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(embedding_dim={self.embedding_dim}, max_len={self.max_len})"


def embed_nodes_along_edge(g: dgl.DGLHeteroGraph,
                           src_node_type: str,
                           along_edge: str,
                           embedded_node_types: Set[str],
                           aggregate: str = "mean") -> None:
    assert aggregate in {"mean", "sum"}

    edge_types: List[Tuple[str, str, str]] = []
    for src, edge, dst in g.canonical_etypes:
        if (
                src == src_node_type
                and src in embedded_node_types
                and along_edge == edge
                and dst not in embedded_node_types
        ):
            edge_types.append((src, edge, dst))

    if len(edge_types) == 0:
        return

    reduce_fn = dfn.mean if aggregate == "mean" else dfn.sum
    for edge_type in edge_types:
        g.send_and_recv(
            edges=g.edges(etype=edge_type),
            message_func=dfn.copy_u("emb", "embeddings"),
            reduce_func=reduce_fn("embeddings", "emb"),
            etype=edge_type
        )
        _, _, dst = edge_type
        embedded_node_types.add(dst)
        embed_nodes_along_edge(g, dst, along_edge, embedded_node_types, aggregate)


class NodeEmbedding(nn.Module):
    def __init__(self,
                 hidden_feature: str,
                 embedding_dim: int,
                 sample_g: dgl.DGLHeteroGraph,
                 embeddings: Dict[str, Tuple[int, int]],
                 max_length: int,
                 dropout: float,
                 init_embeddings_along_edge_type: Optional[str] = None,
                 embed_node_types: bool = False,
                 embed_positions: bool = True,
                 learned_position_embeddings: bool = False) -> None:
        super().__init__()
        self.hidden_feature = hidden_feature
        self.embedding_dim = embedding_dim
        self.init_embeddings_along_edge_type = init_embeddings_along_edge_type

        if embed_node_types:
            self.node_type_emb = NodeTypeEmbedding(embedding_dim, sample_g, norm=False)
        else:
            self.node_type_emb = None

        self.token_emb = nn.ModuleDict(
            {
                node_type: TokenEmbedding(
                    num_embeddings=num_embeddings,
                    embedding_dim=embedding_dim,
                    padding_idx=padding_idx
                )
                for node_type, (num_embeddings, padding_idx) in embeddings.items()
            }
        )
        if embed_positions:
            self.pos_emb = nn.ModuleDict({
                node_type: LearnedPositionalEmbedding(embedding_dim, max_length)
                if learned_position_embeddings else SinusoidalPositionalEmbedding(embedding_dim, max_length)
                for node_type in embeddings
            })
        else:
            self.pos_emb = None

        self.norms = nn.ModuleDict({
            node_type: nn.LayerNorm(normalized_shape=embedding_dim)
            for node_type in sample_g.ntypes
        })

        self.drops = nn.ModuleDict({
            node_type: nn.Dropout(dropout)
            for node_type in sample_g.ntypes
        })

    def embed_tokens(self, nodes: NodeBatch) -> Dict[str, torch.Tensor]:
        emb = self.token_emb[nodes.ntype](nodes.data[f"{nodes.ntype}_id"].long())
        if self.pos_emb is not None:
            emb = emb + self.pos_emb[nodes.ntype](nodes.data["position"].long())
        return {"emb": emb}

    def norm_and_drop_emb(self, nodes: NodeBatch) -> Dict[str, torch.Tensor]:
        return {"emb": self.drops[nodes.ntype](self.norms[nodes.ntype](nodes.data["emb"]))}

    def forward(self, g: dgl.DGLHeteroGraph) -> Dict[str, torch.Tensor]:
        already_embedded_node_types = set(
            node_type for node_type in g.ntypes if self.hidden_feature in g.nodes[node_type].data
        )
        with g.local_scope():
            # token embed first (ignore already embedded nodes)
            for node_type in set(self.token_emb) - already_embedded_node_types:
                # keep hidden feature if already present in the graph
                g.apply_nodes(func=self.embed_tokens,
                              ntype=node_type)

            embedded_node_types = set(self.token_emb).union(already_embedded_node_types)

            if self.init_embeddings_along_edge_type is not None:
                for node_type in set(embedded_node_types):
                    embed_nodes_along_edge(g,
                                           node_type,
                                           self.init_embeddings_along_edge_type,
                                           embedded_node_types,
                                           aggregate="mean")

            # initialize all still unembedded nodes to zero
            for node_type in set(g.ntypes) - embedded_node_types:
                g.nodes[node_type].data["emb"] = torch.zeros(
                    (g.num_nodes(node_type), self.embedding_dim),
                    dtype=torch.float,
                    device=g.device
                )

            if self.node_type_emb is not None:
                node_type_embeddings = self.node_type_emb(g)
                # add node type embeddings to all newly embedded nodes
                for node_type in set(node_type_embeddings) - already_embedded_node_types:
                    g.nodes[node_type].data["emb"] = g.nodes[node_type].data["emb"] + node_type_embeddings[node_type]

            for node_type in set(g.ntypes) - already_embedded_node_types:
                if g.num_nodes(node_type) == 0:
                    continue

                g.apply_nodes(
                    func=self.norm_and_drop_emb,
                    ntype=node_type
                )

            for node_type in already_embedded_node_types:
                g.nodes[node_type].data["emb"] = g.nodes[node_type].data[self.hidden_feature]

            return {g.ntypes[0]: g.ndata["emb"]} if g.is_homogeneous else g.ndata["emb"]


class NodeTypeEmbedding(nn.Module):
    def __init__(self,
                 embedding_dim: int,
                 sample_g: dgl.DGLHeteroGraph,
                 norm: bool = True) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.node_types = sorted(sample_g.ntypes)

        self.node_type_dict = {node_type: i for i, node_type in enumerate(self.node_types)}
        self.node_type_emb = TokenEmbedding(num_embeddings=len(self.node_types),
                                            embedding_dim=embedding_dim)

        if norm:
            self.norms = nn.ModuleDict({
                node_type: nn.LayerNorm(normalized_shape=embedding_dim)
                for node_type in self.node_types
            })
        else:
            self.norms = None

    def embed_node_types(self, device: torch.device) -> Callable:
        def udf(nodes: NodeBatch) -> Dict[str, torch.Tensor]:
            n_type = nodes.ntype
            node_type_tensor = torch.full(
                (nodes.batch_size(),),
                fill_value=self.node_type_dict[n_type],
                dtype=torch.long,
                device=device
            )
            emb = self.node_type_emb(node_type_tensor)
            if self.norms:
                emb = self.norms[n_type](emb)
            return {"emb": emb}

        return udf

    def forward(self, g: dgl.DGLHeteroGraph) -> Dict[str, torch.Tensor]:
        with g.local_scope():
            for n_type in g.ntypes:
                if g.num_nodes(n_type) == 0:
                    continue
                g.apply_nodes(
                    func=self.embed_nodes_types(g.device),
                    ntype=n_type
                )
            return {g.ntypes[0]: g.ndata["emb"]} if g.is_homogeneous else g.ndata["emb"]


class EdgeTypeEmbedding(nn.Module):
    def __init__(self,
                 embedding_dim: int,
                 sample_g: dgl.DGLHeteroGraph,
                 dropout: float) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.edge_types = sorted(sample_g.etypes)

        self.edge_type_dict = {edge_type: i for i, edge_type in enumerate(self.edge_types)}
        self.edge_type_emb = TokenEmbedding(num_embeddings=len(self.edge_types),
                                            embedding_dim=embedding_dim)

        self.norms = nn.ModuleDict({
            edge_type: nn.LayerNorm(normalized_shape=embedding_dim)
            for edge_type in self.edge_types
        })

        self.drops = nn.ModuleDict({
            edge_type: nn.Dropout(dropout)
            for edge_type in self.edge_types
        })

    def embed_edge_types(self, device: torch.device) -> Callable:
        def udf(edges: EdgeBatch) -> Dict[str, torch.Tensor]:
            e_type = edges.canonical_etype[1]
            edge_type_tensor = torch.full(
                (edges.batch_size(),),
                fill_value=self.edge_type_dict[e_type],
                dtype=torch.long,
                device=device
            )
            emb = self.edge_type_emb(edge_type_tensor)
            emb = self.norms[e_type](emb)
            return {"emb": self.drops[e_type](emb)}

        return udf

    def forward(self, g: dgl.DGLHeteroGraph) -> Dict[Tuple[str, str, str], torch.Tensor]:
        with g.local_scope():
            for e_type in g.canonical_etypes:
                if g.num_edges(e_type) == 0:
                    continue
                g.apply_edges(
                    func=self.embed_edge_types(g.device),
                    etype=e_type
                )
            return {g.canonical_etypes[0]: g.edata["emb"]} if g.is_homogeneous else g.edata["emb"]


@dataclass
class GraphEmbeddingConfig:
    # embedding
    init_embeddings_along_edge_type: Optional[str] = None

    # positional embedding
    learned_position_embedding: bool = False
    embed_positions: bool = True

    embed_node_types: bool = False
    embed_edge_types: bool = False
    dropout: float = 0.1


class GraphEmbedding(nn.Module):
    def __init__(self,
                 sample_g: dgl.DGLHeteroGraph,
                 node_hidden_dim: int,
                 hidden_feature: str,
                 embeddings: Dict[str, Tuple[int, int]],
                 max_length: int,
                 cfg: GraphEmbeddingConfig,
                 edge_hidden_dim: Optional[int] = None) -> None:
        super().__init__()
        self.node_hidden_dim = node_hidden_dim
        self.edge_hidden_dim = edge_hidden_dim
        self.hidden_feature = hidden_feature
        self.cfg = cfg

        self.node_embedding = NodeEmbedding(
            hidden_feature=hidden_feature,
            embedding_dim=self.node_hidden_dim,
            sample_g=sample_g,
            embeddings=embeddings,
            max_length=max_length,
            dropout=cfg.dropout,
            init_embeddings_along_edge_type=cfg.init_embeddings_along_edge_type,
            embed_node_types=cfg.embed_node_types,
            embed_positions=cfg.embed_positions,
            learned_position_embeddings=cfg.learned_position_embedding
        )

        if self.cfg.embed_edge_types:
            assert self.edge_hidden_dim is not None, \
                "if you want to embed edges you need to provide an edge embedding dim"
            self.edge_type_embedding = EdgeTypeEmbedding(
                embedding_dim=self.edge_hidden_dim,
                sample_g=sample_g,
                dropout=cfg.dropout
            )
        else:
            self.edge_type_embedding = None

    def forward(self, g: dgl.DGLHeteroGraph) -> dgl.DGLHeteroGraph:
        h = self.node_embedding(g)
        g.ndata[self.hidden_feature] = h[g.ntypes[0]] if g.is_homogeneous else h

        if self.edge_type_embedding is not None:
            h_e = self.edge_type_embedding(g)
            g.edata[self.hidden_feature] = h_e[g.canonical_etypes[0]] if g.is_homogeneous else h_e

        return g


@dataclass
class TensorEmbeddingConfig:
    # positional embedding
    learned_position_embedding: bool = False
    embed_positions: bool = True

    dropout: float = 0.1


class TensorEmbedding(nn.Module):
    def __init__(self,
                 hidden_dim: int,
                 num_embeddings: int,
                 max_length: int,
                 cfg: TensorEmbeddingConfig,
                 padding_idx: Optional[int] = None) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_embeddings = num_embeddings
        self.cfg = cfg

        self.embedding = TokenEmbedding(
            hidden_dim,
            num_embeddings,
            padding_idx
        )

        if self.cfg.embed_positions:
            self.pos_emb = (
                LearnedPositionalEmbedding(hidden_dim, max_length) if self.cfg.learned_position_embedding
                else SinusoidalPositionalEmbedding(hidden_dim, max_length)
            )
        else:
            self.pos_emb = None

        self.norm = nn.LayerNorm(hidden_dim)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor, positions: Optional[torch.Tensor] = None) -> torch.Tensor:
        assert len(x.shape) == 2
        emb = self.embedding(x)
        if self.pos_emb is not None:
            if positions is None:
                positions = einops.repeat(
                    torch.arange(x.shape[1], dtype=torch.long, device=x.device),
                    "p -> b p",
                    b=x.shape[0]
                )
            emb = emb + self.pos_emb(positions)
        return self.drop(self.norm(emb))


def get_embedding_from_config(cfg: omegaconf.DictConfig,
                              sample_inputs: DataInput,
                              **kwargs: Any) -> Union[GraphEmbedding, TensorEmbedding]:
    if isinstance(sample_inputs, dgl.DGLHeteroGraph):
        cfg = omegaconf.OmegaConf.structured(GraphEmbeddingConfig(**cfg))
        return GraphEmbedding(
            sample_g=sample_inputs,
            cfg=cfg,
            **kwargs
        )
    else:
        cfg = omegaconf.OmegaConf.structured(TensorEmbeddingConfig(**cfg))
        return TensorEmbedding(
            cfg=cfg,
            **kwargs
        )
