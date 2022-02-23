from typing import Dict, Optional, Union, Tuple, Callable

import dgl
import einops
from dgl import function as dfn, DGLError
from dgl.ops import edge_softmax
from dgl.udf import EdgeBatch
from dgl.utils import expand_as_pair
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint as cp


class ConvolutionLayer(nn.Module):
    def __init__(self,
                 sample_g: dgl.DGLHeteroGraph,
                 node_input_dims: Dict[str, int],
                 node_hidden_dim: int,
                 dropout: float,
                 message_gating: bool = False,
                 activation: Optional[str] = "gelu",
                 edge_feat_dims: Optional[Dict[str, int]] = None) -> None:
        super().__init__()

        src_types = sorted(set(e_type[0] for e_type in sample_g.canonical_etypes))
        self.node_transforms = nn.ModuleDict({
            node_type: nn.Linear(node_input_dims[node_type], node_hidden_dim)
            for node_type in src_types
        })

        if activation is None:
            activation_module = nn.Identity()
        elif activation == "relu":
            activation_module = nn.ReLU()
        elif activation == "gelu":
            activation_module = nn.GELU()
        else:
            raise ValueError(f"Unknown activation {activation}, must be either relu or gelu")

        self.out_transforms = nn.ModuleDict({
            node_type: nn.Sequential(
                nn.Linear(node_hidden_dim, node_hidden_dim),
                activation_module
            )
            for node_type in sample_g.ntypes
        })

        self.edge_transforms = nn.ModuleDict({
            edge_type: nn.Linear(node_hidden_dim, node_hidden_dim)
            for edge_type in sample_g.etypes
        })

        self.message_gating = message_gating
        if self.message_gating:
            self.gate_transforms = nn.ModuleDict({
                edge_type: nn.Sequential(
                    nn.Linear(node_hidden_dim, node_hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(node_hidden_dim, 1),
                    nn.Sigmoid()
                )
                for edge_type in sample_g.etypes
            })

        self.normalize = "out_in_degree"

    def forward(self,
                g: dgl.DGLHeteroGraph,
                feat: Dict[str, torch.Tensor],
                edge_feat: Optional[Dict[Tuple[str, str, str], torch.Tensor]] = None) -> Dict[str, torch.Tensor]:
        with g.local_scope():
            for src, edge, dst in g.canonical_etypes:
                rel_graph: dgl.DGLHeteroGraph = g[src, edge, dst]
                if rel_graph.number_of_edges() == 0:
                    continue

                src_feat = self.node_transforms[src](feat[src])
                messages = self.edge_transforms[edge](src_feat)
                if self.message_gating:
                    messages = self.gate_transforms[edge](src_feat) * messages

                rel_type_str = "_".join((src, edge, dst))
                # normalizing messages for each relation:
                #   1. normalizing by destination node in-degree (same as averaging messages for each relation)
                #   2. normalizing by source and destination node degrees
                if self.normalize == "in_degree":
                    norm = 1 / rel_graph.in_degrees().clamp(min=1)
                elif self.normalize == "out_in_degree":
                    messages = messages * (rel_graph.out_degrees().clamp(min=1) ** -0.5).unsqueeze(1)
                    norm = rel_graph.in_degrees().clamp(min=1) ** -0.5
                else:
                    raise ValueError(f"Unknown normalization scheme {self.normalize}")

                rel_graph.srcdata[f"messages_{rel_type_str}"] = messages
                rel_graph.dstdata[f"norm_{rel_type_str}"] = norm

            g.multi_update_all(
                {
                    e_type: (
                        dfn.u_mul_v(f"messages_{'_'.join(e_type)}", f"norm_{'_'.join(e_type)}", "messages"),
                        dfn.sum("messages", "update")
                    ) for e_type in g.canonical_etypes if g.num_edges(e_type) > 0
                },
                cross_reducer="sum",
                apply_node_func=lambda nodes: {"update": self.out_transforms[nodes.ntype](nodes.data["update"])}
            )

            return {g.ntypes[0]: g.ndata["update"]} if g.is_homogeneous else g.ndata["update"]


class AttentionLayer(nn.Module):
    def __init__(self,
                 sample_g: dgl.DGLHeteroGraph,
                 node_input_dims: Dict[str, int],
                 node_hidden_dim: int,
                 dropout: float,
                 num_heads: Optional[int] = None,
                 message_gating: bool = False,
                 edge_feat_dims: Optional[Dict[str, int]] = None) -> None:
        super().__init__()

        src_types = sorted(set(e_type[0] for e_type in sample_g.canonical_etypes))
        dst_types = sorted(set(e_type[2] for e_type in sample_g.canonical_etypes))
        self.key_transforms = nn.ModuleDict({
            node_type: nn.Linear(node_input_dims[node_type], node_hidden_dim)
            for node_type in src_types
        })
        self.value_transforms = nn.ModuleDict({
            node_type: nn.Linear(node_input_dims[node_type], node_hidden_dim)
            for node_type in src_types
        })
        self.query_transforms = nn.ModuleDict({
            node_type: nn.Linear(node_input_dims[node_type], node_hidden_dim)
            for node_type in dst_types
        })
        self.out_transforms = nn.ModuleDict({
            node_type: nn.Linear(node_hidden_dim, node_hidden_dim)
            for node_type in dst_types
        })

        self.edge_transforms = nn.ModuleDict({
            edge_type: nn.Linear(node_hidden_dim, node_hidden_dim)
            for edge_type in sample_g.etypes
        })

        if num_heads is None:
            num_heads = max(node_hidden_dim // 64, 1)
        self.num_heads = num_heads
        self.head_dim = node_hidden_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.message_gating = message_gating
        if self.message_gating:
            self.gate_transforms = nn.ModuleDict({
                edge_type: nn.Sequential(
                    nn.Linear(node_hidden_dim, node_hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(node_hidden_dim, 1),
                    nn.Sigmoid()
                )
                for edge_type in sample_g.etypes
            })

        self.attention_dropout = nn.Dropout(dropout)

    def forward(self,
                g: dgl.DGLHeteroGraph,
                feat: Dict[str, torch.Tensor],
                edge_feat: Optional[Dict[Tuple[str, str, str], torch.Tensor]] = None) -> Dict[str, torch.Tensor]:
        with g.local_scope():
            for src, edge, dst in g.canonical_etypes:
                rel_graph: dgl.DGLHeteroGraph = g[src, edge, dst]
                if rel_graph.number_of_edges() == 0:
                    continue

                # if edge_feat is not None:
                #     e_feat = edge_feat[(src, edge, dst)]
                # else:
                #     e_feat = torch.zeros()

                k_feat = self.key_transforms[src](feat[src])
                k_feat = self.edge_transforms[edge](k_feat)

                q_feat = self.query_transforms[dst](feat[dst])
                q_feat = self.edge_transforms[edge](q_feat) * self.scale

                v_feat = self.value_transforms[src](feat[src])
                messages = self.edge_transforms[edge](v_feat)
                if self.message_gating:
                    messages = self.gate_transforms[edge](v_feat) * messages

                rel_type_str = "_".join((src, edge, dst))
                rel_graph.dstdata["queries"] = einops.rearrange(
                    q_feat,
                    "n (h hf) -> n h hf",
                    h=self.num_heads,
                    hf=self.head_dim
                )
                rel_graph.srcdata["keys"] = einops.rearrange(
                    k_feat,
                    "n (h hf) -> n h hf",
                    h=self.num_heads,
                    hf=self.head_dim
                )
                rel_graph.srcdata[f"messages_{rel_type_str}"] = einops.rearrange(
                    messages,
                    "n (h hf) -> n h hf",
                    h=self.num_heads,
                    hf=self.head_dim
                )

                rel_graph.apply_edges(dfn.u_dot_v("keys", "queries", f"attn_weights_{rel_type_str}"))
                rel_graph.edata[f"attn_weights_{rel_type_str}"] = self.attention_dropout(
                    edge_softmax(rel_graph, rel_graph.edata[f"attn_weights_{rel_type_str}"])
                )

            g.multi_update_all(
                {
                    e_type: (
                        dfn.u_mul_e(f"messages_{'_'.join(e_type)}", f"attn_weights_{'_'.join(e_type)}",
                                    "weighted_messages"),
                        dfn.sum("weighted_messages", "update")
                    ) for e_type in g.canonical_etypes if g.num_edges(e_type) > 0
                },
                cross_reducer="sum",
                apply_node_func=lambda nodes: {
                    "update": self.out_transforms[nodes.ntype](
                        einops.rearrange(nodes.data["update"], "n h hf -> n (h hf)")
                    )
                }
            )

            return {g.ntypes[0]: g.ndata["update"]} if g.is_homogeneous else g.ndata["update"]


def _cat_and_forward_checkpoint(module: nn.Module):
    def _cat_and_forward(*inputs: torch.Tensor) -> torch.Tensor:
        inputs_cat = torch.cat(inputs, dim=1)
        return module(inputs_cat)

    return _cat_and_forward


class MessagePassingLayer(nn.Module):
    def __init__(self,
                 sample_g: dgl.DGLHeteroGraph,
                 node_input_dims: Dict[str, int],
                 node_hidden_dim: int,
                 edge_hidden_dim: int,
                 dropout: float,
                 message_gating: bool = False,
                 edge_feat_dims: Optional[Dict[str, int]] = None) -> None:
        super().__init__()
        if edge_feat_dims is None:
            edge_feat_dims = {}

        src_types = sorted(set(e_type[0] for e_type in sample_g.canonical_etypes))
        dst_types = sorted(set(e_type[2] for e_type in sample_g.canonical_etypes))

        self.src_transforms = nn.ModuleDict({
            node_type: nn.Linear(node_input_dims[node_type], edge_hidden_dim)
            for node_type in src_types
        })
        self.dst_transforms = nn.ModuleDict({
            node_type: nn.Linear(node_input_dims[node_type], edge_hidden_dim)
            for node_type in dst_types
        })

        self.edge_transforms = nn.ModuleDict({
            edge_type: nn.Sequential(
                nn.Linear(edge_hidden_dim * 2 + edge_feat_dims.get(edge_type, 0), edge_hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(edge_hidden_dim, edge_hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(edge_hidden_dim, edge_hidden_dim),
                nn.LayerNorm(edge_hidden_dim)
            )
            for edge_type in sample_g.etypes
        })

        self.out_transforms = nn.ModuleDict({
            node_type: nn.Linear(edge_hidden_dim, node_hidden_dim)
            for node_type in dst_types
        })

        self.message_gating = message_gating
        if self.message_gating:
            self.gate_transforms = nn.ModuleDict({
                edge_type: nn.Sequential(
                    nn.Linear(edge_hidden_dim * 2 + edge_feat_dims.get(edge_type, 0), edge_hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(edge_hidden_dim, 1),
                    nn.Sigmoid()
                )
                for edge_type in sample_g.etypes
            })

    def generate_messages(self, edges: EdgeBatch) -> Dict[str, torch.Tensor]:
        rel_type_str = "_".join(edges.canonical_etype)
        e_type = edges.canonical_etype[1]
        # we use gradient checkpointing here for message computing
        # and message gating because this layer otherwise consumes way too much memory
        # because of this user defined message function, which copies the src and dst node features
        # and the tensor concatenation which then creates a new tensor out of these two
        uve_feat = [edges.src["src"], edges.dst["dst"]]
        if "edge" in edges.data:
            uve_feat.append(edges.data["edge"])
        messages_fn = _cat_and_forward_checkpoint(self.edge_transforms[e_type])
        messages = cp(messages_fn, *uve_feat)
        if self.message_gating:
            gating_fn = _cat_and_forward_checkpoint(self.gate_transforms[e_type])
            messages = cp(gating_fn, *uve_feat) * messages
        return {f"messages_{rel_type_str}": messages}

    def forward(self,
                g: dgl.DGLHeteroGraph,
                feat: Dict[str, torch.Tensor],
                edge_feat: Optional[Dict[Tuple[str, str, str], torch.Tensor]] = None) -> Dict[str, torch.Tensor]:
        with g.local_scope():
            for src, edge, dst in g.canonical_etypes:
                rel_graph: dgl.DGLHeteroGraph = g[src, edge, dst]
                if rel_graph.number_of_edges() == 0:
                    continue

                src_feat = self.src_transforms[src](feat[src])
                dst_feat = self.dst_transforms[dst](feat[dst])

                rel_graph.srcdata["src"] = src_feat
                rel_graph.dstdata["dst"] = dst_feat
                if edge_feat is not None:
                    rel_graph.edata["edge"] = edge_feat[(src, edge, dst)]

                rel_graph.apply_edges(self.generate_messages)

            g.multi_update_all(
                {
                    e_type: (
                        dfn.copy_e(f"messages_{'_'.join(e_type)}", "messages"),
                        dfn.sum("messages", "update")
                    ) for e_type in g.canonical_etypes if g.num_edges(e_type) > 0
                },
                cross_reducer="sum",
                apply_node_func=lambda nodes: {"update": self.out_transforms[nodes.ntype](nodes.data["update"])}
            )

            return {g.ntypes[0]: g.ndata["update"]} if g.is_homogeneous else g.ndata["update"]


class MultiHeadAttention(nn.Module):
    def __init__(self,
                 num_features: int,
                 num_heads: int,
                 dropout: float) -> None:
        super().__init__()
        self.num_features = num_features
        self.q_proj = nn.Linear(num_features, num_features)
        self.v_proj = nn.Linear(num_features, num_features)
        self.k_proj = nn.Linear(num_features, num_features)
        self.dropout_attn = nn.Dropout(dropout)
        self.out_proj = nn.Linear(num_features, num_features)
        self.num_heads = num_heads
        self.head_dim = num_features // num_heads
        self.scale = self.head_dim ** -0.5

    def forward(self,
                g: dgl.DGLHeteroGraph,
                feat: torch.Tensor
                ) -> torch.Tensor:
        with g.local_scope():
            # dot product attention as in Transformer
            if isinstance(feat, tuple):
                kv_feat, q_feat = feat[0], feat[1]
            else:
                q_feat = kv_feat = feat
            q = self.q_proj(q_feat) * self.scale
            g.dstdata["queries"] = einops.rearrange(
                q,
                "n (h hf) -> n h hf",
                h=self.num_heads,
                hf=self.head_dim
            )
            k = self.k_proj(kv_feat)
            g.srcdata["keys"] = einops.rearrange(
                k,
                "n (h hf) -> n h hf",
                h=self.num_heads,
                hf=self.head_dim
            )
            v = self.v_proj(kv_feat)
            g.srcdata["values"] = einops.rearrange(
                v,
                "n (h hf) -> n h hf",
                h=self.num_heads,
                hf=self.head_dim
            )

            g.apply_edges(
                func=dfn.u_dot_v("keys", "queries", "attn_weights")
            )
            g.edata["attn_weights"] = self.dropout_attn(
                edge_softmax(g, g.edata["attn_weights"])
            )
            g.update_all(
                message_func=dfn.u_mul_e(
                    "values",
                    "attn_weights",
                    "weighted_values"
                ),
                reduce_func=dfn.sum("weighted_values", "attn")
            )
            attn = einops.rearrange(
                g.dstdata["attn"],
                "n h hf -> n (h hf)"
            )
            attn = self.out_proj(attn)
            return attn


class TransformerEncoderConv(nn.Module):
    def __init__(self,
                 num_features: int,
                 num_heads: int,
                 feedforward_dim: int,
                 dropout: float) -> None:
        super().__init__()
        self.self_attention = MultiHeadAttention(
            num_features,
            num_heads,
            dropout
        )

        self.linear1 = nn.Linear(num_features, feedforward_dim)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(feedforward_dim, num_features)

        self.norm1 = nn.LayerNorm(num_features)
        self.norm2 = nn.LayerNorm(num_features)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self,
                g: dgl.DGLGraph,
                feat: torch.Tensor) -> torch.Tensor:
        assert g.is_homogeneous, f"transformer only works with homogeneous graphs"
        with g.local_scope():
            self_attn = self.self_attention(g, feat)
            # transform features as in Transformer
            if isinstance(feat, tuple):
                feat = feat[1]
            feat = self.norm1(feat + self.dropout1(self_attn))
            feat2 = self.linear2(self.dropout(F.gelu(self.linear1(feat))))
            feat = self.norm2(feat + self.dropout2(feat2))
            return feat


# copied and slightly modified
# from official implementation at https://github.com/tech-srl/how_attentive_are_gats/blob/main/gatv2_conv_DGL.py,
# introduced in https://arxiv.org/pdf/2105.14491.pdf
class GATv2Conv(nn.Module):
    r"""
    in_feats : int, or pair of ints
        Input feature size; i.e, the number of dimensions of :math:`h_i^{(l)}`.
    out_feats : int
        Output feature size; i.e, the number of dimensions of :math:`h_i^{(l+1)}`.
    num_heads : int
        Number of heads in Multi-Head Attention.
    feat_drop : float, optional
        Dropout rate on feature. Defaults: ``0``.
    attn_drop : float, optional
        Dropout rate on attention weight. Defaults: ``0``.
    negative_slope : float, optional
        LeakyReLU angle of negative slope. Defaults: ``0.2``.
    residual : bool, optional
        If True, use residual connection. Defaults: ``False``.
    activation : callable activation function/layer or None, optional.
        If not None, applies an activation function to the updated node features.
        Default: ``None``.
    allow_zero_in_degree : bool, optional
        If there are 0-in-degree nodes in the graph, output for those nodes will be invalid
        since no message will be passed to those nodes. This is harmful for some applications
        causing silent performance regression. This module will raise a DGLError if it detects
        0-in-degree nodes in input graph. By setting ``True``, it will suppress the check
        and let the users handle it by themselves. Defaults: ``False``.
    bias (bool, optional): If set to :obj:`False`, the layer will not learn
        an additive bias. (default: :obj:`True`)
    share_weights (bool, optional): If set to :obj:`True`, the same matrix
        will be applied to the source and the target node of every edge.
        (default: :obj:`False`)
    """

    def __init__(self,
                 in_feats: Union[Tuple[int, int], int],
                 out_feats: int,
                 num_heads: int,
                 feat_drop: float = 0.,
                 attn_drop: float = 0.,
                 negative_slope: float = 0.2,
                 residual: bool = False,
                 activation: Optional[Callable] = None,
                 allow_zero_in_degree: bool = False,
                 bias: bool = True,
                 share_weights: bool = False) -> None:
        super().__init__()
        self._num_heads = num_heads
        self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)
        self._out_feats = out_feats
        self._allow_zero_in_degree = allow_zero_in_degree
        if isinstance(in_feats, tuple):
            self.fc_src = nn.Linear(
                self._in_src_feats, out_feats * num_heads, bias=bias)
            self.fc_dst = nn.Linear(
                self._in_dst_feats, out_feats * num_heads, bias=bias)
        else:
            self.fc_src = nn.Linear(
                self._in_src_feats, out_feats * num_heads, bias=bias)
            if share_weights:
                self.fc_dst = self.fc_src
            else:
                self.fc_dst = nn.Linear(
                    self._in_src_feats, out_feats * num_heads, bias=bias)
        self.attn = nn.Parameter(torch.zeros(size=(1, num_heads, out_feats), dtype=torch.float))
        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        if residual:
            if self._in_dst_feats != out_feats:
                self.res_fc = nn.Linear(
                    self._in_dst_feats, num_heads * out_feats, bias=bias)
            else:
                self.res_fc = nn.Identity()
        else:
            self.register_buffer('res_fc', None)
        self.reset_parameters()
        self.activation = activation

    def reset_parameters(self) -> None:
        """
        Description
        -----------
        Reinitialize learnable parameters.
        Note
        ----
        The fc weights :math:`W^{(l)}` are initialized using Glorot uniform initialization.
        The attention weights are using xavier initialization method.
        """
        gain = nn.init.calculate_gain('relu')
        if hasattr(self, 'fc'):
            nn.init.xavier_normal_(self.fc.weight, gain=gain)
        else:
            nn.init.xavier_normal_(self.fc_src.weight, gain=gain)
            nn.init.xavier_normal_(self.fc_dst.weight, gain=gain)
        nn.init.xavier_normal_(self.attn, gain=gain)
        if isinstance(self.res_fc, nn.Linear):
            nn.init.xavier_normal_(self.res_fc.weight, gain=gain)

    def set_allow_zero_in_degree(self, set_value: bool) -> None:
        r"""
        Description
        -----------
        Set allow_zero_in_degree flag.
        Parameters
        ----------
        set_value : bool
            The value to be set to the flag.
        """
        self._allow_zero_in_degree = set_value

    def forward(self,
                graph: dgl.DGLHeteroGraph,
                feat: Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor],
                get_attention: bool = False) -> Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        r"""
        Description
        -----------
        Compute graph attention network layer.
        Parameters
        ----------
        graph : DGLGraph
            The graph.
        feat : torch.Tensor or pair of torch.Tensor
            If a torch.Tensor is given, the input feature of shape :math:`(N, D_{in})` where
            :math:`D_{in}` is size of input feature, :math:`N` is the number of nodes.
            If a pair of torch.Tensor is given, the pair must contain two tensors of shape
            :math:`(N_{in}, D_{in_{src}})` and :math:`(N_{out}, D_{in_{dst}})`.
        get_attention : bool, optional
            Whether to return the attention values. Default to False.
        Returns
        -------
        torch.Tensor
            The output feature of shape :math:`(N, H, D_{out})` where :math:`H`
            is the number of heads, and :math:`D_{out}` is size of output feature.
        torch.Tensor, optional
            The attention values of shape :math:`(E, H, 1)`, where :math:`E` is the number of
            edges. This is returned only when :attr:`get_attention` is ``True``.
        Raises
        ------
        DGLError
            If there are 0-in-degree nodes in the input graph, it will raise DGLError
            since no message will be passed to those nodes. This will cause invalid output.
            The error can be ignored by setting ``allow_zero_in_degree`` parameter to ``True``.
        """
        with graph.local_scope():
            if not self._allow_zero_in_degree:
                if (graph.in_degrees() == 0).any():
                    raise DGLError('There are 0-in-degree nodes in the graph, '
                                   'output for those nodes will be invalid. '
                                   'This is harmful for some applications, '
                                   'causing silent performance regression. '
                                   'Adding self-loop on the input graph by '
                                   'calling `g = dgl.add_self_loop(g)` will resolve '
                                   'the issue. Setting ``allow_zero_in_degree`` '
                                   'to be `True` when constructing this module will '
                                   'suppress the check and let the code run.')

            if isinstance(feat, tuple):
                h_src = self.feat_drop(feat[0])
                h_dst = self.feat_drop(feat[1])
                feat_src = self.fc_src(h_src).view(-1, self._num_heads, self._out_feats)
                feat_dst = self.fc_dst(h_dst).view(-1, self._num_heads, self._out_feats)
            else:
                h_src = h_dst = self.feat_drop(feat)
                feat_src = self.fc_src(h_src).view(
                    -1, self._num_heads, self._out_feats)
                feat_dst = self.fc_dst(h_src).view(
                    -1, self._num_heads, self._out_feats)
                if graph.is_block:
                    feat_dst = feat_src[:graph.number_of_dst_nodes()]
            graph.srcdata.update({'el': feat_src})  # (num_src_edge, num_heads, out_dim)
            graph.dstdata.update({'er': feat_dst})
            graph.apply_edges(dfn.u_add_v('el', 'er', 'e'))
            e = self.leaky_relu(graph.edata.pop('e'))  # (num_src_edge, num_heads, out_dim)
            e = (e * self.attn).sum(dim=-1).unsqueeze(dim=2)  # (num_edge, num_heads, 1)
            # compute softmax
            graph.edata['a'] = self.attn_drop(edge_softmax(graph, e))  # (num_edge, num_heads)
            # message passing
            graph.update_all(dfn.u_mul_e('el', 'a', 'm'),
                             dfn.sum('m', 'ft'))
            rst = graph.dstdata['ft']
            # residual
            if self.res_fc is not None:
                resval = self.res_fc(h_dst).view(h_dst.shape[0], -1, self._out_feats)
                rst = rst + resval
            # activation
            if self.activation:
                rst = self.activation(rst)

            if get_attention:
                return rst, graph.edata['a']
            else:
                return rst
