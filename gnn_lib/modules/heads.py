from typing import Dict, Any, Optional, List, Set

import dgl
import einops
import torch
from torch import nn
from torch.nn import functional as F

from gnn_lib.modules import utils, embedding


class Head(nn.Module):
    def forward(self, g: dgl.DGLHeteroGraph, **kwargs: Any) -> Any:
        raise NotImplementedError


class GraphClassificationHead(Head):
    def __init__(self,
                 feat: str,
                 num_features: int,
                 num_classes: int,
                 node_type: str,
                 pooling_type: str = "mean"
                 ):
        super().__init__()
        self.feat = feat
        self.pooling_type = pooling_type
        self.node_type = node_type

        self.clf = nn.Sequential(
            nn.Linear(in_features=num_features,
                      out_features=num_features),
            nn.GELU(),
            nn.Linear(in_features=num_features,
                      out_features=num_classes)
        )

    def forward(self, g: dgl.DGLHeteroGraph, **kwargs: Any) \
            -> torch.Tensor:
        if self.pooling_type == "mean":
            h = dgl.mean_nodes(g, self.feat, ntype=self.node_type)
        elif self.pooling_type == "sum":
            h = dgl.sum_nodes(g, self.feat, ntype=self.node_type)
        elif self.pooling_type == "max":
            h = dgl.max_nodes(g, self.feat, ntype=self.node_type)
        elif self.pooling_type == "first_node":
            indices = g.batch_num_nodes(self.node_type)
            indices = torch.cat([torch.zeros(1, dtype=indices.dtype, device=indices.device), indices[:-1]])
            h = torch.index_select(g.nodes[self.node_type].data[self.feat], 0, indices)
        else:
            raise ValueError(f"Unknown pooling type {self.pooling_type}")
        return self.clf(torch.flatten(h, 1))


class MultiNodeClassificationGroupHead(Head):
    def __init__(self,
                 feat: str,
                 num_features: int,
                 num_classes: Dict[str, int],
                 group_nodes: Optional[Set[str]] = None,
                 group_feature: str = "group",
                 group_aggregation: str = "mean"):
        super().__init__()
        self.feat = feat
        self.num_classes = num_classes

        self.clf = nn.ModuleDict(
            {
                node_type: nn.Sequential(
                    nn.Linear(in_features=num_features,
                              out_features=num_features),
                    nn.GELU(),
                    nn.Linear(in_features=num_features,
                              out_features=classes)
                )
                for node_type, classes in self.num_classes.items()
            }
        )

        self.group_nodes = set() if group_nodes is None else group_nodes
        self.group_feature = group_feature
        self.group_aggregation = group_aggregation

    def forward(self, g: dgl.DGLHeteroGraph, **kwargs: Any) \
            -> Dict[str, torch.Tensor]:
        h = {g.ntypes[0]: g.ndata[self.feat]} if g.is_homogeneous else g.ndata[self.feat]

        outputs = {}
        for node_type in self.num_classes:
            if node_type not in h:
                continue
            if node_type not in self.group_nodes:
                outputs[node_type] = self.clf[node_type](h[node_type])
            else:
                h_grouped = []
                batch_num_nodes = utils.tensor_to_python(g.batch_num_nodes(node_type), force_list=True)
                batch_groups = torch.split(g.nodes[node_type].data[self.group_feature], batch_num_nodes)
                batch_feats = torch.split(h[node_type], batch_num_nodes)
                for groups, feats in zip(batch_groups, batch_feats):
                    _, group_lengths = torch.unique(groups, sorted=True, return_counts=True)
                    group_feats = torch.split(feats, utils.tensor_to_python(group_lengths, force_list=True))

                    for group_feat in group_feats:
                        if self.group_aggregation == "mean":
                            group_feat = torch.mean(group_feat, dim=0, keepdim=True)
                        else:
                            raise ValueError(f"Unknown group aggregation {self.group_aggregation}")
                        h_grouped.append(group_feat)

                    # grouped = [[] for _ in range(groups[-1] + 1)]
                    # for group, feat in zip(groups, feats):
                    #     grouped[group].append(feat)
                    # for i, group_feats in enumerate(grouped):
                    #     group_feats = torch.stack(group_feats)
                    #     if self.group_aggregation == "mean":
                    #         group_feats = torch.mean(group_feats, dim=0)
                    #     else:
                    #         raise ValueError(f"Unknown group aggregation {self.group_aggregation}")
                    #     h_grouped.append(group_feats)

                outputs[node_type] = self.clf[node_type](torch.cat(h_grouped, dim=0))
        return outputs


class SequenceDecoderHead(Head, utils.DecoderMixin):
    def __init__(self,
                 hidden_feature: str,
                 hidden_dim: int,
                 bos_token_id: int,
                 eos_token_id: int,
                 max_length: int) -> None:
        super().__init__()
        self.hidden_feature = hidden_feature
        self.hidden_dim = hidden_dim
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.max_length = max_length

    def forward(self,
                _: dgl.DGLHeteroGraph,
                decoder_inputs: Optional[torch.Tensor] = None,
                decoder_lengths: Optional[torch.Tensor] = None,
                encoder_outputs: Optional[Dict[str, torch.Tensor]] = None,
                encoder_lengths: Optional[Dict[str, torch.Tensor]] = None,
                **kwargs: Any) -> torch.Tensor:
        assert (encoder_outputs is not None
                and encoder_lengths is not None
                and decoder_inputs is not None
                and decoder_lengths is not None)
        return self.decode(
            decoder_inputs=decoder_inputs,
            decoder_lengths=decoder_lengths,
            encoder_outputs=encoder_outputs,
            encoder_lengths=encoder_lengths,
            **kwargs
        )


# modified version of standard pytorch nn.TransformerDecoderLayer for cross attention to multiple memories
class MultiNodeTransformerDecoderLayer(nn.Module):
    def __init__(self,
                 context_node_types: List[str],
                 hidden_dim: int,
                 num_heads: int,
                 feed_forward_dim: int,
                 dropout: float,
                 batch_first: bool = True) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.context_node_types = context_node_types
        self.self_attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=batch_first)

        self.multihead_attns = nn.ModuleDict({
            node_type: nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=batch_first)
            for node_type in context_node_types
        })

        self.linear1 = nn.Linear(hidden_dim, feed_forward_dim)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(feed_forward_dim, hidden_dim)

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norms = nn.ModuleDict({
            node_type: nn.LayerNorm(hidden_dim) for node_type in context_node_types
        })
        self.norm3 = nn.LayerNorm(hidden_dim)

        self.dropout1 = nn.Dropout(dropout)
        self.dropouts = nn.ModuleDict({
            node_type: nn.LayerNorm(hidden_dim) for node_type in context_node_types
        })
        self.dropout3 = nn.Dropout(dropout)

    def forward(self,
                tgt: torch.Tensor,
                memory: Dict[str, torch.Tensor],
                tgt_mask: Optional[torch.Tensor] = None,
                memory_masks: Optional[Dict[str, torch.Tensor]] = None,
                tgt_key_padding_mask: Optional[torch.Tensor] = None,
                memory_key_padding_masks: Optional[Dict[str, torch.Tensor]] = None) -> torch.Tensor:
        x = tgt

        if tgt_mask is not None and tgt_mask.ndim > 2 and len(tgt_mask) != len(tgt) * self.num_heads:
            tgt_mask = einops.repeat(tgt_mask, "b l s -> (b nh) l s", nh=self.num_heads)

        x = self.norm1(x + self._sa_block(x, tgt_mask, tgt_key_padding_mask))
        for node_type in self.context_node_types:
            if node_type not in memory:
                continue

            memory_mask = memory_masks.get(node_type, None) if memory_masks is not None else None
            if (
                    memory_mask is not None and memory_mask.ndim > 2
                    and len(memory_mask) != len(memory[node_type]) * self.num_heads
            ):
                memory_mask = einops.repeat(memory_mask, "b l s -> (b nh) l s", nh=self.num_heads)

            x = self.norms[node_type](
                x + self._mha_block(
                    node_type,
                    x,
                    memory[node_type],
                    memory_mask,
                    memory_key_padding_masks.get(node_type, None) if memory_key_padding_masks is not None else None
                )
            )
        x = self.norm3(x + self._ff_block(x))
        return x

    def _sa_block(self,
                  x: torch.Tensor,
                  attn_mask: Optional[torch.Tensor],
                  key_padding_mask: Optional[torch.Tensor]) -> torch.Tensor:
        x = self.self_attn(
            x, x, x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False
        )[0]
        return self.dropout1(x)

    def _mha_block(self,
                   node_type: str,
                   x: torch.Tensor,
                   mem: torch.Tensor,
                   attn_mask: Optional[torch.Tensor] = None,
                   key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.multihead_attns[node_type](
            x, mem, mem,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False
        )[0]
        return self.dropouts[node_type](x)

    def _ff_block(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear2(self.dropout(F.gelu(self.linear1(x))))
        return self.dropout3(x)


class TransformerDecoderHead(SequenceDecoderHead):
    def __init__(self,
                 context_node_types: List[str],
                 bos_token_id: int,
                 eos_token_id: int,
                 pad_token_id: int,
                 max_length: int,
                 hidden_feature: str,
                 hidden_dim: int,
                 num_outputs: int,
                 dropout: float,
                 num_layers: int = 1,
                 share_parameters: bool = False,
                 feed_forward_dim: Optional[int] = None,
                 num_heads: Optional[int] = None) -> None:
        super().__init__(
            hidden_feature, hidden_dim, bos_token_id, eos_token_id, max_length
        )
        self.context_node_types = context_node_types
        self.num_outputs = num_outputs
        self.num_layers = num_layers
        self.share_parameters = share_parameters

        self.decoder_token_emb = embedding.TokenEmbedding(self.hidden_dim, num_outputs, padding_idx=pad_token_id)
        self.decoder_pos_emb = embedding.SinusoidalPositionalEmbedding(self.hidden_dim)
        self.norm_emb = nn.LayerNorm(self.hidden_dim)

        feed_forward_dim = feed_forward_dim if feed_forward_dim is not None else hidden_dim * 2
        num_heads = num_heads if num_heads is not None else max(1, hidden_dim // 64)

        self.decoder_layers = nn.ModuleList(
            MultiNodeTransformerDecoderLayer(
                context_node_types=context_node_types,
                hidden_dim=hidden_dim,
                feed_forward_dim=feed_forward_dim,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True
            )
            for _ in range(1 if self.share_parameters else num_layers)
        )

        self.clf = nn.Linear(in_features=hidden_dim,
                             out_features=num_outputs)

    def decode(self,
               decoder_inputs: torch.Tensor,
               decoder_lengths: Optional[torch.Tensor] = None,
               encoder_outputs: Optional[Dict[str, torch.Tensor]] = None,
               encoder_lengths: Optional[Dict[str, torch.Tensor]] = None,
               encoder_masks: Optional[Dict[str, torch.Tensor]] = None,
               decoder_mask: Optional[torch.Tensor] = None,
               decoder_padding_mask: Optional[torch.Tensor] = None,
               encoder_padding_masks: Optional[Dict[str, torch.Tensor]] = None,
               decoder_positions: Optional[torch.Tensor] = None) -> torch.Tensor:
        # decoder_inputs: shape [B, L]
        _, L = decoder_inputs.shape
        if decoder_positions is None:
            decoder_positions = torch.arange(
                L, device=decoder_inputs.device, dtype=torch.long
            ).unsqueeze(0)

        # dec: shape [B, L, H]
        dec = self.norm_emb(self.decoder_token_emb(decoder_inputs) + self.decoder_pos_emb(decoder_positions))

        # mask the encoder padding if encoder lengths are given and encoder padding masks are not given
        if encoder_lengths is not None and encoder_padding_masks is None:
            encoder_padding_masks = {}
            for node_type, lengths in encoder_lengths.items():
                encoder_padding_masks[node_type] = utils.padding_mask(
                    encoder_outputs[node_type], lengths
                )
        # mask the decoder padding if decoder lengths are given and decoder padding mask is not given
        if decoder_lengths is not None and decoder_padding_mask is None:
            decoder_padding_mask = utils.padding_mask(dec, decoder_lengths)

        # by default mask all tokens in the future
        if decoder_mask is None:
            decoder_mask = utils.square_causal_mask(L, dec.device)

        for i in range(self.num_layers):
            i = 0 if self.share_parameters else i
            dec = self.decoder_layers[i](
                tgt=dec,
                memory=encoder_outputs,
                tgt_mask=decoder_mask,
                memory_masks=encoder_masks,
                tgt_key_padding_mask=decoder_padding_mask,
                memory_key_padding_masks=encoder_padding_masks
            )

        return self.clf(dec)
