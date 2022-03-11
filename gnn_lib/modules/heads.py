from typing import Dict, Any, Optional, List

import dgl
import einops
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from gnn_lib.modules import utils, embedding


class GraphHead(nn.Module):
    def forward(self, g: dgl.DGLHeteroGraph, **kwargs: Any) -> Any:
        raise NotImplementedError


class GraphClassificationHead(GraphHead):
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


class MultiNodeClassificationGroupHead(GraphHead):
    def __init__(self,
                 feat: str,
                 num_features: int,
                 num_classes: Dict[str, int],
                 num_additional_features: Dict[str, Dict[str, int]],
                 aggregation: Dict[str, Dict[str, str]]):
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

        self.additional_features = nn.ModuleDict(
            {
                node_type: nn.ModuleDict({
                    stage: nn.Sequential(
                        nn.Linear(in_features=num_features + additional_features,
                                  out_features=num_features),
                        nn.GELU(),
                        nn.Linear(in_features=num_features,
                                  out_features=num_features)
                    )
                    for stage, additional_features in num_additional_features[node_type].items()
                })
                for node_type in num_additional_features
            }
        )

        self.aggregation = aggregation

    def forward(
            self,
            g: dgl.DGLHeteroGraph,
            groups: Optional[List[Dict[str, List[Dict[str, Any]]]]] = None,
            **kwargs: Any
    ) -> Dict[str, torch.Tensor]:
        h = {g.ntypes[0]: g.ndata[self.feat]} if g.is_homogeneous else g.ndata[self.feat]

        outputs = {}
        for node_type in self.num_classes:
            if node_type not in h:
                continue

            # when there is no need to group, just run the classifier on the node features
            if groups is None or len(groups) == 0 or node_type not in groups[0]:
                outputs[node_type] = self.clf[node_type](h[node_type])
                continue

            batch_num_nodes = utils.tensor_to_python(g.batch_num_nodes(node_type), force_list=True)
            assert len(groups) == len(batch_num_nodes)
            grouped_feats = h[node_type]

            # iterate through stages
            for i in range(len(groups[0][node_type])):
                stage_name = groups[0][node_type][i]["stage"]
                aggregation = self.aggregation[node_type][stage_name]

                has_additional_features = (
                        node_type in self.additional_features and stage_name in self.additional_features[node_type]
                )
                batch_groups = []
                stage_features = []
                for group in groups:
                    batch_groups.append(group[node_type][i]["groups"])
                    if has_additional_features:
                        stage_features.extend(group[node_type][i]["features"])

                if has_additional_features:
                    stage_features = torch.tensor(stage_features, dtype=torch.float, device=g.device)
                    grouped_feats = self.additional_features[node_type][stage_name](
                        torch.cat([grouped_feats, stage_features], dim=1)
                    )

                batch_grouped_feats = torch.split(grouped_feats, [len(batch_group) for batch_group in batch_groups])
                grouped_feats = []
                for batch_group, batch_grouped_feat in zip(batch_groups, batch_grouped_feats):
                    _, batch_group_lengths = np.unique(batch_group, return_counts=True)

                    for group_feat in torch.split(batch_grouped_feat, list(batch_group_lengths)):
                        if aggregation == "mean":
                            group_feat = torch.mean(group_feat, dim=0, keepdim=True)
                        elif aggregation == "max":
                            group_feat = torch.max(group_feat, dim=0, keepdim=True).values
                        elif aggregation == "sum":
                            group_feat = torch.sum(group_feat, dim=0, keepdim=True)
                        else:
                            raise ValueError(f"Unknown group aggregation {self.group_aggregation}")
                        grouped_feats.append(group_feat)

                grouped_feats = torch.cat(grouped_feats, dim=0)

            outputs[node_type] = self.clf[node_type](grouped_feats)
        return outputs


class SequenceDecoderHead(utils.DecoderMixin):
    def __init__(self,
                 hidden_dim: int,
                 max_length: int) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_length = max_length

    def forward(self,
                decoder_inputs: torch.Tensor,
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


# modified version of standard pytorch nn.TransformerDecoderLayer for cross attention to multiple memories/contexts
class MultiNodeTransformerDecoderLayer(nn.Module):
    def __init__(self,
                 contexts: List[str],
                 hidden_dim: int,
                 num_heads: int,
                 feed_forward_dim: int,
                 dropout: float,
                 batch_first: bool = True) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.contexts = contexts
        self.self_attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=batch_first)

        self.multihead_attns = nn.ModuleDict({
            ctx_name: nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=batch_first)
            for ctx_name in contexts
        })

        self.linear1 = nn.Linear(hidden_dim, feed_forward_dim)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(feed_forward_dim, hidden_dim)

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norms = nn.ModuleDict({
            node_type: nn.LayerNorm(hidden_dim) for node_type in contexts
        })
        self.norm3 = nn.LayerNorm(hidden_dim)

        self.dropout1 = nn.Dropout(dropout)
        self.dropouts = nn.ModuleDict({
            node_type: nn.LayerNorm(hidden_dim) for node_type in contexts
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
        for ctx_name in self.contexts:
            if ctx_name not in memory:
                continue

            memory_mask = memory_masks.get(ctx_name, None) if memory_masks is not None else None
            if (
                    memory_mask is not None and memory_mask.ndim > 2
                    and len(memory_mask) != len(memory[ctx_name]) * self.num_heads
            ):
                memory_mask = einops.repeat(memory_mask, "b l s -> (b nh) l s", nh=self.num_heads)

            x = self.norms[ctx_name](
                x + self._mha_block(
                    ctx_name,
                    x,
                    memory[ctx_name],
                    memory_mask,
                    memory_key_padding_masks.get(ctx_name, None) if memory_key_padding_masks is not None else None
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
                 contexts: List[str],
                 pad_token_id: int,
                 max_length: int,
                 hidden_dim: int,
                 num_outputs: int,
                 dropout: float,
                 num_layers: int = 1,
                 share_parameters: bool = False,
                 feed_forward_dim: Optional[int] = None,
                 num_heads: Optional[int] = None,
                 share_input_output_embeddings: bool = True) -> None:
        super().__init__(
            hidden_dim, max_length
        )
        self.contexts = contexts
        self.num_outputs = num_outputs
        self.num_layers = num_layers
        self.share_parameters = share_parameters

        self.emb = embedding.TensorEmbedding(
            hidden_dim=self.hidden_dim,
            num_embeddings=num_outputs,
            cfg=embedding.TensorEmbeddingConfig(dropout=dropout),
            padding_idx=pad_token_id
        )

        feed_forward_dim = feed_forward_dim if feed_forward_dim is not None else hidden_dim * 2
        num_heads = num_heads if num_heads is not None else max(1, hidden_dim // 64)

        self.decoder_layers = nn.ModuleList(
            MultiNodeTransformerDecoderLayer(
                contexts=contexts,
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

        if share_input_output_embeddings:
            self.cfg.weight = self.emb.embedding.emb.weight

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
        dec = self.emb(decoder_inputs, positions=decoder_positions)
        # dec: shape [B, L, H]

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
            decoder_mask = utils.square_causal_mask(dec.shape[1], dec.device)

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
