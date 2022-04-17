from typing import Dict, Any, Optional, List

import dgl
import einops
import torch
from nsc.modules import utils, embedding
from torch import nn
from torch.nn import functional as F


class GraphHead(nn.Module):
    def forward(self, g: dgl.DGLHeteroGraph, **kwargs: Any) -> Any:
        raise NotImplementedError


class GraphClassificationHead(GraphHead):
    def __init__(
            self,
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
        else:
            raise ValueError(f"Unknown pooling type {self.pooling_type}")
        return self.clf(torch.flatten(h, 1))


class MultiNodeClassificationGroupHead(GraphHead):
    def __init__(
            self,
            feat: str,
            num_features: int,
            num_classes: Dict[str, int],
            num_additional_features: Optional[Dict[str, Dict[str, int]]] = None,
            aggregation: Optional[Dict[str, Dict[str, str]]] = None
    ) -> None:
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

        self.additional_features = nn.ModuleDict({
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
            for node_type in (num_additional_features or {})
        })

        self.aggregation = aggregation or {}

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
            if groups is None or node_type not in groups[0]:
                outputs[node_type] = self.clf[node_type](h[node_type])
                continue

            stages = [group["stage"] for group in groups[0][node_type]]
            grouped_feats = torch.split(
                h[node_type],
                utils.tensor_to_python(g.batch_num_nodes(node_type), force_list=True)
            )

            for i, stage in enumerate(stages):
                feature_encoder = (
                    self.additional_features[node_type][stage]
                    if node_type in self.additional_features and stage in self.additional_features[node_type]
                    else None
                )
                aggregation = (
                    self.aggregation[node_type][stage]
                    if node_type in self.aggregation and stage in self.aggregation[node_type]
                    else "mean"
                )
                grouped_feats = utils.group_features(
                    grouped_feats=grouped_feats,
                    groups=[group[node_type][i] for group in groups],
                    additional_feature_encoder=feature_encoder,
                    aggregation=aggregation
                )

            outputs[node_type] = self.clf[node_type](torch.cat(grouped_feats, dim=0))

        return outputs


class TensorGroupHead(nn.Module):
    def __init__(
            self,
            hidden_dim: int,
            num_classes: int,
            num_additional_features: Optional[Dict[str, int]] = None,
            aggregation: Optional[Dict[str, str]] = None
    ) -> None:
        super().__init__()
        self.clf = nn.Sequential(
            nn.Linear(in_features=hidden_dim,
                      out_features=hidden_dim),
            nn.GELU(),
            nn.Linear(in_features=hidden_dim,
                      out_features=num_classes)
        )

        self.additional_features = nn.ModuleDict(
            {
                stage: nn.Sequential(
                    nn.Linear(in_features=hidden_dim + additional_features,
                              out_features=hidden_dim),
                    nn.GELU(),
                    nn.Linear(in_features=hidden_dim,
                              out_features=hidden_dim)
                )
                for stage, additional_features in (num_additional_features or {}).items()
            }
        )

        self.aggregation = aggregation or {}

    def forward(
            self,
            x: List[torch.Tensor],
            groups: Optional[List[List[Dict[str, Any]]]] = None,
            **kwargs: Any
    ) -> List[torch.Tensor]:
        if groups is not None:
            assert len(groups) == len(x)
            stages = [group["stage"] for group in groups[0]]
            # cycle through stages
            for i, stage in enumerate(stages):
                feature_encoder = self.additional_features[stage] if stage in self.additional_features else None
                aggregation = self.aggregation[stage] if stage in self.aggregation else "mean"
                x = utils.group_features(
                    grouped_feats=x,
                    groups=[group[i] for group in groups],
                    additional_feature_encoder=feature_encoder,
                    aggregation=aggregation
                )
        return list(torch.split(self.clf(torch.cat(x, dim=0)), [len(t) for t in x]))


class SequenceDecoderHead(utils.DecoderMixin):
    def __init__(self,
                 hidden_dim: int,
                 max_length: int) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_length = max_length

    def forward(self,
                decoder_inputs: torch.Tensor,
                **kwargs: Any) -> torch.Tensor:
        return self.decode(
            decoder_inputs=decoder_inputs,
            **kwargs
        )


# modified version of standard pytorch nn.TransformerDecoderLayer for cross attention to multiple memories/contexts
class MultiContextTransformerDecoderLayer(nn.Module):
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

        self.context_attn = nn.ModuleDict({
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
            need_weights=True
        )[0]
        return self.dropout1(x)

    def _mha_block(self,
                   ctx_name: str,
                   x: torch.Tensor,
                   mem: torch.Tensor,
                   attn_mask: Optional[torch.Tensor] = None,
                   key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.context_attn[ctx_name](
            x, mem, mem,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=True
        )[0]
        return self.dropouts[ctx_name](x)

    def _ff_block(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear2(self.dropout(F.gelu(self.linear1(x))))
        return self.dropout3(x)


# for legacy models
MultiNodeTransformerDecoderLayer = MultiContextTransformerDecoderLayer


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
            max_length=max_length,
            cfg=embedding.TensorEmbeddingConfig(dropout=dropout),
            padding_idx=pad_token_id
        )

        feed_forward_dim = feed_forward_dim if feed_forward_dim is not None else hidden_dim * 2
        num_heads = num_heads if num_heads is not None else max(1, hidden_dim // 64)

        self.decoder_layers = nn.ModuleList(
            MultiContextTransformerDecoderLayer(
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
            self.clf.weight = self.emb.embedding.emb.weight

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
