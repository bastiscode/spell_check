import copy
import enum
import time
from dataclasses import dataclass
from typing import Optional, Dict, Tuple, List, Any

import dgl
import numpy as np
import omegaconf
import torch
from omegaconf import MISSING
from torch import nn
from torch.nn import init

from gnn_lib.data import tokenization
from gnn_lib.data.tokenization import TokenizerConfig, get_tokenizer_from_config, Tokenizer
from gnn_lib.modules import heads, embedding, utils
from gnn_lib.modules.embedding import EmbeddingConfig
from gnn_lib.modules.utils import EncoderMixin, DecoderMixin, pad


class Models(enum.IntEnum):
    MODEL_FOR_GRAPH_CLASSIFICATION = 1
    MODEL_FOR_MULTI_NODE_CLASSIFICATION = 2
    MODEL_FOR_GRAPH2SEQ = 3
    MODEL_FOR_MULTI_NODE2SEQ = 4


class GNNs(enum.IntEnum):
    ATTENTION_GNN = 1
    SIMPLE_GNN = 2
    TRANSFORMER_ENCODER_GNN = 3
    TRANSFORMER_ENCODER = 4
    HIERARCHICAL_GNN = 5
    MESSAGE_PASSING_GNN = 6
    IDENTITY_GNN = 7
    GENERAL_GNN = 8


@dataclass
class GNNConfig:
    type: GNNs  # = MISSING


class GNN(nn.Module):
    def __init__(self,
                 cfg: GNNConfig,
                 node_hidden_dim: int,
                 hidden_feature: str,
                 edge_hidden_dim: Optional[int] = None) -> None:
        super().__init__()
        self.cfg = cfg
        self.node_hidden_dim = node_hidden_dim
        self.edge_hidden_dim = edge_hidden_dim
        self.hidden_feature = hidden_feature

    @property
    def name(self) -> str:
        return self.cfg.type.name

    def get_additional_losses(self) -> Dict[str, torch.Tensor]:
        return {}

    def forward(self, g: dgl.DGLHeteroGraph) -> dgl.DGLHeteroGraph:
        raise NotImplementedError


@dataclass
class IdentityGNNConfig(GNNConfig):
    type: GNNs = GNNs.IDENTITY_GNN


class IdentityGNN(GNN):
    def __init__(self,
                 node_hidden_dim: int,
                 hidden_feature: str,
                 cfg: IdentityGNNConfig,
                 sample_g: dgl.DGLHeteroGraph,
                 edge_hidden_dim: Optional[int] = None) -> None:
        super().__init__(cfg, node_hidden_dim, hidden_feature)

    def forward(self, g: dgl.DGLHeteroGraph) -> dgl.DGLHeteroGraph:
        return g


def get_gnn_from_config(
        cfg: omegaconf.DictConfig,
        sample_g: dgl.DGLHeteroGraph,
        node_hidden_dim: int,
        edge_hidden_dim: int,
        hidden_feature: str) -> GNN:
    from gnn_lib.models.attention_gnn import AttentionGNNConfig, AttentionGNN
    from gnn_lib.models.simple_gnn import SimpleGNNConfig, SimpleGNN
    from gnn_lib.models.transformer_gnn import (
        TransformerEncoderGNN,
        TransformerEncoderGNNConfig,
        TransformerEncoder,
        TransformerEncoderConfig
    )
    from gnn_lib.models.hierarchical_gnn import HierarchicalGNN, HierarchicalGNNConfig
    from gnn_lib.models.message_passing_gnn import MessagePassingGNN, MessagePassingGNNConfig
    from gnn_lib.models.general_gnn import GeneralGNN, GeneralGNNConfig

    kwargs = {
        "node_hidden_dim": node_hidden_dim,
        "edge_hidden_dim": edge_hidden_dim,
        "hidden_feature": hidden_feature,
        "sample_g": sample_g
    }

    gnn_type = GNNs[cfg.type]
    if gnn_type == GNNs.ATTENTION_GNN:
        kwargs["cfg"] = omegaconf.OmegaConf.structured(AttentionGNNConfig(**cfg))
        return AttentionGNN(**kwargs)
    elif gnn_type == GNNs.SIMPLE_GNN:
        kwargs["cfg"] = omegaconf.OmegaConf.structured(SimpleGNNConfig(**cfg))
        return SimpleGNN(**kwargs)
    elif gnn_type == GNNs.TRANSFORMER_ENCODER_GNN:
        kwargs["cfg"] = omegaconf.OmegaConf.structured(TransformerEncoderGNNConfig(**cfg))
        return TransformerEncoderGNN(**kwargs)
    elif gnn_type == GNNs.TRANSFORMER_ENCODER:
        kwargs["cfg"] = omegaconf.OmegaConf.structured(TransformerEncoderConfig(**cfg))
        return TransformerEncoder(**kwargs)
    elif gnn_type == GNNs.HIERARCHICAL_GNN:
        kwargs["cfg"] = omegaconf.OmegaConf.structured(HierarchicalGNNConfig(**cfg))
        return HierarchicalGNN(**kwargs)
    elif gnn_type == GNNs.MESSAGE_PASSING_GNN:
        kwargs["cfg"] = omegaconf.OmegaConf.structured(MessagePassingGNNConfig(**cfg))
        return MessagePassingGNN(**kwargs)
    elif gnn_type == GNNs.IDENTITY_GNN:
        kwargs["cfg"] = omegaconf.OmegaConf.structured(IdentityGNNConfig(**cfg))
        return IdentityGNN(**kwargs)
    elif gnn_type == GNNs.GENERAL_GNN:
        kwargs["cfg"] = omegaconf.OmegaConf.structured(GeneralGNNConfig(**cfg))
        return GeneralGNN(**kwargs)
    else:
        raise ValueError(f"Unknown gnn type {cfg.type}")


@dataclass
class ModelConfig:
    type: Models  # = MISSING

    node_hidden_dim: int = MISSING
    edge_hidden_dim: Optional[int] = None
    hidden_feature: str = "h"

    embedding: EmbeddingConfig = MISSING
    gnn: Any = MISSING


class Model(nn.Module):
    def __init__(self,
                 sample_g: dgl.DGLHeteroGraph,
                 cfg: ModelConfig,
                 device: torch.device) -> None:
        super().__init__()
        self.cfg = cfg
        self.device = device

        self.embedding = self.build_embedding(sample_g)
        self.gnn = self.build_gnn(sample_g)
        self.head = self.build_head(sample_g)

    def reset_parameters(self, module: Optional[nn.Module] = None) -> None:
        if module is None:
            module = self
        for p in module.parameters():
            if p.ndim > 1:
                init.xavier_uniform_(p)

    def build_embedding(self, sample_g: dgl.DGLHeteroGraph) -> embedding.Embedding:
        # embedding is also sometimes model specific, because of the use of tokenizers, so we override this
        # in the models
        raise NotImplementedError

    def build_gnn(self, sample_g: dgl.DGLHeteroGraph) -> GNN:
        return get_gnn_from_config(
            self.cfg.gnn,
            sample_g,
            self.cfg.node_hidden_dim,
            self.cfg.edge_hidden_dim,
            self.cfg.hidden_feature
        )

    def build_head(self, sample_g: dgl.DGLHeteroGraph) -> heads.Head:
        # head is mostly model specific, so we override this in the models
        raise NotImplementedError

    def forward(self,
                g: dgl.DGLHeteroGraph,
                **kwargs: Any) -> \
            Tuple[Any, Dict[str, torch.Tensor]]:
        g = g.to(self.device)

        g = self.embedding(g)
        g = self.gnn(g)
        output = self.head(g, **kwargs)

        return output, self.gnn.get_additional_losses()


def get_model_from_config(
        cfg: omegaconf.DictConfig,
        sample_g: dgl.DGLHeteroGraph,
        device: torch.device) -> Model:
    model_type = Models[cfg.type]
    if model_type == Models.MODEL_FOR_GRAPH_CLASSIFICATION:
        cfg = omegaconf.OmegaConf.structured(ModelForGraphClassificationConfig(**cfg))
        return ModelForGraphClassification(sample_g, cfg, device)
    elif model_type == Models.MODEL_FOR_MULTI_NODE_CLASSIFICATION:
        cfg = omegaconf.OmegaConf.structured(ModelForMultiNodeClassificationConfig(**cfg))
        return ModelForMultiNodeClassification(sample_g, cfg, device)
    elif model_type == Models.MODEL_FOR_GRAPH2SEQ:
        cfg = omegaconf.OmegaConf.structured(ModelForGraph2SeqConfig(**cfg))
        return ModelForGraph2Seq(sample_g, cfg, device)
    elif model_type == Models.MODEL_FOR_MULTI_NODE2SEQ:
        cfg = omegaconf.OmegaConf.structured(ModelForMultiNode2SeqConfig(**cfg))
        return ModelForMultiNode2Seq(sample_g, cfg, device)
    else:
        raise ValueError(f"Unknown model type {model_type}")


@dataclass
class ModelForGraphClassificationConfig(ModelConfig):
    type: Models = Models.MODEL_FOR_GRAPH_CLASSIFICATION
    num_classes: int = MISSING
    node_type: str = MISSING
    pooling_type: str = "mean"
    tokenizers: Dict[str, TokenizerConfig] = MISSING


class ModelForGraphClassification(Model):
    def __init__(self,
                 sample_g: dgl.DGLHeteroGraph,
                 cfg: ModelForGraphClassificationConfig,
                 device: torch.device) -> None:
        self.tokenizers = {k: get_tokenizer_from_config(cfg) for k, cfg in cfg.tokenizers.items()}
        super().__init__(sample_g, cfg, device)

    def build_embedding(self, sample_g: dgl.DGLHeteroGraph) -> embedding.Embedding:
        num_token_embeddings = {
            node_type: self.tokenizers[node_type].vocab_size
            for node_type in sample_g.ntypes
            if f"{node_type}_id" in sample_g.node_attr_schemes(node_type)
        }
        return embedding.get_embedding_from_config(
            self.cfg.embedding,
            sample_g,
            self.cfg.node_hidden_dim,
            self.cfg.hidden_feature,
            num_token_embeddings,
            self.cfg.edge_hidden_dim
        )

    def build_head(self, sample_g: dgl.DGLHeteroGraph) -> heads.GraphClassificationHead:
        self.cfg: ModelForGraphClassificationConfig
        return heads.GraphClassificationHead(
            feat=self.cfg.hidden_feature,
            num_features=self.cfg.node_hidden_dim,
            num_classes=self.cfg.num_classes,
            node_type=self.cfg.node_type,
            pooling_type=self.cfg.pooling_type
        )


@dataclass
class ModelForMultiNodeClassificationConfig(ModelConfig):
    type: Models = Models.MODEL_FOR_MULTI_NODE_CLASSIFICATION
    num_classes: Dict[str, int] = MISSING
    tokenizers: Dict[str, TokenizerConfig] = MISSING
    group_nodes: Optional[List[str]] = None


class ModelForMultiNodeClassification(Model):
    def __init__(self,
                 sample_g: dgl.DGLHeteroGraph,
                 cfg: ModelForMultiNodeClassificationConfig,
                 device: torch.device) -> None:
        self.tokenizers: Dict[str, Tokenizer] = {k: get_tokenizer_from_config(cfg) for k, cfg in cfg.tokenizers.items()}
        super().__init__(sample_g, cfg, device)

    def build_embedding(self, sample_g: dgl.DGLHeteroGraph) -> embedding.Embedding:
        num_token_embeddings = {
            node_type: self.tokenizers[node_type].vocab_size
            for node_type in sample_g.ntypes
            if f"{node_type}_id" in sample_g.node_attr_schemes(node_type)
        }
        return embedding.get_embedding_from_config(
            self.cfg.embedding,
            sample_g,
            self.cfg.node_hidden_dim,
            self.cfg.hidden_feature,
            num_token_embeddings,
            self.cfg.edge_hidden_dim
        )

    def build_head(self, sample_g: dgl.DGLHeteroGraph) -> heads.MultiNodeClassificationGroupHead:
        self.cfg: ModelForMultiNodeClassificationConfig
        return heads.MultiNodeClassificationGroupHead(
            feat=self.cfg.hidden_feature,
            num_features=self.cfg.node_hidden_dim,
            num_classes=self.cfg.num_classes,
            group_nodes=set(self.cfg.group_nodes) if self.cfg.group_nodes is not None else None,
            group_feature="group"
        )


@dataclass
class ModelForGraph2SeqConfig(ModelConfig):
    type: Models = Models.MODEL_FOR_GRAPH2SEQ
    context_node_types: List[str] = MISSING
    tokenizers: Dict[str, TokenizerConfig] = MISSING
    decoder_dropout: float = 0.2
    decoder_num_layers: int = MISSING
    decoder_share_parameters: bool = False


class ModelForGraph2Seq(Model, EncoderMixin, DecoderMixin):
    def __init__(self, sample_g: dgl.DGLHeteroGraph, cfg: ModelForGraph2SeqConfig, device: torch.device) -> None:
        self.tokenizers: Dict[str, Tokenizer] = {
            k: get_tokenizer_from_config(cfg) for k, cfg in cfg.tokenizers.items()
        }

        self.bos_token_id = self.tokenizers["output_tokenizer"].token_to_id(tokenization.BOS)
        self.eos_token_id = self.tokenizers["output_tokenizer"].token_to_id(tokenization.EOS)
        self.pad_token_id = self.tokenizers["output_tokenizer"].token_to_id(tokenization.PAD)
        self.num_outputs = self.tokenizers["output_tokenizer"].vocab_size

        super().__init__(sample_g, cfg, device)

    def build_embedding(self, sample_g: dgl.DGLHeteroGraph) -> embedding.Embedding:
        num_token_embeddings = {
            node_type: self.tokenizers[node_type].vocab_size
            for node_type in sample_g.ntypes
            if f"{node_type}_id" in sample_g.node_attr_schemes(node_type)
        }
        return embedding.get_embedding_from_config(
            self.cfg.embedding,
            sample_g,
            self.cfg.node_hidden_dim,
            self.cfg.hidden_feature,
            num_token_embeddings,
            self.cfg.edge_hidden_dim
        )

    def build_head(self, sample_g: dgl.DGLHeteroGraph) -> utils.DecoderMixin:
        self.cfg: ModelForGraph2SeqConfig
        return heads.TransformerDecoderHead(
            context_node_types=self.cfg.context_node_types,
            bos_token_id=self.bos_token_id,
            eos_token_id=self.eos_token_id,
            pad_token_id=self.pad_token_id,
            max_length=512,
            hidden_feature=self.cfg.hidden_feature,
            hidden_dim=self.cfg.node_hidden_dim,
            num_outputs=self.num_outputs,
            dropout=self.cfg.decoder_dropout,
            num_layers=self.cfg.decoder_num_layers,
            share_parameters=self.cfg.decoder_share_parameters
        )

    def forward(self, g: dgl.DGLHeteroGraph, **kwargs: Any) -> \
            Tuple[Any, Dict[str, torch.Tensor]]:
        g = self.encode(g)

        encoder_outputs, encoder_lengths = utils.graph2seq_encoder_outputs_from_graph(
            g, self.cfg.context_node_types, self.cfg.hidden_feature
        )

        assert "decoder_inputs" in kwargs and "decoder_lengths" in kwargs
        output = self.decode(
            decoder_inputs=kwargs["decoder_inputs"],
            decoder_lengths=kwargs["decoder_lengths"],
            encoder_outputs=encoder_outputs,
            encoder_lengths=encoder_lengths,
        )
        return output, self.gnn.get_additional_losses()

    def encode(self, g: dgl.DGLHeteroGraph, **kwargs: Any) -> dgl.DGLHeteroGraph:
        g = g.to(self.device)
        g = self.embedding(g)
        return self.gnn(g)

    def decode(self,
               decoder_inputs: torch.Tensor,
               **kwargs: Any) -> torch.Tensor:
        return self.head.decode(decoder_inputs, **kwargs)


@dataclass
class ModelForMultiNode2SeqConfig(ModelConfig):
    type: Models = Models.MODEL_FOR_MULTI_NODE2SEQ
    context_node_types: Dict[str, List[str]] = MISSING
    align_positions_with: Optional[Dict[str, str]] = None
    tokenizers: Dict[str, TokenizerConfig] = MISSING
    decoder_dropout: float = 0.2
    decoder_num_layers: int = MISSING
    decoder_share_parameters: bool = False
    decoder_node_types: Optional[List[str]] = None


class ModelForMultiNode2Seq(Model, EncoderMixin):
    def __init__(self, sample_g: dgl.DGLHeteroGraph, cfg: ModelForMultiNode2SeqConfig, device: torch.device) -> None:
        self.tokenizers: Dict[str, Tokenizer] = {
            k: get_tokenizer_from_config(cfg) for k, cfg in cfg.tokenizers.items()
        }

        self.bos_token_ids = {
            node_type: self.tokenizers[f"{node_type}_output_tokenizer"].token_to_id(tokenization.BOS)
            for node_type in cfg.decoder_node_types
        }
        self.eos_token_ids = {
            node_type: self.tokenizers[f"{node_type}_output_tokenizer"].token_to_id(tokenization.EOS)
            for node_type in cfg.decoder_node_types
        }
        self.pad_token_ids = {
            node_type: self.tokenizers[f"{node_type}_output_tokenizer"].token_to_id(tokenization.PAD)
            for node_type in cfg.decoder_node_types
        }
        self.num_outputs = {
            node_type: self.tokenizers[f"{node_type}_output_tokenizer"].vocab_size
            for node_type in cfg.decoder_node_types
        }
        super().__init__(sample_g, cfg, device)

    def build_embedding(self, sample_g: dgl.DGLHeteroGraph) -> embedding.Embedding:
        num_token_embeddings = {
            node_type: self.tokenizers[node_type].vocab_size
            for node_type in sample_g.ntypes
            if f"{node_type}_id" in sample_g.node_attr_schemes(node_type)
        }
        return embedding.get_embedding_from_config(
            self.cfg.embedding,
            sample_g,
            self.cfg.node_hidden_dim,
            self.cfg.hidden_feature,
            num_token_embeddings,
            self.cfg.edge_hidden_dim
        )

    def build_head(self, sample_g: dgl.DGLHeteroGraph) -> nn.ModuleDict:
        self.cfg: ModelForMultiNode2SeqConfig
        node2seq_head = nn.ModuleDict()
        for node_type in self.cfg.decoder_node_types:
            node2seq_head[node_type] = heads.TransformerDecoderHead(
                context_node_types=self.cfg.context_node_types[node_type] + [node_type],
                bos_token_id=self.bos_token_ids[node_type],
                eos_token_id=self.eos_token_ids[node_type],
                pad_token_id=self.pad_token_ids[node_type],
                max_length=512,
                hidden_feature=self.cfg.hidden_feature,
                hidden_dim=self.cfg.node_hidden_dim,
                num_outputs=self.num_outputs[node_type],
                dropout=self.cfg.decoder_dropout,
                num_layers=self.cfg.decoder_num_layers,
                share_parameters=self.cfg.decoder_share_parameters
            )
        return node2seq_head

    def encode(self, g: dgl.DGLHeteroGraph, **kwargs: Any) -> dgl.DGLHeteroGraph:
        g = g.to(self.device)
        g = self.embedding(g)
        return self.gnn(g)

    def forward(self, g: dgl.DGLHeteroGraph, **kwargs: Any) -> Tuple[Any, Dict[str, torch.Tensor]]:
        self.cfg: ModelForMultiNode2SeqConfig
        g = self.encode(g)

        assert "decoder_inputs" in kwargs and "decoder_lengths" in kwargs
        outputs = {}
        for node_type in self.cfg.decoder_node_types:
            # start = time.perf_counter()
            (
                encoder_outputs, encoder_lengths, aligned_encoder_positions
            ) = utils.multi_node2seq_encoder_outputs_from_graph2(
                g,
                node_type,
                self.cfg.context_node_types[node_type],
                self.cfg.hidden_feature,
                self.cfg.align_positions_with[node_type] if self.cfg.align_positions_with is not None else None
            )
            # end = time.perf_counter()
            # print(f"encoder outputs from graph took {(end - start) * 1000:.2f}ms")

            decoder_lengths = kwargs["decoder_lengths"][node_type]
            decoder_inputs = kwargs["decoder_inputs"][node_type]

            # create decoder positions and encoder masks
            if aligned_encoder_positions is not None:
                assert sum(len(pos) for pos in aligned_encoder_positions) == \
                       sum(len(lengths) for lengths in decoder_lengths)
                decoder_positions = []
                for encoder_positions, lengths in zip(aligned_encoder_positions, decoder_lengths):
                    lengths = lengths.cpu().numpy()
                    encoder_positions = encoder_positions.cpu().numpy()
                    encoder_positions_plus_lengths = encoder_positions + lengths
                    upper_indices = np.cumsum(lengths, 0)
                    lower_indices = np.concatenate([[0], upper_indices[:-1]])

                    positions = np.zeros((upper_indices[-1],), dtype=int)
                    for idx_lower, idx_upper, range_lower, range_upper in zip(
                            lower_indices,
                            upper_indices,
                            encoder_positions,
                            encoder_positions_plus_lengths
                    ):
                        positions[idx_lower:idx_upper] = np.arange(range_lower, range_upper)
                    decoder_positions.append(torch.from_numpy(positions))
                decoder_positions = pad(decoder_positions).to(g.device)
            else:
                decoder_positions = None

            # end_pos = time.perf_counter()
            # print(f"aligning with encoder positions took {1000 * (end_pos - end):.2f}ms")

            # bring encoder outputs into correct format
            encoder_outputs = {
                enc_node_type: pad([
                    torch.cat(enc_output)
                    for enc_output in enc_outputs
                ])
                for enc_node_type, enc_outputs in encoder_outputs.items()
            }

            # end_enc = time.perf_counter()
            # print(f"padding and cating enc outputs took {1000 * (end_enc - end_pos):.2f}ms")

            decoder_mask = torch.stack([
                utils.square_causal_block_mask(
                    decoder_inputs.shape[1],
                    dec_lengths.cpu().numpy(),
                    device=g.device
                ) for dec_lengths in decoder_lengths
            ])

            # end_mask_dec = time.perf_counter()
            # print(f"creating decoder causal masks took {(end_mask_dec - end_enc) * 1000:.2f}ms")

            encoder_masks = {}
            for enc_node_type, enc_lengths in encoder_lengths.items():
                encoder_masks[enc_node_type] = torch.stack([
                    utils.rectangular_block_mask(
                        decoder_inputs.shape[1],
                        encoder_outputs[enc_node_type].shape[1],
                        dec_lengths.cpu().numpy(),
                        lengths.cpu().numpy(),
                        device=g.device
                    ) for lengths, dec_lengths in
                    zip(enc_lengths, decoder_lengths)
                ])

            # end_mask_mem = time.perf_counter()
            # print(f"creating memory masks took {(end_mask_mem - end_mask_dec) * 1000:.2f}ms")

            # sum up decoder and encoder lengths
            decoder_lengths = torch.stack([lengths.sum() for lengths in decoder_lengths])

            encoder_lengths = {
                enc_node_type: torch.stack([lengths.sum() for lengths in enc_lengths])
                for enc_node_type, enc_lengths in encoder_lengths.items()
            }

            # end_lens = time.perf_counter()
            # print(f"preparing lengths took {(end_lens - end_mask_mem) * 1000:.2f}ms")

            # end2 = time.perf_counter()
            # print(f"creating masks took {(end2 - end_enc) * 1000:.2f}ms")
            # print(f"total took {(end2 - start) * 1000:.2f}ms")

            outputs[node_type] = self.head[node_type].decode(
                decoder_inputs=decoder_inputs,
                decoder_lengths=decoder_lengths,
                encoder_outputs=encoder_outputs,
                encoder_lengths=encoder_lengths,
                encoder_masks=encoder_masks,
                decoder_mask=decoder_mask,
                decoder_positions=decoder_positions
            )

        return outputs, self.gnn.get_additional_losses()
