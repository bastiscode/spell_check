import enum
from dataclasses import dataclass
from typing import Optional, Dict, Tuple, List, Any, Union

import dgl
import einops
import omegaconf
import torch
from omegaconf import MISSING
from torch import nn

from nsc.data import tokenization
from nsc.data.tokenization import TokenizerConfig, get_tokenizer_from_config, Tokenizer
from nsc.modules import heads, embedding, utils, encoders
from nsc.modules.embedding import GraphEmbeddingConfig, TensorEmbeddingConfig
from nsc.modules.utils import GraphEncoderMixin, TensorEncoderMixin, DecoderMixin, pad
from nsc.utils import TensorInput, Batch, DataInput, to, io, hooks


class Models(enum.IntEnum):
    # graph based models (Graph neural networks)
    MODEL_FOR_GRAPH_CLASSIFICATION = 1
    MODEL_FOR_MULTI_NODE_CLASSIFICATION = 2
    MODEL_FOR_GRAPH2SEQ = 3
    MODEL_FOR_MULTI_NODE2SEQ = 4
    # tensor based models (Standard architectures like CNN, LSTM and Transformer)
    MODEL_FOR_TOKEN_CLASSIFICATION = 5  # analog to multi node classification
    MODEL_FOR_SEQ2SEQ = 6  # analog to graph2seq
    MODEL_FOR_TOKEN2SEQ = 7  # analog to multi node2seq
    MODEL_FOR_SEQUENCE_CLASSIFICATION = 8  # analog to graph classification

    MODEL_FOR_TOKENIZATION_REPAIR_PLUS = 9


class GNNs(enum.IntEnum):
    ATTENTION_GNN = 1
    SIMPLE_GNN = 2
    TRANSFORMER_ENCODER_GNN = 3
    TRANSFORMER_ENCODER = 4
    MESSAGE_PASSING_GNN = 5
    IDENTITY_GNN = 6
    GENERAL_GNN = 7


@dataclass
class GNNConfig:
    type: GNNs


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
        cfg: Union[GNNConfig, omegaconf.DictConfig],
        sample_g: dgl.DGLHeteroGraph,
        node_hidden_dim: int,
        edge_hidden_dim: int,
        hidden_feature: str
) -> GNN:
    from nsc.models.attention_gnn import AttentionGNNConfig, AttentionGNN
    from nsc.models.simple_gnn import SimpleGNNConfig, SimpleGNN
    from nsc.models.transformer_gnn import (
        TransformerEncoderGNN,
        TransformerEncoderGNNConfig,
        TransformerEncoder,
        TransformerEncoderConfig
    )
    from nsc.models.message_passing_gnn import MessagePassingGNN, MessagePassingGNNConfig
    from nsc.models.general_gnn import GeneralGNN, GeneralGNNConfig

    kwargs = {
        "node_hidden_dim": node_hidden_dim,
        "edge_hidden_dim": edge_hidden_dim,
        "hidden_feature": hidden_feature,
        "sample_g": sample_g
    }

    # explicitly convert ot dict config first, this way we support both dictconfigs
    # and structured configs as input
    cfg: omegaconf.DictConfig = omegaconf.DictConfig(cfg)
    gnn_type = GNNs[cfg.type] if isinstance(cfg.type, str) else cfg.type
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
    type: Models

    max_length: int = 512


class Model(nn.Module):
    def __init__(
            self,
            cfg: ModelConfig,
            device: torch.device
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.device = device

    @staticmethod
    def _check_input(data: DataInput) -> None:
        raise NotImplementedError


@dataclass
class GraphModelConfig(ModelConfig):
    node_hidden_dim: int = MISSING
    edge_hidden_dim: Optional[int] = None
    hidden_feature: str = "h"

    embedding: GraphEmbeddingConfig = MISSING
    gnn: GNNConfig = MISSING


class GraphModel(Model):
    def __init__(self, sample_inputs: Batch, cfg: GraphModelConfig, device: torch.device) -> None:
        super().__init__(cfg, device)
        self._check_input(sample_inputs.data)

        self.embedding = self.build_embedding(sample_inputs.data)
        self.gnn = self.build_gnn(sample_inputs.data)
        self.head = self.build_head(sample_inputs)

    @staticmethod
    def _check_input(data: DataInput) -> None:
        if not isinstance(data, dgl.DGLHeteroGraph):
            raise RuntimeError(
                f"expected data input to be a dgl heterograph instance for a graph model, but got {type(data)}"
            )

    def build_embedding(self, sample_g: DataInput) -> embedding.GraphEmbedding:
        # embedding is also sometimes model specific, because of the use of tokenizers, so we override this
        # in the models
        raise NotImplementedError

    def build_gnn(self, sample_g: DataInput) -> GNN:
        self.cfg: GraphModelConfig
        return get_gnn_from_config(
            self.cfg.gnn,
            sample_g,
            self.cfg.node_hidden_dim,
            self.cfg.edge_hidden_dim,
            self.cfg.hidden_feature
        )

    def build_head(self, sample_inputs: Batch) -> nn.Module:
        # head is mostly model specific, so we override this in the models
        raise NotImplementedError

    def forward(self,
                g: DataInput,
                **kwargs: Any) -> \
            Tuple[Any, Dict[str, torch.Tensor]]:
        g = to(g, self.device)

        g = self.embedding(g)
        g = self.gnn(g)
        output = self.head(g, **kwargs)

        return output, self.gnn.get_additional_losses()


@dataclass
class TensorModelConfig(ModelConfig):
    hidden_dim: int = MISSING


class TensorModel(Model):
    def __init__(self, sample_inputs: Batch, cfg: TensorModelConfig, device: torch.device) -> None:
        super().__init__(cfg, device)
        self._check_input(sample_inputs.data)

        self.embedding = self.build_embedding(sample_inputs.data)
        self.encoder = self.build_encoder(sample_inputs.data)
        self.head = self.build_head(sample_inputs)

    @staticmethod
    def _check_input(data: DataInput) -> None:
        if (
                # check for non empty tensor list
                isinstance(data, list)
                and all(isinstance(d, torch.Tensor) for d in data)
        ):
            return
        raise RuntimeError(
            f"expected data input to be a non empty tensor list, but got {type(data)}"
        )

    def build_embedding(self, sample_input: DataInput) -> embedding.TensorEmbedding:
        # embedding is also sometimes model specific, because of the use of tokenizers, so we override this
        # in the models
        raise NotImplementedError

    def build_encoder(self, sample_input: DataInput) -> nn.Module:
        raise NotImplementedError

    def build_head(self, sample_input: Batch) -> nn.Module:
        # head is mostly model specific, so we override this in the models
        raise NotImplementedError

    def forward(self,
                x: TensorInput,
                **kwargs: Any) -> \
            Tuple[Any, Dict[str, torch.Tensor]]:
        raise NotImplementedError


def get_model_from_config(
        cfg: Union[ModelConfig, omegaconf.DictConfig],
        sample_inputs: Batch,
        device: torch.device
) -> Model:
    # explicitly convert ot dict config first, this way we support both dictconfigs
    # and structured configs as input
    cfg: omegaconf.DictConfig = omegaconf.DictConfig(cfg)
    model_type = Models[cfg.type] if isinstance(cfg.type, str) else cfg.type
    if model_type == Models.MODEL_FOR_GRAPH_CLASSIFICATION:
        cfg = omegaconf.OmegaConf.structured(ModelForGraphClassificationConfig(**cfg))
        return ModelForGraphClassification(sample_inputs, cfg, device)
    elif model_type == Models.MODEL_FOR_MULTI_NODE_CLASSIFICATION:
        cfg = omegaconf.OmegaConf.structured(ModelForMultiNodeClassificationConfig(**cfg))
        return ModelForMultiNodeClassification(sample_inputs, cfg, device)
    elif model_type == Models.MODEL_FOR_GRAPH2SEQ:
        cfg = omegaconf.OmegaConf.structured(ModelForGraph2SeqConfig(**cfg))
        return ModelForGraph2Seq(sample_inputs, cfg, device)
    elif model_type == Models.MODEL_FOR_MULTI_NODE2SEQ:
        cfg = omegaconf.OmegaConf.structured(ModelForMultiNode2SeqConfig(**cfg))
        return ModelForMultiNode2Seq(sample_inputs, cfg, device)
    elif model_type == Models.MODEL_FOR_SEQ2SEQ:
        cfg = omegaconf.OmegaConf.structured(ModelForSeq2SeqConfig(**cfg))
        return ModelForSeq2Seq(sample_inputs, cfg, device)
    elif model_type == Models.MODEL_FOR_TOKEN2SEQ:
        cfg = omegaconf.OmegaConf.structured(ModelForToken2SeqConfig(**cfg))
        return ModelForToken2Seq(sample_inputs, cfg, device)
    elif model_type == Models.MODEL_FOR_TOKEN_CLASSIFICATION:
        cfg = omegaconf.OmegaConf.structured(ModelForTokenClassificationConfig(**cfg))
        return ModelForTokenClassification(sample_inputs, cfg, device)
    elif model_type == Models.MODEL_FOR_SEQUENCE_CLASSIFICATION:
        cfg = omegaconf.OmegaConf.structured(ModelForSequenceClassificationConfig(**cfg))
        return ModelForSequenceClassification(sample_inputs, cfg, device)
    elif model_type == Models.MODEL_FOR_TOKENIZATION_REPAIR_PLUS:
        cfg = omegaconf.OmegaConf.structured(ModelForTokenizationRepairPlusConfig(**cfg))
        return ModelForTokenizationRepairPlus(sample_inputs, cfg, device)
    else:
        raise ValueError(f"Unknown model type {model_type}")


@dataclass
class ModelForGraphClassificationConfig(GraphModelConfig):
    type: Models = Models.MODEL_FOR_GRAPH_CLASSIFICATION
    num_classes: int = MISSING
    node_type: str = MISSING
    pooling_type: str = "mean"
    tokenizers: Dict[str, TokenizerConfig] = MISSING


class ModelForGraphClassification(GraphModel):
    def __init__(self,
                 sample_inputs: Batch,
                 cfg: ModelForGraphClassificationConfig,
                 device: torch.device) -> None:
        self.tokenizers = {k: get_tokenizer_from_config(cfg) for k, cfg in cfg.tokenizers.items()}
        super().__init__(sample_inputs, cfg, device)

    def build_embedding(self, sample_g: dgl.DGLHeteroGraph) -> embedding.GraphEmbedding:
        embeddings = {
            node_type: (
                self.tokenizers[node_type].vocab_size,
                self.tokenizers[node_type].token_to_id(tokenization.PAD)
            )
            for node_type in sample_g.ntypes
            if f"{node_type}_id" in sample_g.node_attr_schemes(node_type)
        }
        self.cfg: GraphModelConfig
        return embedding.get_embedding_from_config(
            self.cfg.embedding,
            sample_g,
            node_hidden_dim=self.cfg.node_hidden_dim,
            hidden_feature=self.cfg.hidden_feature,
            embeddings=embeddings,
            max_length=self.cfg.max_length,
            edge_hidden_dim=self.cfg.edge_hidden_dim
        )

    def build_head(self, sample_inputs: Batch) -> heads.GraphClassificationHead:
        self.cfg: ModelForGraphClassificationConfig
        return heads.GraphClassificationHead(
            feat=self.cfg.hidden_feature,
            num_features=self.cfg.node_hidden_dim,
            num_classes=self.cfg.num_classes,
            node_type=self.cfg.node_type,
            pooling_type=self.cfg.pooling_type
        )


@dataclass
class ModelForMultiNodeClassificationConfig(GraphModelConfig):
    type: Models = Models.MODEL_FOR_MULTI_NODE_CLASSIFICATION
    num_classes: Dict[str, int] = MISSING
    tokenizers: Dict[str, TokenizerConfig] = MISSING
    group_nodes: Optional[List[str]] = None


class ModelForMultiNodeClassification(GraphModel):
    def __init__(self,
                 sample_inputs: Batch,
                 cfg: ModelForMultiNodeClassificationConfig,
                 device: torch.device) -> None:
        self.tokenizers: Dict[str, Tokenizer] = {k: get_tokenizer_from_config(cfg) for k, cfg in cfg.tokenizers.items()}
        super().__init__(sample_inputs, cfg, device)

    def build_embedding(self, sample_g: DataInput) -> embedding.GraphEmbedding:
        embeddings = {
            node_type: (
                self.tokenizers[node_type].vocab_size,
                self.tokenizers[node_type].token_to_id(tokenization.PAD)
            )
            for node_type in sample_g.ntypes
            if f"{node_type}_id" in sample_g.node_attr_schemes(node_type)
        }
        self.cfg: GraphModelConfig
        return embedding.get_embedding_from_config(
            self.cfg.embedding,
            sample_g,
            node_hidden_dim=self.cfg.node_hidden_dim,
            hidden_feature=self.cfg.hidden_feature,
            embeddings=embeddings,
            max_length=self.cfg.max_length,
            edge_hidden_dim=self.cfg.edge_hidden_dim
        )

    def build_head(self, sample_input: Batch) -> heads.MultiNodeClassificationGroupHead:
        self.cfg: ModelForMultiNodeClassificationConfig

        if "groups" in sample_input.info:
            additional_features = {}
            aggregation = {}
            for node_type, stages in sample_input.info["groups"][0].items():
                (
                    additional_features[node_type],
                    aggregation[node_type]
                ) = utils.get_additional_features_and_aggregations_from_group_stages(
                    stages=stages
                )
        else:
            additional_features = aggregation = None

        return heads.MultiNodeClassificationGroupHead(
            feat=self.cfg.hidden_feature,
            num_features=self.cfg.node_hidden_dim,
            num_classes=self.cfg.num_classes,
            num_additional_features=additional_features,
            aggregation=aggregation
        )


@dataclass
class ModelForGraph2SeqConfig(GraphModelConfig):
    type: Models = Models.MODEL_FOR_GRAPH2SEQ

    context_node_types: List[str] = MISSING
    input_tokenizers: Dict[str, TokenizerConfig] = MISSING
    output_tokenizer: TokenizerConfig = MISSING

    max_output_length: int = 512

    decoder_dropout: float = 0.1
    num_decoder_layers: int = MISSING
    decoder_share_parameters: bool = False


class ModelForGraph2Seq(GraphModel, GraphEncoderMixin, DecoderMixin):
    def __init__(self, sample_inputs: Batch, cfg: ModelForGraph2SeqConfig, device: torch.device) -> None:
        self.input_tokenizers: Dict[str, Tokenizer] = {
            k: get_tokenizer_from_config(cfg) for k, cfg in cfg.input_tokenizers.items()
        }

        self.output_tokenizer = get_tokenizer_from_config(cfg.output_tokenizer)
        self.output_pad_token_id = self.output_tokenizer.token_to_id(tokenization.PAD)

        super().__init__(sample_inputs, cfg, device)

    def build_embedding(self, sample_g: dgl.DGLHeteroGraph) -> embedding.GraphEmbedding:
        embeddings = {
            node_type: (
                self.input_tokenizers[node_type].vocab_size,
                self.input_tokenizers[node_type].token_to_id(tokenization.PAD)
            )
            for node_type in sample_g.ntypes
            if f"{node_type}_id" in sample_g.node_attr_schemes(node_type)
        }
        self.cfg: GraphModelConfig
        return embedding.get_embedding_from_config(
            self.cfg.embedding,
            sample_g,
            node_hidden_dim=self.cfg.node_hidden_dim,
            hidden_feature=self.cfg.hidden_feature,
            embeddings=embeddings,
            max_length=self.cfg.max_length,
            edge_hidden_dim=self.cfg.edge_hidden_dim
        )

    def build_head(self, sample_inputs: Batch) -> utils.DecoderMixin:
        self.cfg: ModelForGraph2SeqConfig
        return heads.TransformerDecoderHead(
            contexts=self.cfg.context_node_types,
            pad_token_id=self.output_pad_token_id,
            max_length=self.cfg.max_output_length,
            hidden_dim=self.cfg.node_hidden_dim,
            num_outputs=self.output_tokenizer.vocab_size,
            dropout=self.cfg.decoder_dropout,
            num_layers=self.cfg.num_decoder_layers,
            share_parameters=self.cfg.decoder_share_parameters
        )

    def forward(
            self,
            g: DataInput,
            decoder_inputs: Optional[List[torch.Tensor]] = None,
            **kwargs: Any
    ) -> Tuple[Any, Dict[str, torch.Tensor]]:
        self.cfg: ModelForGraph2SeqConfig
        assert decoder_inputs is not None
        g = self.encode(g)

        encoder_outputs, encoder_lengths = utils.encoder_outputs_from_graph(
            g, self.cfg.context_node_types, self.cfg.hidden_feature
        )

        decoder_lengths = [len(t) for t in decoder_inputs]
        decoder_inputs = to(utils.pad(decoder_inputs, self.output_pad_token_id).long(), self.device)

        output = self.decode(
            decoder_inputs=decoder_inputs,
            decoder_lengths=decoder_lengths,
            encoder_outputs=encoder_outputs,
            encoder_lengths=encoder_lengths,
        )
        return output, self.gnn.get_additional_losses()

    def encode(self, g: dgl.DGLHeteroGraph, **kwargs: Any) -> dgl.DGLHeteroGraph:
        g = to(g, self.device)
        g = self.embedding(g)
        return self.gnn(g)

    def decode(
            self,
            decoder_inputs: torch.Tensor,
            **kwargs: Any
    ) -> torch.Tensor:
        return self.head.decode(decoder_inputs, **kwargs)


@dataclass
class ModelForMultiNode2SeqConfig(GraphModelConfig):
    type: Models = Models.MODEL_FOR_MULTI_NODE2SEQ
    context_node_types: Dict[str, List[str]] = MISSING
    align_positions_with: Optional[Dict[str, str]] = None
    input_tokenizers: Dict[str, TokenizerConfig] = MISSING
    output_tokenizers: Dict[str, TokenizerConfig] = MISSING

    max_output_length: int = 512

    decoder_dropout: float = 0.2
    num_decoder_layers: int = MISSING
    decoder_share_parameters: bool = False
    decoder_node_types: List[str] = MISSING


class ModelForMultiNode2Seq(GraphModel, GraphEncoderMixin):
    def __init__(self, sample_inputs: Batch, cfg: ModelForMultiNode2SeqConfig, device: torch.device) -> None:
        self.input_tokenizers: Dict[str, Tokenizer] = {
            k: get_tokenizer_from_config(cfg) for k, cfg in cfg.input_tokenizers.items()
        }
        self.output_tokenizers: Dict[str, Tokenizer] = {
            k: get_tokenizer_from_config(cfg) for k, cfg in cfg.output_tokenizers.items()
        }

        assert all(node_type in self.output_tokenizers for node_type in cfg.decoder_node_types), \
            "need an output tokenizer for each decoder node type"

        self.output_pad_token_ids = {
            node_type: self.output_tokenizers[node_type].token_to_id(tokenization.PAD)
            for node_type in cfg.decoder_node_types
        }
        self.num_outputs = {
            node_type: self.output_tokenizers[node_type].vocab_size
            for node_type in cfg.decoder_node_types
        }
        super().__init__(sample_inputs, cfg, device)

    def build_embedding(self, sample_g: dgl.DGLHeteroGraph) -> embedding.GraphEmbedding:
        embeddings = {
            node_type: (
                self.input_tokenizers[node_type].vocab_size,
                self.input_tokenizers[node_type].token_to_id(tokenization.PAD)
            )
            for node_type in sample_g.ntypes
            if f"{node_type}_id" in sample_g.node_attr_schemes(node_type)
        }
        self.cfg: GraphModelConfig
        return embedding.get_embedding_from_config(
            self.cfg.embedding,
            sample_g,
            node_hidden_dim=self.cfg.node_hidden_dim,
            hidden_feature=self.cfg.hidden_feature,
            embeddings=embeddings,
            max_length=self.cfg.max_length,
            edge_hidden_dim=self.cfg.edge_hidden_dim
        )

    def build_head(self, sample_inputs: Batch) -> nn.ModuleDict:
        self.cfg: ModelForMultiNode2SeqConfig
        node2seq_head = nn.ModuleDict()
        for node_type in self.cfg.decoder_node_types:
            node2seq_head[node_type] = heads.TransformerDecoderHead(
                contexts=self.cfg.context_node_types[node_type] + [node_type],
                pad_token_id=self.output_pad_token_ids[node_type],
                max_length=self.cfg.max_output_length,
                hidden_dim=self.cfg.node_hidden_dim,
                num_outputs=self.num_outputs[node_type],
                dropout=self.cfg.decoder_dropout,
                num_layers=self.cfg.num_decoder_layers,
                share_parameters=self.cfg.decoder_share_parameters
            )
        return node2seq_head

    def encode(self, g: dgl.DGLHeteroGraph, **kwargs: Any) -> dgl.DGLHeteroGraph:
        g = to(g, self.device)
        g = self.embedding(g)
        return self.gnn(g)

    def forward(
            self,
            g: dgl.DGLHeteroGraph,
            encoder_group_lengths: Optional[List[Dict[str, Dict[str, torch.Tensor]]]] = None,
            decoder_inputs: Optional[Dict[str, List[torch.Tensor]]] = None,
            decoder_group_lengths: Optional[Dict[str, List[torch.Tensor]]] = None,
            **kwargs: Any
    ) -> Tuple[Any, Dict[str, torch.Tensor]]:
        self.cfg: ModelForMultiNode2SeqConfig
        g = self.encode(g)

        assert decoder_inputs is not None and decoder_group_lengths is not None
        decoder_lengths = {
            node_type: [len(t) for t in tensors]
            for node_type, tensors in decoder_inputs.items()
        }
        decoder_inputs: Dict[str, torch.Tensor] = {
            node_type: to(pad(tensors, self.output_pad_token_ids[node_type]).long(), self.device)
            for node_type, tensors in decoder_inputs.items()
        }

        outputs = {}
        for node_type in self.cfg.decoder_node_types:
            node_type_encoder_group_lengths = [group_lengths[node_type] for group_lengths in encoder_group_lengths]
            context_node_types = self.cfg.context_node_types[node_type] + [node_type]
            (
                encoder_outputs,
                encoder_lengths
            ) = utils.encoder_outputs_from_graph(
                g,
                context_node_types,
                self.cfg.hidden_feature
            )

            if self.cfg.align_positions_with is not None:
                align_positions_with = self.cfg.align_positions_with.get(node_type, node_type)
            else:
                align_positions_with = node_type
            assert align_positions_with in context_node_types

            # create decoder masks, decoder positions and encoder masks
            decoder_positions = []
            decoder_masks = []
            encoder_masks = {
                ctx_node_type: []
                for ctx_node_type in self.cfg.context_node_types[node_type] + [node_type]
            }

            for enc_lengths_dict, dec_lengths in zip(
                    node_type_encoder_group_lengths, decoder_group_lengths[node_type]
            ):
                for ctx_node_type in context_node_types:
                    encoder_masks[ctx_node_type].append(
                        utils.rectangular_block_mask(
                            decoder_inputs[node_type].shape[1],
                            encoder_outputs[ctx_node_type].shape[1],
                            dec_lengths,
                            enc_lengths_dict[ctx_node_type],
                            device=self.device
                        )
                    )

                decoder_masks.append(
                    utils.square_causal_block_mask(
                        decoder_inputs[node_type].shape[1],
                        dec_lengths,
                        device=self.device
                    )
                )

                position_enc_lengths = enc_lengths_dict[align_positions_with]
                positions = torch.cat([
                    torch.arange(enc_position, enc_position + dec_length)
                    for enc_position, dec_length
                    in zip(torch.cat([torch.tensor([0]), torch.cumsum(position_enc_lengths, dim=0)[:-1]]), dec_lengths)
                ])
                decoder_positions.append(positions)
                assert len(decoder_positions[-1]) == dec_lengths.sum()

            assert all(
                len(positions) == length for positions, length in zip(decoder_positions, decoder_lengths[node_type])
            )

            outputs[node_type] = self.head[node_type].decode(
                decoder_inputs=decoder_inputs[node_type],
                decoder_lengths=decoder_lengths[node_type],
                encoder_outputs=encoder_outputs,
                encoder_lengths=encoder_lengths,
                encoder_masks={ctx_node_type: torch.stack(masks) for ctx_node_type, masks in encoder_masks.items()},
                decoder_mask=torch.stack(decoder_masks),
                decoder_positions=to(utils.pad(decoder_positions).long(), self.device)
            )

        return outputs, self.gnn.get_additional_losses()


@dataclass
class ModelForToken2SeqConfig(TensorModelConfig):
    type: Models = Models.MODEL_FOR_TOKEN2SEQ
    input_tokenizer: TokenizerConfig = MISSING
    output_tokenizer: TokenizerConfig = MISSING

    max_output_length: int = 512

    embedding: TensorEmbeddingConfig = MISSING
    dropout: float = 0.1
    num_encoder_layers: int = MISSING
    num_decoder_layers: int = MISSING
    use_sequence_context: bool = False


class ModelForToken2Seq(TensorModel, TensorEncoderMixin, DecoderMixin):
    def __init__(self,
                 sample_inputs: Batch,
                 cfg: ModelForToken2SeqConfig,
                 device: torch.device) -> None:
        self.input_tokenizer = get_tokenizer_from_config(cfg.input_tokenizer)
        self.output_tokenizer = get_tokenizer_from_config(cfg.output_tokenizer)

        self.input_pad_token_id = self.input_tokenizer.token_to_id(tokenization.PAD)
        self.output_pad_token_id = self.output_tokenizer.token_to_id(tokenization.PAD)

        super().__init__(sample_inputs, cfg, device)

    def build_embedding(self, sample_input: DataInput) -> embedding.TensorEmbedding:
        self.cfg: ModelForToken2SeqConfig
        return embedding.get_embedding_from_config(
            cfg=self.cfg.embedding,
            sample_inputs=sample_input,
            hidden_dim=self.cfg.hidden_dim,
            num_embeddings=self.input_tokenizer.vocab_size,
            max_length=self.cfg.max_length,
            padding_idx=self.input_pad_token_id
        )

    def build_encoder(self, sample_input: DataInput) -> nn.Module:
        self.cfg: ModelForToken2SeqConfig
        return encoders.get_feature_encoder(
            encoder=encoders.Encoders.TRANSFORMER,
            in_dim=self.cfg.hidden_dim,
            hidden_dim=self.cfg.hidden_dim,
            dropout=self.cfg.dropout,
            num_layers=self.cfg.num_encoder_layers
        )

    def build_head(self, sample_input: Batch) -> heads.TransformerDecoderHead:
        self.cfg: ModelForToken2SeqConfig
        contexts = ["encoder_outputs"]
        if self.cfg.use_sequence_context:
            contexts.append("full_encoder_outputs")
        return heads.TransformerDecoderHead(
            contexts=contexts,
            pad_token_id=self.output_pad_token_id,
            max_length=self.cfg.max_output_length,
            hidden_dim=self.cfg.hidden_dim,
            num_outputs=self.output_tokenizer.vocab_size,
            dropout=self.cfg.dropout,
            num_layers=self.cfg.num_decoder_layers
        )

    def pad_inputs(self, x: DataInput) -> Tuple[torch.Tensor, torch.Tensor]:
        assert all(t.ndim == 1 for t in x)
        encoder_lengths = [len(t) for t in x]
        encoder_inputs = to(utils.pad(x, self.input_pad_token_id).long(), self.device)
        padding_mask = utils.padding_mask(encoder_inputs, encoder_lengths)
        return encoder_inputs, padding_mask

    def forward(
            self,
            x: DataInput,
            encoder_group_lengths: Optional[List[torch.Tensor]] = None,
            decoder_inputs: Optional[List[torch.Tensor]] = None,
            decoder_group_lengths: Optional[List[torch.Tensor]] = None,
            **kwargs: Any
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        self.cfg: ModelForToken2SeqConfig
        assert decoder_inputs is not None

        encoder_inputs, encoder_padding_mask = self.pad_inputs(x)
        enc = self.encode(encoder_inputs, encoder_padding_mask)

        decoder_lengths = [len(t) for t in decoder_inputs]
        decoder_inputs = to(utils.pad(decoder_inputs, self.output_pad_token_id).long(), self.device)

        decoder_masks = []
        encoder_masks = []
        decoder_positions = []
        for enc_lengths, dec_lengths in zip(encoder_group_lengths, decoder_group_lengths):
            decoder_positions.append(
                torch.cat(
                    [
                        torch.arange(enc_position, enc_position + dec_length)
                        for enc_position, dec_length
                        in zip(torch.cat([torch.tensor([0]), torch.cumsum(enc_lengths, dim=0)[:-1]]), dec_lengths)
                    ]
                )
            )
            assert len(decoder_positions[-1]) == dec_lengths.sum()

            decoder_masks.append(
                utils.square_causal_block_mask(
                    decoder_inputs.shape[1],
                    dec_lengths,
                    device=self.device
                )
            )

            encoder_masks.append(
                utils.rectangular_block_mask(
                    decoder_inputs.shape[1],
                    enc.shape[1],
                    dec_lengths,
                    enc_lengths,
                    device=self.device
                )
            )

        assert all(len(positions) == length for positions, length in zip(decoder_positions, decoder_lengths))

        encoder_outputs = {"encoder_outputs": enc}
        encoder_padding_masks = {"encoder_outputs": encoder_padding_mask}
        if self.cfg.use_sequence_context:
            encoder_padding_masks["full_encoder_outputs"] = encoder_padding_mask
            encoder_outputs["full_encoder_outputs"] = enc

        dec = self.decode(
            decoder_inputs=decoder_inputs,
            decoder_lengths=decoder_lengths,
            encoder_outputs=encoder_outputs,
            encoder_padding_masks=encoder_padding_masks,
            encoder_masks={"encoder_outputs": torch.stack(encoder_masks)},
            decoder_mask=torch.stack(decoder_masks),
            decoder_positions=to(utils.pad(decoder_positions).long(), self.device)
        )
        return dec, {}

    def encode(self, x: torch.Tensor, padding_mask: Optional[torch.Tensor] = None, **kwargs: Any) -> torch.Tensor:
        assert padding_mask is not None
        enc = self.embedding(x)
        return self.encoder(enc, padding_mask=padding_mask)

    def decode(self,
               decoder_inputs: torch.Tensor,
               **kwargs: Any) -> torch.Tensor:
        self.head: heads.TransformerDecoderHead
        return self.head.decode(decoder_inputs, **kwargs)


@dataclass
class ModelForSeq2SeqConfig(TensorModelConfig):
    type: Models = Models.MODEL_FOR_SEQ2SEQ
    input_tokenizer: TokenizerConfig = MISSING
    output_tokenizer: TokenizerConfig = MISSING

    max_output_length: int = 512

    embedding: TensorEmbeddingConfig = MISSING
    dropout: float = 0.1
    num_encoder_layers: int = MISSING
    num_decoder_layers: int = MISSING


class ModelForSeq2Seq(TensorModel, TensorEncoderMixin, DecoderMixin):
    def __init__(self,
                 sample_inputs: Batch,
                 cfg: ModelForSeq2SeqConfig,
                 device: torch.device) -> None:
        self.input_tokenizer = get_tokenizer_from_config(cfg.input_tokenizer)
        self.output_tokenizer = get_tokenizer_from_config(cfg.output_tokenizer)

        self.input_pad_token_id = self.input_tokenizer.token_to_id(tokenization.PAD)
        self.output_pad_token_id = self.output_tokenizer.token_to_id(tokenization.PAD)

        super().__init__(sample_inputs, cfg, device)

    def build_embedding(self, sample_input: DataInput) -> embedding.TensorEmbedding:
        self.cfg: ModelForSeq2SeqConfig
        return embedding.get_embedding_from_config(
            cfg=self.cfg.embedding,
            sample_inputs=sample_input,
            hidden_dim=self.cfg.hidden_dim,
            num_embeddings=self.input_tokenizer.vocab_size,
            max_length=self.cfg.max_length,
            padding_idx=self.input_pad_token_id
        )

    def build_encoder(self, sample_input: DataInput) -> nn.Module:
        self.cfg: ModelForSeq2SeqConfig
        return encoders.get_feature_encoder(
            encoder=encoders.Encoders.TRANSFORMER,
            in_dim=self.cfg.hidden_dim,
            hidden_dim=self.cfg.hidden_dim,
            dropout=self.cfg.dropout,
            num_layers=self.cfg.num_encoder_layers
        )

    def build_head(self, sample_input: Batch) -> heads.TransformerDecoderHead:
        self.cfg: ModelForSeq2SeqConfig
        return heads.TransformerDecoderHead(
            contexts=["encoder_outputs"],
            pad_token_id=self.output_pad_token_id,
            max_length=self.cfg.max_output_length,
            hidden_dim=self.cfg.hidden_dim,
            num_outputs=self.output_tokenizer.vocab_size,
            dropout=self.cfg.dropout,
            num_layers=self.cfg.num_decoder_layers
        )

    def pad_inputs(self, x: DataInput) -> Tuple[torch.Tensor, torch.Tensor]:
        assert all(t.ndim == 1 for t in x)
        encoder_lengths = [len(t) for t in x]
        encoder_inputs = to(utils.pad(x, self.input_pad_token_id).long(), self.device)
        padding_mask = utils.padding_mask(encoder_inputs, encoder_lengths)
        return encoder_inputs, padding_mask

    def forward(
            self,
            x: DataInput,
            decoder_inputs: Optional[List[torch.Tensor]] = None,
            **kwargs: Any
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        assert decoder_inputs is not None

        encoder_inputs, encoder_padding_mask = self.pad_inputs(x)
        enc = self.encode(encoder_inputs, encoder_padding_mask)

        decoder_lengths = [len(t) for t in decoder_inputs]
        decoder_inputs = to(utils.pad(decoder_inputs, self.output_pad_token_id).long(), self.device)

        dec = self.decode(
            decoder_inputs=decoder_inputs,
            decoder_lengths=decoder_lengths,
            encoder_outputs={"encoder_outputs": enc},
            encoder_padding_masks={"encoder_outputs": encoder_padding_mask}
        )
        return dec, {}

    def encode(self, x: torch.Tensor, padding_mask: Optional[torch.Tensor] = None, **kwargs: Any) -> torch.Tensor:
        assert padding_mask is not None
        enc = self.embedding(x)
        return self.encoder(enc, padding_mask=padding_mask)

    def decode(self,
               decoder_inputs: torch.Tensor,
               **kwargs: Any) -> torch.Tensor:
        self.head: heads.TransformerDecoderHead
        return self.head.decode(decoder_inputs, **kwargs)


@dataclass
class ModelForSequenceClassificationConfig(TensorModelConfig):
    type = Models.MODEL_FOR_SEQUENCE_CLASSIFICATION
    tokenizer: TokenizerConfig = MISSING
    num_classes: int = MISSING

    embedding: TensorEmbeddingConfig = MISSING
    dropout: float = 0.1
    num_layers: int = MISSING


class ModelForSequenceClassification(TensorModel):
    def __init__(
            self,
            sample_inputs: Batch,
            cfg: ModelForSequenceClassificationConfig,
            device: torch.device
    ) -> None:
        self.tokenizer = get_tokenizer_from_config(cfg.tokenizer)
        self.pad_token_id = self.tokenizer.token_to_id(tokenization.PAD)

        super().__init__(sample_inputs, cfg, device)

    def build_embedding(self, sample_input: DataInput) -> embedding.TensorEmbedding:
        self.cfg: ModelForSequenceClassificationConfig
        return embedding.get_embedding_from_config(
            self.cfg.embedding,
            sample_input,
            hidden_dim=self.cfg.hidden_dim,
            num_embeddings=self.tokenizer.vocab_size,
            max_length=self.cfg.max_length,
            padding_idx=self.pad_token_id
        )

    def build_encoder(self, sample_input: DataInput) -> nn.Module:
        self.cfg: ModelForSequenceClassificationConfig
        return encoders.Transformer(
            in_dim=self.cfg.hidden_dim,
            hidden_dim=self.cfg.hidden_dim,
            dropout=self.cfg.dropout,
            num_layers=self.cfg.num_layers
        )

    def build_head(self, sample_input: Batch) -> heads.TensorGroupHead:
        self.cfg: ModelForSequenceClassificationConfig

        if "groups" in sample_input.info:
            additional_features, aggregation = utils.get_additional_features_and_aggregations_from_group_stages(
                stages=sample_input.info["groups"][0]
            )
        else:
            additional_features = aggregation = None

        return heads.TensorGroupHead(
            hidden_dim=self.cfg.hidden_dim,
            num_classes=self.cfg.num_classes,
            num_additional_features=additional_features,
            aggregation=aggregation
        )

    def forward(
            self,
            x: DataInput,
            **kwargs: Any
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        assert all(t.ndim == 1 for t in x)

        lengths = [len(t) for t in x]
        x = to(utils.pad(x, self.pad_token_id).long(), self.device)
        padding_mask = utils.padding_mask(x, lengths)

        x = self.embedding(x)
        x = self.encoder(x, padding_mask=padding_mask)
        x = [x[i, :l] for i, l in enumerate(lengths)]
        output = self.head(x, **kwargs)
        return torch.cat(output, dim=0), {}


@dataclass
class ModelForTokenClassificationConfig(TensorModelConfig):
    type = Models.MODEL_FOR_TOKEN_CLASSIFICATION
    # embedding
    tokenizer: TokenizerConfig = MISSING
    embedding: Any = MISSING

    # encoder
    dropout: float = 0.1
    num_layers: int = MISSING
    feed_forward_dim: Optional[int] = None
    norm: bool = True
    activation: str = "relu"

    # head
    num_clf_layers: int = 2
    num_classes: int = MISSING


class ModelForTokenClassification(TensorModel):
    def __init__(
            self,
            sample_inputs: Batch,
            cfg: ModelForTokenClassificationConfig,
            device: torch.device
    ) -> None:
        self.tokenizer = get_tokenizer_from_config(cfg.tokenizer)
        self.pad_token_id = self.tokenizer.token_to_id(tokenization.PAD)

        super().__init__(sample_inputs, cfg, device)

    def build_embedding(self, sample_input: DataInput) -> embedding.TensorEmbedding:
        self.cfg: ModelForTokenClassificationConfig
        return embedding.get_embedding_from_config(
            self.cfg.embedding,
            sample_input,
            hidden_dim=self.cfg.hidden_dim,
            num_embeddings=self.tokenizer.vocab_size,
            max_length=self.cfg.max_length,
            padding_idx=self.pad_token_id
        )

    def build_encoder(self, sample_input: DataInput) -> nn.Module:
        self.cfg: ModelForTokenClassificationConfig
        return encoders.Transformer(
            in_dim=self.cfg.hidden_dim,
            hidden_dim=self.cfg.hidden_dim,
            dropout=self.cfg.dropout,
            num_layers=self.cfg.num_layers,
            feed_forward_dim=self.cfg.feed_forward_dim,
            norm=self.cfg.norm,
            activation=self.cfg.activation
        )

    def build_head(self, sample_input: Batch) -> heads.TensorGroupHead:
        self.cfg: ModelForTokenClassificationConfig

        if "groups" in sample_input.info:
            additional_features, aggregation = utils.get_additional_features_and_aggregations_from_group_stages(
                stages=sample_input.info["groups"][0]
            )
        else:
            additional_features = aggregation = None

        return heads.TensorGroupHead(
            hidden_dim=self.cfg.hidden_dim,
            num_classes=self.cfg.num_classes,
            num_additional_features=additional_features,
            aggregation=aggregation,
            num_layers=self.cfg.num_clf_layers
        )

    def forward(
            self,
            x: DataInput,
            **kwargs: Any
    ) -> Tuple[List[torch.Tensor], Dict[str, torch.Tensor]]:
        assert all(t.ndim == 1 for t in x)

        lengths = [len(t) for t in x]
        x = to(utils.pad(x, self.pad_token_id).long(), self.device)
        padding_mask = utils.padding_mask(x, lengths)

        x = self.embedding(x)
        x = self.encoder(x, padding_mask=padding_mask)
        x = [x[i, :l] for i, l in enumerate(lengths)]
        output = self.head(x, **kwargs)
        return output, {}


@dataclass
class ModelForTokenizationRepairPlusConfig(TensorModelConfig):
    type: Models = Models.MODEL_FOR_TOKENIZATION_REPAIR_PLUS

    input_type: str = "char"  # one of {char, byte}
    # one of {tokenization_repair_plus_sed, tokenization_repair_plus_sed_plus_sec}
    output_type: str = "tokenization_repair_plus_sed"

    embedding: TensorEmbeddingConfig = MISSING
    dropout: float = 0.1

    # tokenization repair backbone args
    start_from_tokenization_repair_checkpoint: Optional[str] = None
    fix_tokenization_repair: bool = False
    tokenization_repair_feature_layers: Tuple[int] = (-1,)
    num_tokenization_repair_layers: int = MISSING
    num_tokenization_repair_clf_layers: int = 1
    tokenization_repair_feed_forward_dim: Optional[int] = None
    tokenization_repair_norm: bool = True
    tokenization_repair_activation: str = "relu"

    # word backbone args
    num_word_layers: int = MISSING

    # special args for sed head
    num_sed_clf_layers: int = 2

    # special args for sec head
    sec_tokenizer: Optional[TokenizerConfig] = None
    num_sec_layers: int = MISSING
    sec_max_output_length: int = 512


class ModelForTokenizationRepairPlus(TensorModel, TensorEncoderMixin):
    def __init__(
            self,
            sample_inputs: Batch,
            cfg: ModelForTokenizationRepairPlusConfig,
            device: torch.device
    ) -> None:
        assert cfg.output_type in {
            "tokenization_repair_plus_sed",
            "tokenization_repair_plus_sed_plus_sec"
        }

        if cfg.input_type == "char":
            self.input_tokenizer = tokenization.CharTokenizer()
        elif cfg.input_type == "byte":
            self.input_tokenizer = tokenization.ByteTokenizer()
        else:
            raise ValueError(f"unknown input type {cfg.input_type}, must be one of {{char, byte}}")

        self.input_pad_token_id = self.input_tokenizer.token_to_id(tokenization.PAD)

        if cfg.output_type == "tokenization_repair_plus_sed_plus_sec":
            assert cfg.sec_tokenizer is not None
            self.sec_tokenizer = get_tokenizer_from_config(cfg.sec_tokenizer)
            self.sec_pad_token_id = self.sec_tokenizer.token_to_id(tokenization.PAD)

        super().__init__(sample_inputs, cfg, device)

        if cfg.start_from_tokenization_repair_checkpoint is not None:
            checkpoint = io.load_checkpoint(cfg.start_from_tokenization_repair_checkpoint, self.device)
            ckpt_state_dict = checkpoint["model_state_dict"]
            ckpt_encoder_state_dict = io.filter_state_dict(ckpt_state_dict, "encoder.")
            ckpt_head_state_dict = io.filter_state_dict(ckpt_state_dict, "head.")
            ckpt_embedding_emb_state_dict = io.filter_state_dict(ckpt_state_dict, "embedding.embedding.")
            ckpt_embedding_norm_state_dict = io.filter_state_dict(ckpt_state_dict, "embedding.norm.")

            if cfg.embedding.embed_positions and cfg.embedding.learned_position_embedding:
                ckpt_embedding_pos_emb_state_dict = io.filter_state_dict(ckpt_state_dict, "embedding.pos_emb.")
                self.embedding.pos_emb.load_state_dict(ckpt_embedding_pos_emb_state_dict)

            self.embedding.norm.load_state_dict(ckpt_embedding_norm_state_dict)
            self.embedding.embedding.load_state_dict(ckpt_embedding_emb_state_dict)
            self.encoder.load_state_dict(ckpt_encoder_state_dict)
            self.head["tokenization_repair"].load_state_dict(ckpt_head_state_dict)

        if cfg.fix_tokenization_repair:
            for param in self.embedding.parameters():
                param.requires_grad = False
            for param in self.encoder.parameters():
                param.requires_grad = False
            for param in self.head["tokenization_repair"].parameters():
                param.requires_grad = False

        self.word_encoder = self.build_word_encoder(sample_inputs)
        assert len(cfg.tokenization_repair_feature_layers) > 0 and all(
            -cfg.num_tokenization_repair_layers <= layer < cfg.num_tokenization_repair_layers
            for layer in cfg.tokenization_repair_feature_layers
        ), f"expected all layers for feature extraction to be in [{-cfg.num_tokenization_repair_layers}," \
           f"{cfg.num_tokenization_repair_layers}), but got {cfg.tokenization_repair_feature_layers}"
        # initialize weights uniformly and let the model figure during training which layers are important
        self.encoder_layer_weights = nn.Parameter(torch.zeros(len(cfg.tokenization_repair_feature_layers)))
        self.encoder_hooks = hooks.ModelHook()
        self.encoder_layer_indices = sorted(list(
            set(cfg.num_tokenization_repair_layers + layer if layer < 0 else layer
                for layer in cfg.tokenization_repair_feature_layers)
        ))
        for layer_idx in self.encoder_layer_indices:
            self.encoder_hooks.attach(
                name=f"layer_{layer_idx}",
                module=self.encoder.encoder.layers[layer_idx],
                hook=hooks.SaveOutputHook()
            )

    def build_embedding(self, sample_input: DataInput) -> embedding.TensorEmbedding:
        self.cfg: ModelForTokenClassificationConfig
        return embedding.get_embedding_from_config(
            self.cfg.embedding,
            sample_input,
            hidden_dim=self.cfg.hidden_dim,
            num_embeddings=self.input_tokenizer.vocab_size,
            max_length=self.cfg.max_length,
            padding_idx=self.input_pad_token_id
        )

    def build_encoder(self, sample_input: DataInput) -> nn.Module:
        self.cfg: ModelForTokenizationRepairPlusConfig
        return encoders.Transformer(
            in_dim=self.cfg.hidden_dim,
            hidden_dim=self.cfg.hidden_dim,
            dropout=self.cfg.dropout,
            num_layers=self.cfg.num_tokenization_repair_layers,
            feed_forward_dim=self.cfg.tokenization_repair_feed_forward_dim,
            norm=self.cfg.tokenization_repair_norm,
            activation=self.cfg.tokenization_repair_activation
        )

    def pad_inputs(self, x: DataInput, pad_val: float = 0) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
        assert all(t.ndim == 1 for t in x)
        lengths = [len(t) for t in x]
        inputs = to(utils.pad(x, pad_val).long(), self.device)
        padding_mask = utils.padding_mask(inputs, lengths)
        return inputs, padding_mask, lengths

    def encode(self, x: torch.Tensor, padding_mask: Optional[torch.Tensor] = None, **kwargs: Any) \
            -> Tuple[torch.Tensor, torch.Tensor]:
        assert padding_mask is not None
        emb = self.embedding(x)
        enc = self.encoder(emb, padding_mask=padding_mask)

        enc_features = torch.stack([
            self.encoder_hooks[f"layer_{layer_idx}"][f"layer_{layer_idx}"]
            for layer_idx in self.encoder_layer_indices
        ])
        layer_weights = torch.softmax(self.encoder_layer_weights, dim=0)
        print(f"layer weights: {layer_weights}, layer indices: {self.encoder_layer_indices}")
        enc_features = (enc_features * einops.repeat(layer_weights, "f -> f b l h", b=1, l=1, h=1)).sum(dim=0)
        return enc, enc_features

    def build_word_encoder(self, sample_input: Batch) -> nn.Module:
        self.cfg: ModelForTokenizationRepairPlusConfig

        word_features = sample_input.info.get("word_features")
        if word_features is not None:
            additional_word_features = word_features[0].shape[1]
        else:
            additional_word_features = 0

        return encoders.Transformer(
            in_dim=self.cfg.hidden_dim + additional_word_features,
            hidden_dim=self.cfg.hidden_dim,
            dropout=self.cfg.dropout,
            num_layers=self.cfg.num_word_layers
        )

    def build_head(self, sample_input: Batch) -> nn.ModuleDict:
        self.cfg: ModelForTokenizationRepairPlusConfig

        assert "word_ws_groups" in sample_input.info
        _, sed_aggregation = utils.get_additional_features_and_aggregations_from_group_stages(
            stages=sample_input.info["word_ws_groups"][0]
        )

        heads_dict = nn.ModuleDict({
            "tokenization_repair": heads.TensorGroupHead(
                hidden_dim=self.cfg.hidden_dim,
                num_classes=3,
                num_layers=self.cfg.num_tokenization_repair_clf_layers
            ),
            "sed": heads.TensorGroupHead(
                hidden_dim=self.cfg.hidden_dim,
                num_classes=2,
                aggregation=sed_aggregation,
                num_layers=self.cfg.num_sed_clf_layers
            )
        })
        if self.cfg.output_type == "tokenization_repair_plus_sed_plus_sec":
            heads_dict["sec"] = heads.TransformerDecoderHead(
                contexts=[self.cfg.input_type, "word"],
                pad_token_id=self.sec_pad_token_id,
                max_length=self.cfg.sec_max_output_length,
                hidden_dim=self.cfg.hidden_dim,
                num_outputs=self.sec_tokenizer.vocab_size,
                dropout=self.cfg.dropout,
                num_layers=self.cfg.num_sec_layers
            )
        return heads_dict

    def forward(
            self,
            x: DataInput,
            char_groups: Optional[List[Dict[str, torch.Tensor]]] = None,
            word_groups: Optional[List[Dict[str, torch.Tensor]]] = None,
            word_features: Optional[List[torch.Tensor]] = None,
            word_ws_groups: Optional[List[List[Dict[str, torch.Tensor]]]] = None,
            input_group_lengths: Optional[List[torch.Tensor]] = None,
            word_group_lengths: Optional[List[torch.Tensor]] = None,
            sec_decoder_inputs: Optional[List[torch.Tensor]] = None,
            sec_decoder_group_lengths: Optional[List[torch.Tensor]] = None,
            **kwargs: Any
    ) -> Tuple[Dict[str, Any], Dict[str, torch.Tensor]]:
        self.cfg: ModelForTokenizationRepairPlusConfig

        outputs = {}

        assert word_groups is not None and word_ws_groups is not None
        assert all(t.ndim == 1 for t in x)

        x, padding_mask, lengths = self.pad_inputs(x, pad_val=self.input_pad_token_id)

        # embed tokens and encode input representations
        x_tr, x = self.encode(x.long(), padding_mask=padding_mask)

        # tokenization repair output
        x_tr = [x_tr[i, :l] for i, l in enumerate(lengths)]
        outputs["tokenization_repair"] = self.head["tokenization_repair"](x_tr)

        # weighted layer features
        x = [x[i, :l] for i, l in enumerate(lengths)]

        if self.cfg.input_type == "byte":
            assert char_groups is not None
            # if we have bytes as inputs group bytes into characters, since tokenization repair output is on
            # character level
            x = utils.group_features(
                x, char_groups, aggregation="mean"
            )

        # group characters by word (leaving out whitespaces)
        x = utils.group_features(
            x, word_groups, aggregation="stack"
        )

        # average character representations per word to get word representations
        word_feat = [
            torch.cat([torch.mean(t, dim=0, keepdim=True) for t in stacked_feat], dim=0)
            for stacked_feat in x
        ]

        # add additional word features to word representations
        if word_features is not None:
            additional_word_features = to(word_features, self.device)
            word_feat = [
                torch.cat([w_feat, add_w_feat], dim=1)
                for w_feat, add_w_feat in zip(word_feat, additional_word_features)
            ]

        # encode word representations
        lengths = [len(t) for t in word_feat]
        word_feat = to(utils.pad(word_feat), self.device)
        word_padding_mask = utils.padding_mask(word_feat, lengths)
        word_feat = self.word_encoder(word_feat, padding_mask=word_padding_mask)

        outputs["sed"] = self.head["sed"]([word_feat[i, :l, :] for i, l in enumerate(lengths)], groups=word_ws_groups)

        if self.cfg.output_type == "tokenization_repair_plus_sed_plus_sec":
            assert sec_decoder_inputs is not None and sec_decoder_group_lengths is not None
            sec_decoder_lengths = [len(t) for t in sec_decoder_inputs]
            sec_decoder_inputs = to(utils.pad(sec_decoder_inputs, self.sec_pad_token_id).long(), self.device)

            decoder_masks = []
            input_encoder_masks = []
            word_encoder_masks = []
            decoder_positions = []

            # flatten the stacked character representations per word into a single tensor
            x = [
                torch.cat([t.reshape(-1, self.cfg.hidden_dim) for t in stacked_feat])
                for stacked_feat in x
            ]
            lengths = [len(t) for t in x]
            x = utils.pad(x)

            for input_lengths, word_lengths, dec_lengths in zip(
                    input_group_lengths, word_group_lengths, sec_decoder_group_lengths
            ):
                # align decoder positions with words
                decoder_positions.append(
                    torch.cat(
                        [
                            torch.arange(word_position, word_position + dec_length)
                            for word_position, dec_length
                            in zip(torch.cat([torch.tensor([0]), torch.cumsum(word_lengths, dim=0)[:-1]]), dec_lengths)
                        ]
                    )
                )
                assert len(decoder_positions[-1]) == dec_lengths.sum()

                decoder_masks.append(
                    utils.square_causal_block_mask(
                        sec_decoder_inputs.shape[1],
                        dec_lengths,
                        device=self.device
                    )
                )

                input_encoder_masks.append(
                    utils.rectangular_block_mask(
                        sec_decoder_inputs.shape[1],
                        x.shape[1],
                        dec_lengths,
                        input_lengths,
                        device=self.device
                    )
                )

                word_encoder_masks.append(
                    utils.rectangular_block_mask(
                        sec_decoder_inputs.shape[1],
                        word_feat.shape[1],
                        dec_lengths,
                        word_lengths,
                        device=self.device
                    )
                )

            assert all(
                len(positions) == length
                for positions, length in zip(decoder_positions, sec_decoder_lengths)
            )

            outputs["sec"] = self.head["sec"](
                decoder_inputs=sec_decoder_inputs,
                decoder_lengths=sec_decoder_lengths,
                encoder_outputs={
                    self.cfg.input_type: x,
                    "word": word_feat
                },
                encoder_padding_masks={
                    self.cfg.input_type: utils.padding_mask(x, lengths),
                    "word": word_padding_mask
                },
                encoder_masks={
                    self.cfg.input_type: torch.stack(input_encoder_masks),
                    "word": torch.stack(word_encoder_masks)
                },
                decoder_mask=torch.stack(decoder_masks),
                decoder_positions=to(utils.pad(decoder_positions).long(), self.device)
            )

        return outputs, {}
