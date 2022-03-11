import enum
from dataclasses import dataclass
from typing import Optional, Dict, Tuple, List, Any, Union

import dgl
import numpy as np
import omegaconf
from omegaconf import MISSING
import torch
from torch import nn

from gnn_lib.data import tokenization
from gnn_lib.data.tokenization import TokenizerConfig, get_tokenizer_from_config, Tokenizer
from gnn_lib.modules import heads, embedding, utils, encoders
from gnn_lib.modules.embedding import GraphEmbeddingConfig, TensorEmbeddingConfig
from gnn_lib.modules.utils import GraphEncoderMixin, DecoderMixin, pad
from gnn_lib.utils import TENSOR_INPUT, MODEL_INPUTS, DATA_INPUT


class Models(enum.IntEnum):
    MODEL_FOR_GRAPH_CLASSIFICATION = 1
    MODEL_FOR_MULTI_NODE_CLASSIFICATION = 2
    MODEL_FOR_GRAPH2SEQ = 3
    MODEL_FOR_MULTI_NODE2SEQ = 4
    MODEL_FOR_TOKEN_CLASSIFICATION = 5
    MODEL_FOR_SEQ2SEQ = 6
    MODEL_FOR_MULTI_TOKEN2SEQ = 7


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

    gnn_type = cfg.type if isinstance(cfg, GNNConfig) else GNNs[cfg.type]
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
    type: Models


class Model(nn.Module):
    def __init__(self,
                 cfg: ModelConfig,
                 device: torch.device) -> None:
        super().__init__()
        self.cfg = cfg
        self.device = device

    @staticmethod
    def _check_input(data: DATA_INPUT) -> None:
        raise NotImplementedError


@dataclass
class GraphModelConfig(ModelConfig):
    node_hidden_dim: int = MISSING
    edge_hidden_dim: Optional[int] = None
    hidden_feature: str = "h"

    embedding: GraphEmbeddingConfig = MISSING
    gnn: GNNConfig = MISSING


class GraphModel(Model):
    def __init__(self, sample_inputs: MODEL_INPUTS, cfg: GraphModelConfig, device: torch.device) -> None:
        super().__init__(cfg, device)
        graph, infos = sample_inputs
        self._check_input(graph)

        self.embedding = self.build_embedding(graph)
        self.gnn = self.build_gnn(graph)
        self.head = self.build_head(sample_inputs)

    @staticmethod
    def _check_input(data: DATA_INPUT) -> None:
        if not isinstance(data, dgl.DGLHeteroGraph):
            raise RuntimeError(
                f"expected data input to be a dgl heterograph instance for a graph model, but got {type(data)}"
            )

    def build_embedding(self, sample_g: dgl.DGLHeteroGraph) -> embedding.GraphEmbedding:
        # embedding is also sometimes model specific, because of the use of tokenizers, so we override this
        # in the models
        raise NotImplementedError

    def build_gnn(self, sample_g: dgl.DGLHeteroGraph) -> GNN:
        self.cfg: GraphModelConfig
        return get_gnn_from_config(
            self.cfg.gnn,
            sample_g,
            self.cfg.node_hidden_dim,
            self.cfg.edge_hidden_dim,
            self.cfg.hidden_feature
        )

    def build_head(self, sample_g: MODEL_INPUTS) -> nn.Module:
        # head is mostly model specific, so we override this in the models
        raise NotImplementedError

    def forward(self,
                g: DATA_INPUT,
                **kwargs: Any) -> \
            Tuple[Any, Dict[str, torch.Tensor]]:
        self._check_input(g)
        g = g.to(self.device)

        g = self.embedding(g)
        g = self.gnn(g)
        output = self.head(g, **kwargs)

        return output, self.gnn.get_additional_losses()


@dataclass
class TensorModelConfig(ModelConfig):
    hidden_dim: int = MISSING


class TensorModel(Model):
    def __init__(self, sample_inputs: MODEL_INPUTS, cfg: TensorModelConfig, device: torch.device) -> None:
        super().__init__(cfg, device)
        tensor, infos = sample_inputs
        self._check_input(tensor)

        self.embedding = self.build_embedding(tensor)
        self.encoder = self.build_encoder(tensor)
        self.head = self.build_head(sample_inputs)

    @staticmethod
    def _check_input(data: DATA_INPUT) -> None:
        if (
                # check for non empty tensor list
                isinstance(data, list)
                and all(isinstance(d, torch.Tensor) for d in data)
        ):
            return
        raise RuntimeError(
            f"expected data input to be a non empty tensor list, but got {type(data)}"
        )

    def _to_device(self, x: TENSOR_INPUT) -> TENSOR_INPUT:
        return [e.to(self.device) for e in x]

    def build_embedding(self, sample_input: TENSOR_INPUT) -> embedding.TokenEmbedding:
        # embedding is also sometimes model specific, because of the use of tokenizers, so we override this
        # in the models
        raise NotImplementedError

    def build_encoder(self, sample_input: TENSOR_INPUT) -> nn.Module:
        raise NotImplementedError

    def build_head(self, sample_input: MODEL_INPUTS) -> nn.Module:
        # head is mostly model specific, so we override this in the models
        raise NotImplementedError

    def forward(self,
                x: TENSOR_INPUT,
                **kwargs: Any) -> \
            Tuple[Any, Dict[str, torch.Tensor]]:
        raise NotImplementedError


def get_model_from_config(
        cfg: omegaconf.DictConfig,
        sample_inputs: MODEL_INPUTS,
        device: torch.device) -> Model:
    model_type = Models[cfg.type]
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
    elif model_type == Models.MODEL_FOR_MULTI_TOKEN2SEQ:
        cfg = omegaconf.OmegaConf.structured(ModelForMultiToken2SeqConfig(**cfg))
        return ModelForMultiToken2Seq(sample_inputs, cfg, device)
    elif model_type == Models.MODEL_FOR_TOKEN_CLASSIFICATION:
        cfg = omegaconf.OmegaConf.structured(ModelForTokenClassificationConfig(**cfg))
        return ModelForTokenClassification(sample_inputs, cfg, device)
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
                 sample_inputs: MODEL_INPUTS,
                 cfg: ModelForGraphClassificationConfig,
                 device: torch.device) -> None:
        self.tokenizers = {k: get_tokenizer_from_config(cfg) for k, cfg in cfg.tokenizers.items()}
        super().__init__(sample_inputs, cfg, device)

    def build_embedding(self, sample_g: dgl.DGLHeteroGraph) -> embedding.GraphEmbedding:
        num_token_embeddings = {
            node_type: self.tokenizers[node_type].vocab_size
            for node_type in sample_g.ntypes
            if f"{node_type}_id" in sample_g.node_attr_schemes(node_type)
        }
        self.cfg: GraphModelConfig
        return embedding.get_embedding_from_config(
            self.cfg.embedding,
            sample_g,
            node_hidden_dim=self.cfg.node_hidden_dim,
            hidden_feature=self.cfg.hidden_feature,
            num_token_embeddings=num_token_embeddings,
            edge_hidden_dim=self.cfg.edge_hidden_dim
        )

    def build_head(self, sample_inputs: MODEL_INPUTS) -> heads.GraphClassificationHead:
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
                 sample_inputs: MODEL_INPUTS,
                 cfg: ModelForMultiNodeClassificationConfig,
                 device: torch.device) -> None:
        self.tokenizers: Dict[str, Tokenizer] = {k: get_tokenizer_from_config(cfg) for k, cfg in cfg.tokenizers.items()}
        super().__init__(sample_inputs, cfg, device)

    def build_embedding(self, sample_g: dgl.DGLHeteroGraph) -> embedding.GraphEmbedding:
        num_token_embeddings = {
            node_type: self.tokenizers[node_type].vocab_size
            for node_type in sample_g.ntypes
            if f"{node_type}_id" in sample_g.node_attr_schemes(node_type)
        }
        self.cfg: GraphModelConfig
        return embedding.get_embedding_from_config(
            self.cfg.embedding,
            sample_g,
            node_hidden_dim=self.cfg.node_hidden_dim,
            hidden_feature=self.cfg.hidden_feature,
            num_token_embeddings=num_token_embeddings,
            edge_hidden_dim=self.cfg.edge_hidden_dim
        )

    def build_head(self, sample_inputs: MODEL_INPUTS) -> heads.MultiNodeClassificationGroupHead:
        self.cfg: ModelForMultiNodeClassificationConfig
        _, sample_info = sample_inputs
        return heads.MultiNodeClassificationGroupHead(
            feat=self.cfg.hidden_feature,
            num_features=self.cfg.node_hidden_dim,
            num_classes=self.cfg.num_classes,
            num_additional_features={
                node_type: {stage["stage"]: len(stage["features"][0]) for stage in stages if "features" in stage}
                for node_type, stages in sample_info["groups"][0].items()
            },
            aggregation={
                node_type: {stage["stage"]: stage.get("aggregation", "mean") for stage in stages}
                for node_type, stages in sample_info["groups"][0].items()
            }
        )


@dataclass
class ModelForGraph2SeqConfig(GraphModelConfig):
    type: Models = Models.MODEL_FOR_GRAPH2SEQ
    context_node_types: List[str] = MISSING
    tokenizers: Dict[str, TokenizerConfig] = MISSING
    decoder_dropout: float = 0.2
    decoder_num_layers: int = MISSING
    decoder_share_parameters: bool = False


class ModelForGraph2Seq(GraphModel, GraphEncoderMixin, DecoderMixin):
    def __init__(self, sample_inputs: MODEL_INPUTS, cfg: ModelForGraph2SeqConfig, device: torch.device) -> None:
        self.tokenizers: Dict[str, Tokenizer] = {
            k: get_tokenizer_from_config(cfg) for k, cfg in cfg.tokenizers.items()
        }

        self.bos_token_id = self.tokenizers["output_tokenizer"].token_to_id(tokenization.BOS)
        self.eos_token_id = self.tokenizers["output_tokenizer"].token_to_id(tokenization.EOS)
        self.pad_token_id = self.tokenizers["output_tokenizer"].token_to_id(tokenization.PAD)
        self.num_outputs = self.tokenizers["output_tokenizer"].vocab_size

        super().__init__(sample_inputs, cfg, device)

    def build_embedding(self, sample_g: dgl.DGLHeteroGraph) -> embedding.GraphEmbedding:
        num_token_embeddings = {
            node_type: self.tokenizers[node_type].vocab_size
            for node_type in sample_g.ntypes
            if f"{node_type}_id" in sample_g.node_attr_schemes(node_type)
        }
        self.cfg: GraphModelConfig
        return embedding.get_embedding_from_config(
            self.cfg.embedding,
            sample_g,
            node_hidden_dim=self.cfg.node_hidden_dim,
            hidden_feature=self.cfg.hidden_feature,
            num_token_embeddings=num_token_embeddings,
            edge_hidden_dim=self.cfg.edge_hidden_dim
        )

    def build_head(self, sample_inputs: MODEL_INPUTS) -> utils.DecoderMixin:
        self.cfg: ModelForGraph2SeqConfig
        return heads.TransformerDecoderHead(
            contexts=self.cfg.context_node_types,
            pad_token_id=self.pad_token_id,
            max_length=512,
            hidden_dim=self.cfg.node_hidden_dim,
            num_outputs=self.num_outputs,
            dropout=self.cfg.decoder_dropout,
            num_layers=self.cfg.decoder_num_layers,
            share_parameters=self.cfg.decoder_share_parameters
        )

    def forward(
            self,
            g: dgl.DGLHeteroGraph,
            decoder_inputs: Optional[torch.Tensor] = None,
            decoder_lengths: Optional[torch.Tensor] = None,
            **kwargs: Any
    ) -> Tuple[Any, Dict[str, torch.Tensor]]:
        assert decoder_inputs is not None and decoder_lengths is not None
        g = self.encode(g)

        encoder_outputs, encoder_lengths = utils.graph2seq_encoder_outputs_from_graph(
            g, self.cfg.context_node_types, self.cfg.hidden_feature
        )

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
class ModelForMultiNode2SeqConfig(GraphModelConfig):
    type: Models = Models.MODEL_FOR_MULTI_NODE2SEQ
    context_node_types: Dict[str, List[str]] = MISSING
    align_positions_with: Optional[Dict[str, str]] = None
    tokenizers: Dict[str, TokenizerConfig] = MISSING
    decoder_dropout: float = 0.2
    decoder_num_layers: int = MISSING
    decoder_share_parameters: bool = False
    decoder_node_types: Optional[List[str]] = None


class ModelForMultiNode2Seq(GraphModel, GraphEncoderMixin):
    def __init__(self, sample_inputs: MODEL_INPUTS, cfg: ModelForMultiNode2SeqConfig, device: torch.device) -> None:
        self.tokenizers: Dict[str, Tokenizer] = {
            k: get_tokenizer_from_config(cfg) for k, cfg in cfg.tokenizers.items()
        }

        self.pad_token_ids = {
            node_type: self.tokenizers[f"{node_type}_output_tokenizer"].token_to_id(tokenization.PAD)
            for node_type in cfg.decoder_node_types
        }
        self.num_outputs = {
            node_type: self.tokenizers[f"{node_type}_output_tokenizer"].vocab_size
            for node_type in cfg.decoder_node_types
        }
        super().__init__(sample_inputs, cfg, device)

    def build_embedding(self, sample_g: dgl.DGLHeteroGraph) -> embedding.GraphEmbedding:
        num_token_embeddings = {
            node_type: self.tokenizers[node_type].vocab_size
            for node_type in sample_g.ntypes
            if f"{node_type}_id" in sample_g.node_attr_schemes(node_type)
        }
        self.cfg: GraphModelConfig
        return embedding.get_embedding_from_config(
            self.cfg.embedding,
            sample_g,
            node_hidden_dim=self.cfg.node_hidden_dim,
            hidden_feature=self.cfg.hidden_feature,
            num_token_embeddings=num_token_embeddings,
            edge_hidden_dim=self.cfg.edge_hidden_dim
        )

    def build_head(self, sample_inputs: MODEL_INPUTS) -> nn.ModuleDict:
        self.cfg: ModelForMultiNode2SeqConfig
        node2seq_head = nn.ModuleDict()
        for node_type in self.cfg.decoder_node_types:
            node2seq_head[node_type] = heads.TransformerDecoderHead(
                contexts=self.cfg.context_node_types[node_type] + [node_type],
                pad_token_id=self.pad_token_ids[node_type],
                max_length=512,
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
            (
                encoder_outputs, encoder_lengths, aligned_encoder_positions
            ) = utils.multi_node2seq_encoder_outputs_from_graph2(
                g,
                node_type,
                self.cfg.context_node_types[node_type],
                self.cfg.hidden_feature,
                self.cfg.align_positions_with[node_type] if self.cfg.align_positions_with is not None else None
            )

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

            # bring encoder outputs into correct format
            encoder_outputs = {
                enc_node_type: pad([
                    torch.cat(enc_output)
                    for enc_output in enc_outputs
                ])
                for enc_node_type, enc_outputs in encoder_outputs.items()
            }

            decoder_mask = torch.stack([
                utils.square_causal_block_mask(
                    decoder_inputs.shape[1],
                    dec_lengths.cpu().numpy(),
                    device=g.device
                ) for dec_lengths in decoder_lengths
            ])

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

            # sum up decoder and encoder lengths
            decoder_lengths = torch.stack([lengths.sum() for lengths in decoder_lengths])

            encoder_lengths = {
                enc_node_type: torch.stack([lengths.sum() for lengths in enc_lengths])
                for enc_node_type, enc_lengths in encoder_lengths.items()
            }

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


@dataclass
class ModelForMultiToken2SeqConfig(TensorModelConfig):
    type: Models = Models.MODEL_FOR_MULTI_TOKEN2SEQ
    input_tokenizer: TokenizerConfig = MISSING
    output_tokenizer: TokenizerConfig = MISSING

    embedding: TensorEmbeddingConfig = MISSING
    dropout: float = 0.1
    num_encoder_layers: int = MISSING
    num_decoder_layers: int = MISSING


@dataclass
class ModelForMultiToken2Seq(TensorModel):
    pass


@dataclass
class ModelForSeq2SeqConfig(TensorModelConfig):
    type: Models = Models.MODEL_FOR_SEQ2SEQ
    input_tokenizer: TokenizerConfig = MISSING
    output_tokenizer: TokenizerConfig = MISSING

    embedding: TensorEmbeddingConfig = MISSING
    dropout: float = 0.1
    num_encoder_layers: int = MISSING
    num_decoder_layers: int = MISSING


@dataclass
class ModelForSeq2Seq(TensorModel, DecoderMixin):
    def __init__(self,
                 sample_inputs: MODEL_INPUTS,
                 cfg: ModelForSeq2SeqConfig,
                 device: torch.device) -> None:
        super().__init__(sample_inputs, cfg, device)

        self.input_tokenizer = get_tokenizer_from_config(cfg.input_tokenizer)
        self.output_tokenizer = get_tokenizer_from_config(cfg.output_tokenizer)

        self.input_pad_token_id = self.input_tokenizer.token_to_id(tokenization.PAD)
        self.output_pad_token_id = self.output_tokenizer.token_to_id(tokenization.PAD)

    def build_embedding(self, sample_input: TENSOR_INPUT) -> embedding.TensorEmbedding:
        self.cfg: ModelForTokenClassificationConfig
        return embedding.get_embedding_from_config(
            self.cfg.embedding,
            sample_input,
            hidden_dim=self.cfg.node_hidden_dim,
            num_embeddings=self.tokenizer.vocab_size,
            padding_idx=self.pad_token_id
        )

    def build_encoder(self, sample_input: TENSOR_INPUT) -> nn.Module:
        self.cfg: ModelForTokenClassificationConfig
        return encoders.Transformer(
            in_dim=self.cfg.hidden_dim,
            hidden_dim=self.cfg.hidden_dim,
            dropout=self.cfg.dropout,
            num_layers=self.cfg.num_layers
        )

    def build_head(self, sample_input: MODEL_INPUTS) -> heads.TransformerDecoderHead:
        self.cfg: ModelForSeq2SeqConfig
        return heads.TransformerDecoderHead(
            contexts=["encoder_outputs"],
            pad_token_id=self.output_pad_token_id,
            max_length=512,
            hidden_dim=self.cfg.hidden_dim,
            num_outputs=self.output_tokenizer.vocab_size,
            dropout=self.cfg.dropout,
            num_layers=self.cfg.num_decoder_layers
        )

    def forward(
            self,
            x: TENSOR_INPUT,
            decoder_inputs: Optional[TENSOR_INPUT] = None,
            **kwargs: Any
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        assert all(t.ndim == 1 for t in x) and decoder_inputs is not None
        x = self._to_device(x)

        encoder_lengths = [len(t) for t in x]
        encoder_inputs = utils.pad(x, self.input_pad_token_id).long()
        padding_mask = utils.padding_mask(encoder_inputs, encoder_lengths)

        enc = self.embedding(x)
        enc = self.encoder(enc, padding_mask=padding_mask)

        decoder_lengths = [len(t) for t in decoder_inputs]
        decoder_inputs = utils.pad(decoder_inputs, self.output_pad_token_id)
        dec = self.decode(
            decoder_inputs=decoder_inputs,
            decoder_lengths=decoder_lengths,
            encoder_outputs={"encoder_outputs": enc},
            encoder_lengths=encoder_lengths
        )
        return dec, {}

    def decode(self,
               decoder_inputs: torch.Tensor,
               **kwargs: Any) -> torch.Tensor:
        self.head: heads.TransformerDecoderHead
        return self.head.decode(decoder_inputs, **kwargs)


@dataclass
class ModelForTokenClassificationConfig(TensorModelConfig):
    type = Models.MODEL_FOR_TOKEN_CLASSIFICATION
    tokenizer: TokenizerConfig = MISSING
    num_classes: int = MISSING

    embedding: TensorEmbeddingConfig = MISSING
    dropout: float = 0.1
    num_layers: int = MISSING


class ModelForTokenClassification(TensorModel):
    def __init__(self,
                 sample_inputs: MODEL_INPUTS,
                 cfg: ModelForTokenClassificationConfig,
                 device: torch.device) -> None:
        super().__init__(sample_inputs, cfg, device)

        self.tokenizer = get_tokenizer_from_config(cfg.tokenizer)
        self.pad_token_id = self.tokenizer.token_to_id(tokenization.PAD)

    def build_embedding(self, sample_input: TENSOR_INPUT) -> embedding.TensorEmbedding:
        self.cfg: ModelForTokenClassificationConfig
        return embedding.get_embedding_from_config(
            self.cfg.embedding,
            sample_input,
            hidden_dim=self.cfg.node_hidden_dim,
            num_embeddings=self.tokenizer.vocab_size,
            padding_idx=self.pad_token_id
        )

    def build_encoder(self, sample_input: TENSOR_INPUT) -> nn.Module:
        self.cfg: ModelForTokenClassificationConfig
        return encoders.Transformer(
            in_dim=self.cfg.hidden_dim,
            hidden_dim=self.cfg.hidden_dim,
            dropout=self.cfg.dropout,
            num_layers=self.cfg.num_layers
        )

    def build_head(self, sample_input: MODEL_INPUTS) -> nn.Module:
        self.cfg: ModelForTokenClassificationConfig
        return nn.Sequential(
            nn.Linear(self.cfg.hidden_dim, self.cfg.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.cfg.dropout),
            nn.Linear(self.cfg.hidden_dim, self.cfg.num_classes)
        )

    def forward(
            self,
            x: TENSOR_INPUT,
            **kwargs: Any
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        assert all(t.ndim == 1 for t in x)
        x = self._to_device(x)

        x = utils.pad(x, self.pad_token_id).long()
        lengths = [len(t) for t in x]
        padding_mask = utils.padding_mask(x, lengths)

        x = self.embedding(x)
        x = self.encoder(x, padding_mask=padding_mask)
        output = self.head(x, **kwargs)
        return output, {}
