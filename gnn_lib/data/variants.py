import enum
import hashlib
import os
from dataclasses import dataclass
from typing import Tuple, List, Union, Optional, Dict, Any

import dgl
import numpy as np
import omegaconf
from omegaconf import MISSING, OmegaConf

from gnn_lib.data import graph, tokenization, utils, index
from gnn_lib.data.noise import get_noise_from_config
from gnn_lib.data.tokenization import TokenizerConfig, get_tokenizer_from_config
from gnn_lib.utils import tokenization_repair, io


class DatasetVariants(enum.IntEnum):
    SED_SEQUENCE = 1
    SED_WORDS = 2
    TOKENIZATION_REPAIR = 3
    TOKENIZATION_REPAIR_NMT = 4
    SEC_WORDS_NMT = 5
    SEC_NMT = 6


@dataclass
class DatasetVariantConfig:
    type: DatasetVariants  # = MISSING


class DatasetVariant:
    def __init__(self,
                 cfg: DatasetVariantConfig,
                 seed: int):
        self.cfg = cfg
        self.seed = seed
        self.rand = np.random.default_rng(seed)

        self.tok_fn: utils.TOKENIZATION_FN = None
        self.neighbor_fn: Optional[utils.NEIGHBOR_FN] = None
        self.noise_fn: utils.NOISE_FN = None

    @property
    def name(self) -> str:
        return self.cfg.type.name

    @property
    def cfg_string(self) -> str:
        cfg_yaml = OmegaConf.to_yaml(self.cfg, resolve=True, sort_keys=True)
        return str(hashlib.sha1(f"{cfg_yaml}_{self.seed}".encode("utf8")).hexdigest())

    def _get_sample(self,
                    sequence: Union[str, utils.SAMPLE],
                    corrupt_sequence: Optional[Union[str, utils.SAMPLE]] = None,
                    is_inference: bool = False,
                    **kwargs: Any) -> Tuple[utils.SAMPLE, Optional[str]]:
        if corrupt_sequence is not None:
            if isinstance(corrupt_sequence, str):
                return utils.prepare_samples([corrupt_sequence], self.tok_fn, self.neighbor_fn, **kwargs)[0], \
                       str(sequence)
            else:
                return corrupt_sequence, str(sequence)
        elif not is_inference:
            sequence, corrupt_sequence = self.noise_fn(sequence)
            return utils.prepare_samples(
                [corrupt_sequence],
                self.tok_fn,
                self.neighbor_fn,
                **kwargs
            )[0], str(sequence)
        else:
            if isinstance(sequence, str):
                return utils.prepare_samples([sequence], self.tok_fn, self.neighbor_fn, **kwargs)[0], None
            else:
                return sequence, None

    def prepare_sequence(
            self,
            sequence: Union[str, utils.SAMPLE],
            corrupt_sequence: Optional[Union[str, utils.SAMPLE]] = None,
            is_inference: bool = False,
            return_data: bool = False) \
            -> Tuple[
                Union[dgl.DGLHeteroGraph, graph.HeterogeneousDGLData],
                Dict[str, Any]
            ]:
        raise NotImplementedError

    def prepare_sequences_for_inference(
            self,
            sequences: List[str]
    ) -> dgl.DGLHeteroGraph:
        return dgl.batch([
            self.prepare_sequence(s, is_inference=True)[0]
            for s in sequences
        ])


@dataclass
class SEDSequenceConfig(DatasetVariantConfig):
    type: DatasetVariants = DatasetVariants.SED_SEQUENCE
    tokenizer: TokenizerConfig = MISSING
    word_tokenizer: Optional[TokenizerConfig] = None
    noise: Any = MISSING
    dictionary_file: Optional[str] = None
    add_word_features: bool = True
    add_dependency_info: bool = True
    word_fully_connected: bool = True
    token_fully_connected: bool = False
    encoding_scheme: str = "sentence"
    index: Optional[str] = None


class SEDSequence(DatasetVariant):
    def __init__(self, cfg: DatasetVariantConfig, seed: int):
        super().__init__(cfg, seed)
        self.cfg: SEDSequenceConfig
        self.tokenizer = get_tokenizer_from_config(self.cfg.tokenizer)
        if self.cfg.word_tokenizer is not None:
            self.word_tokenizer = get_tokenizer_from_config(self.cfg.word_tokenizer)
        else:
            self.word_tokenizer = None

        if self.cfg.dictionary_file is not None:
            self.dictionary = io.dictionary_from_file(self.cfg.dictionary_file)
        else:
            self.dictionary = None

        if self.cfg.index is not None:
            self.index = index.NNIndex(self.cfg.index)
            self.num_neighbors = int(os.getenv("GNN_LIB_NUM_NEIGHBORS", 5))
            self.neighbor_fn = index.get_neighbor_fn(self.index, self.num_neighbors)
        else:
            self.neighbor_fn = None
            self.num_neighbors = None

        self.noise = get_noise_from_config(self.cfg.noise, self.seed)
        self.noise_fn = self.noise.apply

        self.tok_fn = tokenization.get_tokenization_fn(self.tokenizer)

    def construct_graph(self, sample: utils.SAMPLE) -> graph.HeterogeneousDGLData:
        self.cfg: SEDSequenceConfig
        if self.cfg.encoding_scheme == "tokens":
            data = graph.sequence_to_token_graph(
                sample=sample
            )
        elif self.cfg.encoding_scheme == "words":
            data = graph.sequence_to_word_graph(
                sample=sample,
                tokenizer=self.tokenizer,
                word_tokenizer=self.word_tokenizer,
                dictionary=self.dictionary,
                add_word_features=self.cfg.add_word_features,
                # add_ner_features=True,
                # add_pos_tag_features=True,
                add_dependency_info=self.cfg.add_dependency_info,
                add_num_neighbors=self.num_neighbors,
                word_fully_connected=self.cfg.word_fully_connected,
                token_fully_connected=self.cfg.token_fully_connected,
                scheme="token_to_word"
            )
        elif self.cfg.encoding_scheme == "sentence":
            data = graph.sequence_to_word_graph(
                sample=sample,
                tokenizer=self.tokenizer,
                word_tokenizer=self.word_tokenizer,
                add_sentence_level_node=True,
                dictionary=self.dictionary,
                add_word_features=self.cfg.add_word_features,
                # add_ner_features=True,
                # add_pos_tag_features=True,
                add_dependency_info=self.cfg.add_dependency_info,
                add_num_neighbors=self.num_neighbors,
                word_fully_connected=self.cfg.word_fully_connected,
                token_fully_connected=self.cfg.token_fully_connected,
                scheme="token_to_word"
            )
        else:
            raise ValueError(f"Unknown encoding scheme {self.cfg.encoding_scheme}")
        return data

    def prepare_sequence(
            self,
            sequence: Union[str, utils.SAMPLE],
            corrupt_sequence: Optional[Union[str, utils.SAMPLE]] = None,
            is_inference: bool = False,
            return_data: bool = False) \
            -> Tuple[
                Union[dgl.DGLHeteroGraph, graph.HeterogeneousDGLData],
                Dict[str, Any]
            ]:
        self.cfg: SEDSequenceConfig

        input_sample, target_sequence = self._get_sample(sequence, corrupt_sequence, is_inference)
        input_sample = utils.sanitize_sample(input_sample, unk_token_id=self.tokenizer.token_to_id(tokenization.UNK))
        if not is_inference:
            assert target_sequence is not None
            label = int(str(input_sample) != target_sequence)
        else:
            label = None

        data = self.construct_graph(input_sample)

        return data if return_data else data.to_heterograph(), {"label": label}


@dataclass
class SEDWordsConfig(DatasetVariantConfig):
    type: DatasetVariants = DatasetVariants.SED_WORDS
    tokenizer: TokenizerConfig = MISSING
    word_tokenizer: Optional[TokenizerConfig] = None
    noise: Any = MISSING
    dictionary_file: Optional[str] = None
    add_word_features: bool = True
    add_dependency_info: bool = True
    token_fully_connected: bool = False
    word_fully_connected: bool = True
    encoding_scheme: str = "whitespace_words"
    index: Optional[str] = None


class SEDWords(DatasetVariant):
    def __init__(self, cfg: DatasetVariantConfig, seed: int):
        super().__init__(cfg, seed)
        self.cfg: SEDWordsConfig

        self.tokenizer = get_tokenizer_from_config(self.cfg.tokenizer)
        if self.cfg.word_tokenizer is not None:
            self.word_tokenizer = get_tokenizer_from_config(self.cfg.word_tokenizer)
        else:
            self.word_tokenizer = None

        if self.cfg.dictionary_file is not None:
            self.dictionary = io.dictionary_from_file(self.cfg.dictionary_file)
        else:
            self.dictionary = None

        if self.cfg.index is not None:
            self.index = index.NNIndex(self.cfg.index)
            self.num_neighbors = int(os.getenv("GNN_LIB_NUM_NEIGHBORS", 5))
            self.neighbor_fn = index.get_neighbor_fn(self.index, self.num_neighbors)
        else:
            self.neighbor_fn = None
            self.num_neighbors = None

        self.noise = get_noise_from_config(self.cfg.noise, seed)
        self.noise_fn = self.noise.apply

        self.tok_fn = tokenization.get_tokenization_fn(self.tokenizer)

        self.sample_kwargs = {}
        if self.cfg.encoding_scheme != "tokens":
            self.sample_kwargs["with_dep_parser"] = self.cfg.add_dependency_info

    def construct_graph(self, sample: utils.SAMPLE) -> graph.HeterogeneousDGLData:
        self.cfg: SEDWordsConfig
        if self.cfg.encoding_scheme == "whitespace_words":
            data = graph.sequence_to_word_graph(
                sample=sample,
                tokenizer=self.tokenizer,
                word_tokenizer=self.word_tokenizer,
                add_word_whitespace_groups=True,
                dictionary=self.dictionary,
                add_word_features=self.cfg.add_word_features,
                # add_ner_features=True,
                # add_pos_tag_features=True,
                add_dependency_info=self.cfg.add_dependency_info,
                add_num_neighbors=self.num_neighbors,
                token_fully_connected=self.cfg.token_fully_connected,
                word_fully_connected=self.cfg.word_fully_connected,
                scheme="token_to_word"
            )
        elif self.cfg.encoding_scheme == "tokens":
            data = graph.sequence_to_token_graph(
                sample=sample,
                add_word_whitespace_groups=True
            )
        else:
            raise ValueError(f"Unknown encoding scheme {self.cfg.encoding_scheme}")
        return data

    def prepare_sequence(
            self,
            sequence: Union[str, utils.SAMPLE],
            corrupt_sequence: Optional[Union[str, utils.SAMPLE]] = None,
            is_inference: bool = False,
            return_data: bool = False) \
            -> Tuple[
                Union[dgl.DGLHeteroGraph, graph.HeterogeneousDGLData],
                Dict[str, Any]
            ]:
        self.cfg: SEDWordsConfig

        input_sample, target_sequence = self._get_sample(sequence, corrupt_sequence, is_inference, **self.sample_kwargs)
        input_sample = utils.sanitize_sample(input_sample, unk_token_id=self.tokenizer.token_to_id(tokenization.UNK))
        if not is_inference:
            assert target_sequence is not None
            input_words = str(input_sample).split()
            target_words = target_sequence.split()
            assert len(input_words) == len(target_words), \
                "expected input and target sequence to not differ in whitespaces"
            label = [int(i != t) for i, t in zip(input_words, target_words)]
        else:
            label = None

        data = self.construct_graph(input_sample)

        info = {}
        if not is_inference:
            if self.cfg.encoding_scheme == "tokens":
                info["label"] = {"token": label}
            else:
                info["label"] = {"word": label}

        return data if return_data else data.to_heterograph(), info


@dataclass
class TokenizationRepairConfig(DatasetVariantConfig):
    type: DatasetVariants = DatasetVariants.TOKENIZATION_REPAIR
    noise: Any = MISSING
    encoding_scheme: str = "chars"


class TokenizationRepair(DatasetVariant):
    def __init__(self, cfg: DatasetVariantConfig, seed: int):
        super().__init__(cfg, seed)
        self.cfg: TokenizationRepairConfig
        self.char_tokenizer = tokenization.CharTokenizer()

        self.noise = get_noise_from_config(self.cfg.noise, seed)
        self.noise_fn = self.noise.apply

        self.tok_fn = tokenization.get_tokenization_fn(self.char_tokenizer, True)

    def construct_graph(self, sample: utils.SAMPLE) -> graph.HeterogeneousDGLData:
        self.cfg: TokenizationRepairConfig
        if self.cfg.encoding_scheme == "chars":
            data = graph.sequence_to_token_graph(
                sample=sample
            )
        else:
            raise ValueError(f"Unknown encoding scheme {self.cfg.encoding_scheme}")
        return data

    def prepare_sequence(
            self,
            sequence: Union[str, utils.SAMPLE],
            corrupt_sequence: Union[str, utils.SAMPLE] = None,
            is_inference: bool = False,
            return_data: bool = False) -> Tuple[
        Union[dgl.DGLHeteroGraph, graph.HeterogeneousDGLData],
        Dict[str, Any]
    ]:
        self.cfg: TokenizationRepairConfig

        input_sample, target_sequence = self._get_sample(sequence, corrupt_sequence, is_inference)
        input_sample = utils.sanitize_sample(input_sample,
                                             unk_token_id=self.char_tokenizer.token_to_id(tokenization.UNK))
        if not is_inference:
            assert target_sequence is not None
            # get the whitespace operations to turn input_sample into target_sequence
            label = tokenization_repair.get_whitespace_operations(
                str(input_sample), target_sequence
            )
        else:
            label = None

        data = self.construct_graph(input_sample)

        if not is_inference:
            if self.cfg.encoding_scheme == "chars":
                assert len(label) == data.get_num_nodes()["token"]
                for n in range(data.get_num_nodes()["token"]):
                    data.add_node_data(
                        node_type="token",
                        node=n,
                        feat_dict={"label": label[n]}
                    )
            else:
                raise RuntimeError("should not happen")

        return data if return_data else data.to_heterograph(), {}


@dataclass
class TokenizationRepairNMTConfig(DatasetVariantConfig):
    type: DatasetVariants = DatasetVariants.TOKENIZATION_REPAIR_NMT
    noise: Any = MISSING
    encoding_scheme: str = "chars"


class TokenizationRepairNMT(DatasetVariant):
    def __init__(self, cfg: DatasetVariantConfig, seed: int):
        super().__init__(cfg, seed)
        self.cfg: TokenizationRepairNMTConfig
        self.char_tokenizer = tokenization.CharTokenizer()
        self.tok_tokenizer = tokenization.TokenizationRepairTokenizer()
        self.bos_token_id = self.tok_tokenizer.token_to_id(tokenization.BOS)
        self.eos_token_id = self.tok_tokenizer.token_to_id(tokenization.EOS)
        self.pad_token_id = self.tok_tokenizer.token_to_id(tokenization.PAD)

        self.noise = get_noise_from_config(self.cfg.noise, seed)
        self.noise_fn = self.noise.apply

        self.tok_fn = tokenization.get_tokenization_fn(self.char_tokenizer, True)

    def construct_graph(self, sample: utils.SAMPLE) -> graph.HeterogeneousDGLData:
        self.cfg: TokenizationRepairNMTConfig
        if self.cfg.encoding_scheme == "chars":
            data = graph.sequence_to_token_graph(
                sample=sample
            )
        else:
            raise ValueError(f"Unknown encoding scheme {self.cfg.encoding_scheme}")
        return data

    def prepare_sequence(
            self,
            sequence: str,
            corrupt_sequence: Optional[str] = None,
            is_inference: bool = False,
            return_data: bool = False) \
            -> Tuple[
                Union[dgl.DGLHeteroGraph, graph.HeterogeneousDGLData],
                Dict[str, Any]
            ]:
        input_sample, target_sequence = self._get_sample(sequence, corrupt_sequence, is_inference)
        input_sample = utils.sanitize_sample(input_sample,
                                             unk_token_id=self.char_tokenizer.token_to_id(tokenization.UNK))
        if not is_inference:
            assert target_sequence is not None
            # 0 -> keep, 1 -> insert, 2 -> delete, 3 -> unk, 4 -> bos, 5 -> eos, 6 -> pad
            label = tokenization_repair.get_whitespace_operations(
                str(input_sample), target_sequence
            )
            label = [self.bos_token_id] + label + [self.eos_token_id]
        else:
            label = None

        data = self.construct_graph(input_sample)

        return data if return_data else data.to_heterograph(), {
            "label": label,
            "pad_token_id": self.pad_token_id
        }


@dataclass
class SECWordsNMTConfig(DatasetVariantConfig):
    type: DatasetVariants = DatasetVariants.SEC_WORDS_NMT
    tokenizers: Dict[str, TokenizerConfig] = MISSING
    noise: Any = MISSING
    dictionary_file: Optional[str] = None
    add_word_features: bool = True
    add_dependency_info: bool = True
    token_fully_connected: bool = False
    word_fully_connected: bool = True
    encoding_scheme: str = "whitespace_words"
    index: Optional[str] = None


class SECWordsNMT(DatasetVariant):
    def __init__(self,
                 cfg: DatasetVariantConfig,
                 seed: int):
        super().__init__(cfg, seed)
        self.cfg: SECWordsNMTConfig

        self.input_tokenizer = get_tokenizer_from_config(self.cfg.tokenizers["input_tokenizer"])
        self.output_tokenizer = get_tokenizer_from_config(self.cfg.tokenizers["output_tokenizer"])
        if "word_tokenizer" in self.cfg.tokenizers:
            self.word_tokenizer = get_tokenizer_from_config(self.cfg.tokenizers["word_tokenizer"])
        else:
            self.word_tokenizer = None

        self.pad_token_id = self.output_tokenizer.token_to_id(tokenization.PAD)
        if self.cfg.dictionary_file is not None:
            self.dictionary = io.dictionary_from_file(self.cfg.dictionary_file)
        else:
            self.dictionary = None

        if self.cfg.index is not None:
            self.index = index.NNIndex(self.cfg.index)
            self.num_neighbors = int(os.getenv("GNN_LIB_NUM_NEIGHBORS", 5))
            self.neighbor_fn = index.get_neighbor_fn(self.index, self.num_neighbors)
        else:
            self.neighbor_fn = None
            self.num_neighbors = None

        self.noise = get_noise_from_config(self.cfg.noise, seed)
        self.noise_fn = self.noise.apply

        self.tok_fn = tokenization.get_tokenization_fn(self.input_tokenizer)

        self.sample_kwargs = {}
        if self.cfg.encoding_scheme != "tokens":
            self.sample_kwargs["with_dep_parser"] = self.cfg.add_dependency_info

    def construct_graph(self, sample: utils.SAMPLE) -> graph.HeterogeneousDGLData:
        self.cfg: SECWordsNMTConfig
        if self.cfg.encoding_scheme == "tokens":
            data = graph.sequence_to_token_graph(
                sample=sample,
                add_word_whitespace_groups=True
            )
        elif self.cfg.encoding_scheme == "whitespace_words":
            data = graph.sequence_to_word_graph(
                sample=sample,
                tokenizer=self.input_tokenizer,
                word_tokenizer=self.word_tokenizer,
                add_word_whitespace_groups=True,
                dictionary=self.dictionary,
                add_word_features=self.cfg.add_word_features,
                # add_ner_features=True,
                # add_pos_tag_features=True,
                add_dependency_info=self.cfg.add_dependency_info,
                add_num_neighbors=self.num_neighbors,
                token_fully_connected=self.cfg.token_fully_connected,
                word_fully_connected=self.cfg.word_fully_connected,
                scheme="token_to_word"
            )
        else:
            raise ValueError(f"Unknown encoding scheme {self.cfg.encoding_scheme}")
        return data

    def prepare_sequence(
            self,
            sequence: str,
            corrupt_sequence: Optional[str] = None,
            is_inference: bool = False,
            return_data: bool = False) \
            -> Tuple[
                Union[dgl.DGLHeteroGraph, graph.HeterogeneousDGLData],
                Dict[str, Any]
            ]:
        self.cfg: SECWordsNMTConfig

        input_sample, target_sequence = self._get_sample(sequence, corrupt_sequence, is_inference, **self.sample_kwargs)
        input_sample = utils.sanitize_sample(input_sample,
                                             unk_token_id=self.input_tokenizer.token_to_id(tokenization.UNK))
        if not is_inference:
            assert target_sequence is not None
            input_words = str(input_sample).split()
            target_words = target_sequence.split()
            assert len(input_words) == len(target_words)
            label = [
                self.output_tokenizer.tokenize(word, add_bos_eos=True)
                for word in target_words
            ]
        else:
            label = None

        data = self.construct_graph(input_sample)

        if self.cfg.encoding_scheme == "whitespace_words":
            info = {
                "label": {"word": label},
                "pad_token_id": {"word": self.pad_token_id}
            }
        else:
            info = {
                "label": {"token": label},
                "pad_token_id": {"token": self.pad_token_id}
            }

        return data if return_data else data.to_heterograph(), info


@dataclass
class SECNMTConfig(DatasetVariantConfig):
    type: DatasetVariants = DatasetVariants.SEC_NMT
    tokenizers: Dict[str, TokenizerConfig] = MISSING
    noise: Any = MISSING
    dictionary_file: Optional[str] = None
    add_word_features: bool = True
    add_dependency_info: bool = True
    token_fully_connected: bool = False
    word_fully_connected: bool = True
    self_loops: bool = False
    encoding_scheme: str = "words"
    graph_scheme: str = "token_to_word"
    gnn_decoding: bool = False
    index: Optional[str] = None


class SECNMT(DatasetVariant):
    def __init__(self,
                 cfg: DatasetVariantConfig,
                 seed: int):
        super().__init__(cfg, seed)
        self.cfg: SECNMTConfig
        self.input_tokenizer = get_tokenizer_from_config(self.cfg.tokenizers["input_tokenizer"])
        self.output_tokenizer = get_tokenizer_from_config(self.cfg.tokenizers["output_tokenizer"])
        if "word_tokenizer" in self.cfg.tokenizers:
            self.word_tokenizer = get_tokenizer_from_config(self.cfg.tokenizers["word_tokenizer"])
        else:
            self.word_tokenizer = None

        self.bos_token_id = self.output_tokenizer.token_to_id(tokenization.BOS)
        self.pad_token_id = self.output_tokenizer.token_to_id(tokenization.PAD)
        if self.cfg.dictionary_file is not None:
            self.dictionary = io.dictionary_from_file(self.cfg.dictionary_file)
        else:
            self.dictionary = None

        if self.cfg.index is not None:
            self.index = index.NNIndex(self.cfg.index)
            self.num_neighbors = int(os.getenv("GNN_LIB_NUM_NEIGHBORS", 5))
            self.neighbor_fn = index.get_neighbor_fn(self.index, self.num_neighbors)
        else:
            self.neighbor_fn = None
            self.num_neighbors = None

        self.noise = get_noise_from_config(self.cfg.noise, seed)
        self.noise_fn = self.noise.apply

        self.tok_fn = tokenization.get_tokenization_fn(self.input_tokenizer)

        self.sample_kwargs = {}
        if self.cfg.encoding_scheme != "tokens":
            self.sample_kwargs["with_dep_parser"] = self.cfg.add_dependency_info

    def construct_graph(self, sample: utils.SAMPLE, label: Optional[List[int]] = None) -> graph.HeterogeneousDGLData:
        self.cfg: SECNMTConfig
        if self.cfg.encoding_scheme == "tokens":
            data = graph.sequence_to_token_graph(
                sample=sample
            )
        elif self.cfg.encoding_scheme == "words":
            data = graph.sequence_to_word_graph(
                sample=sample,
                tokenizer=self.input_tokenizer,
                word_tokenizer=self.word_tokenizer,
                dictionary=self.dictionary,
                add_word_features=self.cfg.add_word_features,
                # add_ner_features=True,
                # add_pos_tag_features=True,
                add_dependency_info=self.cfg.add_dependency_info,
                add_num_neighbors=self.num_neighbors,
                token_fully_connected=self.cfg.token_fully_connected,
                word_fully_connected=self.cfg.word_fully_connected,
                scheme=self.cfg.graph_scheme
            )
            # if self.cfg.gnn_decoding:
            #     # decoding is done with a gnn, so add the decoding component of the graph
            #     data = graph.add_graph2seq_decoder_nodes_to_word_graph(
            #         word_graph_data=data,
            #         scheme=self.cfg.graph_scheme,
            #         decoder_input_ids=[self.bos_token_id] if label is None else label[:-1],
            #         decoder_target_ids=None if label is None else label[1:]
            #     )
        else:
            raise ValueError(f"unknown encoding scheme {self.cfg.encoding_scheme}")
        return data

    def prepare_sequence(
            self,
            sequence: str,
            corrupt_sequence: Optional[str] = None,
            is_inference: bool = False,
            return_data: bool = False) \
            -> Tuple[
                Union[dgl.DGLHeteroGraph, graph.HeterogeneousDGLData],
                Dict[str, Any]
            ]:
        self.cfg: SECNMTConfig

        input_sample, target_sequence = self._get_sample(sequence, corrupt_sequence, is_inference, **self.sample_kwargs)
        input_sample = utils.sanitize_sample(input_sample,
                                             unk_token_id=self.input_tokenizer.token_to_id(tokenization.UNK))
        if not is_inference:
            assert target_sequence is not None
            label = self.output_tokenizer.tokenize(target_sequence, add_bos_eos=True)
        else:
            label = None

        data = self.construct_graph(input_sample)

        return data if return_data else data.to_heterograph(), {
            "label": label,
            "pad_token_id": self.pad_token_id
        }


def get_variant_from_config(
        cfg: omegaconf.DictConfig,
        seed: int
) -> DatasetVariant:
    variant_type = DatasetVariants[cfg.type]
    if variant_type == DatasetVariants.SED_SEQUENCE:
        cfg = OmegaConf.structured(SEDSequenceConfig(**cfg))
        return SEDSequence(cfg, seed)
    elif variant_type == DatasetVariants.SED_WORDS:
        cfg = OmegaConf.structured(SEDWordsConfig(**cfg))
        return SEDWords(cfg, seed)
    elif variant_type == DatasetVariants.TOKENIZATION_REPAIR:
        cfg = OmegaConf.structured(TokenizationRepairConfig(**cfg))
        return TokenizationRepair(cfg, seed)
    elif variant_type == DatasetVariants.TOKENIZATION_REPAIR_NMT:
        cfg = OmegaConf.structured(TokenizationRepairNMTConfig(**cfg))
        return TokenizationRepairNMT(cfg, seed)
    elif variant_type == DatasetVariants.SEC_WORDS_NMT:
        cfg = OmegaConf.structured(SECWordsNMTConfig(**cfg))
        return SECWordsNMT(cfg, seed)
    elif variant_type == DatasetVariants.SEC_NMT:
        cfg = OmegaConf.structured(SECNMTConfig(**cfg))
        return SECNMT(cfg, seed)
    else:
        raise ValueError(f"Unknown variant {cfg.type.name}")
