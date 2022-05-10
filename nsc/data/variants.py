import enum
import hashlib
from dataclasses import dataclass
from typing import Tuple, Union, Optional, Dict, Any, List

import dgl
import numpy as np
import omegaconf
import torch
from omegaconf import MISSING, OmegaConf

from nsc.data import graph, tokenization, utils, index
from nsc.data.tokenization import TokenizerConfig, get_tokenizer_from_config
from nsc.utils import tokenization_repair, io, DataInput


class DatasetVariants(enum.IntEnum):
    SED_SEQUENCE = 1
    SED_WORDS = 2
    TOKENIZATION_REPAIR = 3
    SEC_WORDS_NMT = 4
    SEC_NMT = 5
    TOKENIZATION_REPAIR_PLUS = 6


@dataclass
class DatasetVariantConfig:
    type: DatasetVariants  # = MISSING


class DatasetVariant:
    def __init__(
            self,
            cfg: DatasetVariantConfig,
            seed: int,
            tokenization_fn: utils.TokenizationFn,
            unk_token_id: int,
            neighbor_fn: Optional[utils.NeighborFn] = None,
            **prepare_samples_kwargs: Dict[str, Any]
    ):
        self.cfg = cfg
        self.seed = seed
        self.rand = np.random.default_rng(seed)

        self.tokenization_fn = tokenization_fn
        self.neighbor_fn = neighbor_fn
        self.prepare_samples_kwargs = prepare_samples_kwargs

        self.unk_token_id = unk_token_id

    @property
    def name(self) -> str:
        return self.cfg.type.name

    def get_samples(
            self,
            sequences: List[Union[str, utils.Sample]]
    ) -> List[utils.Sample]:
        output_samples: List[utils.Sample] = [None for _ in range(len(sequences))]
        sequences_to_prepare = []
        indices = []
        for i, sequence in enumerate(sequences):
            if isinstance(sequence, utils.Sample):
                output_samples[i] = sequence
            else:
                sequences_to_prepare.append(sequence)
                indices.append(i)
        samples = utils.prepare_samples(
            sequences_to_prepare,
            [{} for _ in range(len(sequences_to_prepare))],
            self.tokenization_fn,
            self.neighbor_fn,
            **self.prepare_samples_kwargs
        )
        for idx, sample in zip(indices, samples):
            output_samples[idx] = sample

        return [utils.sanitize_sample(sample, self.unk_token_id) for sample in output_samples]

    def get_sample(
            self,
            sequence: str
    ) -> utils.Sample:
        return self.get_samples([sequence])[0]

    def get_inputs(
            self,
            sequence: Union[str, utils.Sample],
            target_sequence: Optional[str] = None,
            is_inference: bool = False
    ) -> Tuple[DataInput, Dict[str, Any]]:
        raise NotImplementedError


@dataclass
class SEDSequenceConfig(DatasetVariantConfig):
    type: DatasetVariants = DatasetVariants.SED_SEQUENCE

    tokenizer: TokenizerConfig = MISSING

    data_scheme: str = "tensor"

    add_word_features: bool = True
    dictionary_file: Optional[str] = None

    # options for data_scheme not in {token_graph, tensor}
    index: Optional[str] = None
    index_num_neighbors: int = 5
    add_edit_distance_neighbors: bool = False
    add_dependency_info: bool = True
    word_fully_connected: bool = True
    token_fully_connected: bool = False


class SEDSequence(DatasetVariant):
    def __init__(self, cfg: DatasetVariantConfig, seed: int):
        cfg: SEDSequenceConfig
        self.tokenizer = get_tokenizer_from_config(cfg.tokenizer)

        unk_token_id = self.tokenizer.token_to_id(tokenization.UNK)

        if cfg.dictionary_file:
            self.dictionary = io.dictionary_from_file(cfg.dictionary_file)
        else:
            self.dictionary = None

        tok_fn = tokenization.get_tokenization_fn(self.tokenizer)
        if cfg.index is not None:
            neighbor_index = index.NNIndex(cfg.index)
            neighbor_fn = index.get_neighbor_fn(neighbor_index, cfg.index_num_neighbors)
        else:
            neighbor_fn = None
        prepare_sample_kwargs = {}
        if cfg.data_scheme == "word_graph":
            prepare_sample_kwargs["with_dep_parser"] = cfg.add_dependency_info

        super().__init__(cfg, seed, tok_fn, unk_token_id, neighbor_fn, **prepare_sample_kwargs)

    def _construct_input(self, sample: utils.Sample) -> Union[torch.Tensor, dgl.DGLHeteroGraph]:
        self.cfg: SEDSequenceConfig
        if self.cfg.data_scheme == "token_graph":
            return graph.sequence_to_token_graph(
                sample=sample
            ).to_heterograph()
        elif self.cfg.data_scheme == "word_graph":
            return graph.sequence_to_word_graph(
                sample=sample,
                tokenizer=self.tokenizer,
                dictionary=self.dictionary,
                add_word_features=self.cfg.add_word_features,
                # add_ner_features=True,
                # add_pos_tag_features=True,
                add_dependency_info=self.cfg.add_dependency_info,
                add_num_neighbors=self.cfg.index_num_neighbors if self.cfg.index is not None else None,
                word_fully_connected=self.cfg.word_fully_connected,
                token_fully_connected=self.cfg.token_fully_connected,
                scheme="token_to_word"
            ).to_heterograph()
        elif self.cfg.data_scheme == "sequence_graph":
            return graph.sequence_to_word_graph(
                sample=sample,
                tokenizer=self.tokenizer,
                add_sequence_level_node=True,
                dictionary=self.dictionary,
                add_word_features=self.cfg.add_word_features,
                # add_ner_features=True,
                # add_pos_tag_features=True,
                add_dependency_info=self.cfg.add_dependency_info,
                add_num_neighbors=self.cfg.index_num_neighbors if self.cfg.index is not None else None,
                word_fully_connected=self.cfg.word_fully_connected,
                token_fully_connected=self.cfg.token_fully_connected,
                scheme="token_to_word"
            ).to_heterograph()
        elif self.cfg.data_scheme == "tensor":
            return torch.tensor(
                utils.flatten(sample.tokens),
                dtype=torch.long
            )
        else:
            raise ValueError(f"unknown data scheme {self.cfg.data_scheme}")

    def get_inputs(
            self,
            sequence: Union[str, utils.Sample],
            target_sequence: Optional[Union[str, utils.Sample]] = None,
            is_inference: bool = False
    ) -> Tuple[
        Union[dgl.DGLHeteroGraph, graph.HeterogeneousDGLData],
        Dict[str, Any]
    ]:
        self.cfg: SEDSequenceConfig

        input_sample = self.get_sample(sequence)

        info = {}
        if not is_inference:
            label = int(str(input_sample) != target_sequence)
            info["label"] = torch.tensor([label], dtype=torch.long)

        if self.cfg.data_scheme == "word_graph":
            info["groups"] = {
                "word": [
                    {
                        "stage": "word_to_sequence",
                        "groups": utils.get_word_and_sequence_groups(input_sample)[1]
                    }
                ]
            }

        elif self.cfg.data_scheme == "token_graph":
            if self.cfg.add_word_features:
                # if we add word features, we first aggregate from tokens to words
                # and then from word to whitespace_words
                # we need to do this because the features are on the word level and not on the whitespace word level
                word_groups, sequence_groups = utils.get_word_and_sequence_groups(input_sample)
                info["groups"] = {
                    "token": [
                        {
                            "stage": "token_to_word",
                            "groups": word_groups,
                        },
                        {
                            "stage": "word_to_sequence",
                            "groups": sequence_groups,
                            "features": utils.get_word_features(input_sample.doc, self.dictionary)
                        }
                    ]
                }
            else:
                info["groups"] = {
                    "token": [
                        {
                            "stage": "token_to_sequence",
                            "groups": utils.get_sequence_groups(input_sample)
                        }
                    ]
                }

        elif self.cfg.data_scheme == "sequence_graph":
            # sequence graph does not need groups or features since all word features and the sequence aggregation are
            # already in the graph itself
            pass

        elif self.cfg.data_scheme == "tensor":
            if self.cfg.add_word_features:
                # first aggregate from tokens to words, add word features and then aggregate from words to
                # sequence
                word_groups, sequence_groups = utils.get_word_and_sequence_groups(input_sample)
                info["groups"] = [
                    {
                        "stage": "token_to_word",
                        "groups": word_groups
                    },
                    {
                        # features are added before grouping so this is the correct stage
                        "features": utils.get_word_features(input_sample.doc, self.dictionary),
                        "stage": "word_to_sequence",
                        "groups": sequence_groups
                    }
                ]
            else:
                # just aggregate all tokens into a sequence representation
                info["groups"] = [
                    {
                        "stage": "token_to_sequence",
                        "groups": utils.get_sequence_groups(input_sample)
                    }
                ]

        else:
            raise RuntimeError(f"unknown data scheme {self.cfg.data_scheme}")

        return self._construct_input(input_sample), info


@dataclass
class SEDWordsConfig(DatasetVariantConfig):
    type: DatasetVariants = DatasetVariants.SED_WORDS

    tokenizer: TokenizerConfig = MISSING

    data_scheme: str = "tensor"

    dictionary_file: Optional[str] = None
    add_word_features: bool = True

    # special args for encoding scheme whitespace_words
    index: Optional[str] = None
    index_num_neighbors: int = 5
    add_dependency_info: bool = True
    token_fully_connected: bool = False
    word_fully_connected: bool = True


class SEDWords(DatasetVariant):
    def __init__(self, cfg: DatasetVariantConfig, seed: int):
        cfg: SEDWordsConfig
        self.tokenizer = get_tokenizer_from_config(cfg.tokenizer)

        unk_token_id = self.tokenizer.token_to_id(tokenization.UNK)

        if cfg.dictionary_file:
            self.dictionary = io.dictionary_from_file(cfg.dictionary_file)
        else:
            self.dictionary = None

        tok_fn = tokenization.get_tokenization_fn(self.tokenizer)
        if cfg.index is not None:
            neighbor_index = index.NNIndex(cfg.index)
            neighbor_fn = index.get_neighbor_fn(neighbor_index, cfg.index_num_neighbors)
        else:
            neighbor_fn = None

        prepare_sample_kwargs = {}
        if cfg.data_scheme == "word_graph":
            prepare_sample_kwargs["with_dep_parser"] = cfg.add_dependency_info

        super().__init__(cfg, seed, tok_fn, unk_token_id, neighbor_fn, **prepare_sample_kwargs)

    def _construct_input(self, sample: utils.Sample) -> Union[torch.Tensor, dgl.DGLHeteroGraph]:
        self.cfg: SEDWordsConfig
        if self.cfg.data_scheme == "token_graph":
            return graph.sequence_to_token_graph(
                sample=sample
            ).to_heterograph()
        elif self.cfg.data_scheme == "word_graph":
            return graph.sequence_to_word_graph(
                sample=sample,
                tokenizer=self.tokenizer,
                dictionary=self.dictionary,
                add_word_features=self.cfg.add_word_features,
                # add_ner_features=True,
                # add_pos_tag_features=True,
                add_dependency_info=self.cfg.add_dependency_info,
                add_num_neighbors=self.cfg.index_num_neighbors if self.cfg.index is not None else None,
                word_fully_connected=self.cfg.word_fully_connected,
                token_fully_connected=self.cfg.token_fully_connected,
                scheme="token_to_word"
            ).to_heterograph()
        elif self.cfg.data_scheme == "tensor":
            return torch.tensor(
                utils.flatten(sample.tokens),
                dtype=torch.long
            )
        else:
            raise ValueError(f"unknown data scheme {self.cfg.data_scheme}")

    def get_inputs(
            self,
            sequence: Union[str, utils.Sample],
            target_sequence: Optional[Union[str, utils.Sample]] = None,
            is_inference: bool = False
    ) -> Tuple[
        Union[dgl.DGLHeteroGraph, graph.HeterogeneousDGLData],
        Dict[str, Any]
    ]:
        self.cfg: SEDWordsConfig

        input_sample = self.get_sample(sequence)

        info: Dict[str, Any] = {}
        if not is_inference:
            input_words = str(input_sample).split()
            target_words = target_sequence.split()
            assert len(input_words) == len(target_words), \
                "expected input and target sequence to not differ in whitespaces"
            label = torch.tensor([int(i != t) for i, t in zip(input_words, target_words)], dtype=torch.long)
        else:
            label = None

        if self.cfg.data_scheme == "word_graph":
            info["groups"] = {
                "word": [
                    {
                        "stage": "word_to_word_ws",
                        "groups": utils.get_word_and_word_whitespace_groups(input_sample)[1]
                    }
                ]
            }
            if label is not None:
                info["label"] = {
                    "word": label
                }

        elif self.cfg.data_scheme == "token_graph":
            if self.cfg.add_word_features:
                # if we add word features, we first aggregate from tokens to words
                # and then from words to whitespace_words
                # we need to do this because the features are on the word level and not on the whitespace word level
                word_groups, word_whitespace_groups = utils.get_word_and_word_whitespace_groups(input_sample)
                info["groups"] = {
                    "token": [
                        {
                            "stage": "token_to_word",
                            "groups": word_groups,
                        },
                        {
                            "stage": "word_to_word_ws",
                            "groups": word_whitespace_groups,
                            "features": utils.get_word_features(input_sample.doc, self.dictionary)
                        }
                    ]
                }
            else:
                info["groups"] = {
                    "token": [
                        {
                            "stage": "token_to_word_ws",
                            "groups": utils.get_word_whitespace_groups(input_sample)
                        }
                    ]
                }
            if label is not None:
                info["label"] = {
                    "token": label
                }

        elif self.cfg.data_scheme == "tensor":
            if self.cfg.add_word_features:
                # first aggregate from tokens to words, add word features and then aggregate from words to
                # whitespace words
                word_groups, word_ws_groups = utils.get_word_and_word_whitespace_groups(input_sample)
                info["groups"] = [
                    {
                        "stage": "token_to_word",
                        "groups": word_groups
                    },
                    {
                        # features are added before grouping so this is the correct stage
                        "features": utils.get_word_features(input_sample.doc, self.dictionary),
                        "stage": "word_to_word_ws",
                        "groups": word_ws_groups
                    }
                ]
            else:
                # just aggregate all tokens into whitespace word representations
                info["groups"] = [
                    {
                        "stage": "token_to_word_ws",
                        "groups": utils.get_word_whitespace_groups(input_sample)
                    }
                ]
            if label is not None:
                info["label"] = label

        else:
            raise RuntimeError(f"unknown data scheme {self.cfg.data_scheme}")

        return self._construct_input(input_sample), info


@dataclass
class TokenizationRepairConfig(DatasetVariantConfig):
    type: DatasetVariants = DatasetVariants.TOKENIZATION_REPAIR

    data_scheme: str = "tensor"

    input_type: str = "char"  # one of {char, byte}
    add_bos_eos: bool = False


class TokenizationRepair(DatasetVariant):
    def __init__(self, cfg: DatasetVariantConfig, seed: int):
        cfg: TokenizationRepairConfig

        if cfg.input_type == "char":
            self.tokenizer = tokenization.CharTokenizer()
        elif cfg.input_type == "byte":
            self.tokenizer = tokenization.ByteTokenizer()
        else:
            raise ValueError(f"unknown input type {cfg.input_type}, must be one of {{char, byte}}")

        self.bos_token_id = self.tokenizer.token_to_id(tokenization.BOS)
        self.eos_token_id = self.tokenizer.token_to_id(tokenization.EOS)
        unk_token_id = self.tokenizer.token_to_id(tokenization.UNK)
        tok_fn = tokenization.get_tokenization_fn(self.tokenizer)

        super().__init__(cfg, seed, tok_fn, unk_token_id)

    def _construct_input(self, sample: utils.Sample) -> Union[dgl.DGLHeteroGraph, torch.Tensor]:
        self.cfg: TokenizationRepairConfig
        if self.cfg.data_scheme == "tensor":
            token_ids = utils.flatten(sample.tokens)
            if self.cfg.add_bos_eos:
                token_ids = [self.bos_token_id] + token_ids + [self.eos_token_id]
            return torch.tensor(token_ids, dtype=torch.long)
        else:
            raise ValueError(f"Unknown data scheme {self.cfg.data_scheme}")

    def get_inputs(
            self,
            sequence: Union[str, utils.Sample],
            target_sequence: Optional[str] = None,
            is_inference: bool = False
    ) -> Tuple[
        Union[dgl.DGLHeteroGraph, graph.HeterogeneousDGLData],
        Dict[str, Any]
    ]:
        self.cfg: TokenizationRepairConfig

        input_sample = self.get_sample(sequence)

        info = {}
        if not is_inference:
            # get the whitespace operations to turn input_sample into target_sequence
            label = tokenization_repair.get_whitespace_operations(
                str(input_sample), target_sequence
            )
            info["label"] = torch.tensor(label, dtype=torch.long)

        return self._construct_input(input_sample), info


@dataclass
class SECWordsNMTConfig(DatasetVariantConfig):
    type: DatasetVariants = DatasetVariants.SEC_WORDS_NMT

    input_tokenizer: TokenizerConfig = MISSING
    output_tokenizer: TokenizerConfig = MISSING

    data_scheme: str = "tensor"  # one of {token_graph, word_graph, tensor}

    dictionary_file: Optional[str] = None
    add_word_features: bool = True

    # specific stuff for data_scheme word_graph
    add_dependency_info: bool = False
    token_fully_connected: bool = False
    word_fully_connected: bool = True
    graph_scheme: str = "token_to_word"
    index: Optional[str] = None
    index_num_neighbors: int = 5


class SECWordsNMT(DatasetVariant):
    def __init__(self,
                 cfg: DatasetVariantConfig,
                 seed: int):
        cfg: SECWordsNMTConfig
        self.input_tokenizer = get_tokenizer_from_config(cfg.input_tokenizer)
        self.output_tokenizer = get_tokenizer_from_config(cfg.output_tokenizer)

        unk_token_id = self.input_tokenizer.token_to_id(tokenization.UNK)
        self.output_pad_token_id = self.output_tokenizer.token_to_id(tokenization.PAD)

        if cfg.dictionary_file:
            self.dictionary = io.dictionary_from_file(cfg.dictionary_file)
        else:
            self.dictionary = None

        tok_fn = tokenization.get_tokenization_fn(self.input_tokenizer)
        if cfg.index is not None:
            neighbor_index = index.NNIndex(cfg.index)
            neighbor_fn = index.get_neighbor_fn(neighbor_index, cfg.index_num_neighbors)
        else:
            neighbor_fn = None
        prepare_sample_kwargs = {}
        if cfg.data_scheme == "word_graph":
            prepare_sample_kwargs["with_dep_parser"] = cfg.add_dependency_info

        super().__init__(cfg, seed, tok_fn, unk_token_id, neighbor_fn, **prepare_sample_kwargs)

    def _construct_input(
            self,
            sample: utils.Sample
    ) -> Union[torch.Tensor, dgl.DGLHeteroGraph]:
        self.cfg: SECWordsNMTConfig
        if self.cfg.data_scheme == "token_graph":
            return graph.sequence_to_token_graph(
                sample=sample
            ).to_heterograph()
        elif self.cfg.data_scheme == "word_graph":
            return graph.sequence_to_word_graph(
                sample=sample,
                tokenizer=self.input_tokenizer,
                dictionary=self.dictionary,
                add_word_features=self.cfg.add_word_features,
                add_dependency_info=self.cfg.add_dependency_info,
                add_num_neighbors=self.cfg.index_num_neighbors if self.cfg.index is not None else None,
                token_fully_connected=self.cfg.token_fully_connected,
                word_fully_connected=self.cfg.word_fully_connected,
                scheme=self.cfg.graph_scheme
            ).to_heterograph()
        elif self.cfg.data_scheme == "tensor":
            return torch.tensor(
                utils.flatten(sample.tokens),
                dtype=torch.long
            )
        else:
            raise ValueError(f"unknown data scheme {self.cfg.data_scheme}")

    def get_inputs(
            self,
            sequence: str,
            target_sequence: Optional[str] = None,
            is_inference: bool = False
    ) -> Tuple[
        Union[torch.Tensor, dgl.DGLHeteroGraph],
        Dict[str, Any]
    ]:
        self.cfg: SECWordsNMTConfig

        input_sample = self.get_sample(sequence)

        input_words = str(input_sample).split()
        token_group_lengths = [0] * len(input_words)
        word_ws_group_lengths = [0] * len(input_words)
        word_ws_idx = 0
        for tokens, word in zip(input_sample.tokens, input_sample.doc):
            token_group_lengths[word_ws_idx] += len(tokens)
            word_ws_group_lengths[word_ws_idx] += 1
            if word.whitespace_ == " ":
                word_ws_idx += 1
        token_group_lengths = torch.tensor(token_group_lengths, dtype=torch.long)
        word_ws_group_lengths = torch.tensor(word_ws_group_lengths, dtype=torch.long)

        info: Dict[str, Any] = {}
        if not is_inference:
            target_words = target_sequence.split()
            assert len(input_words) == len(target_words)
            label = [
                self.output_tokenizer.tokenize(word, add_bos_eos=True)
                for word in target_words
            ]
        else:
            label = None

        if self.cfg.data_scheme == "tensor":
            info["encoder_group_lengths"] = token_group_lengths
            if label is not None:
                info["label"] = label
                info["pad_token_id"] = self.output_pad_token_id
        elif self.cfg.data_scheme == "token_graph":
            info["encoder_group_lengths"] = {"token": {"token": token_group_lengths}}
            if label is not None:
                info["label"] = {"token": label}
                info["pad_token_id"] = {"token": self.output_pad_token_id}
        elif self.cfg.data_scheme == "word_graph":
            info["encoder_group_lengths"] = {"word": {"token": token_group_lengths, "word": word_ws_group_lengths}}
            if label is not None:
                info["label"] = {"word": label}
                info["pad_token_id"] = {"word": self.output_pad_token_id}
        else:
            raise RuntimeError(f"unknown data scheme {self.cfg.data_scheme}")

        return self._construct_input(input_sample), info


@dataclass
class SECNMTConfig(DatasetVariantConfig):
    type: DatasetVariants = DatasetVariants.SEC_NMT

    input_tokenizer: TokenizerConfig = MISSING
    output_tokenizer: TokenizerConfig = MISSING

    data_scheme: str = "tensor"  # one of {token_graph, word_graph, tensor}

    dictionary_file: Optional[str] = None
    add_word_features: bool = True

    # specific stuff for data_scheme word_graph
    add_dependency_info: bool = False
    token_fully_connected: bool = False
    word_fully_connected: bool = True
    graph_scheme: str = "token_to_word"
    index: Optional[str] = None
    index_num_neighbors: int = 5


class SECNMT(DatasetVariant):
    def __init__(self,
                 cfg: DatasetVariantConfig,
                 seed: int):
        cfg: SECNMTConfig
        self.input_tokenizer = get_tokenizer_from_config(cfg.input_tokenizer)
        self.output_tokenizer = get_tokenizer_from_config(cfg.output_tokenizer)

        unk_token_id = self.input_tokenizer.token_to_id(tokenization.UNK)
        self.output_pad_token_id = self.output_tokenizer.token_to_id(tokenization.PAD)

        if cfg.dictionary_file:
            self.dictionary = io.dictionary_from_file(cfg.dictionary_file)
        else:
            self.dictionary = None

        tok_fn = tokenization.get_tokenization_fn(self.input_tokenizer)
        if cfg.index is not None:
            neighbor_index = index.NNIndex(cfg.index)
            neighbor_fn = index.get_neighbor_fn(neighbor_index, cfg.index_num_neighbors)
        else:
            neighbor_fn = None
        prepare_sample_kwargs = {}
        if cfg.data_scheme == "word_graph":
            prepare_sample_kwargs["with_dep_parser"] = cfg.add_dependency_info

        super().__init__(cfg, seed, tok_fn, unk_token_id, neighbor_fn, **prepare_sample_kwargs)

    def _construct_input(
            self,
            sample: utils.Sample
    ) -> Union[torch.Tensor, dgl.DGLHeteroGraph]:
        self.cfg: SECNMTConfig
        if self.cfg.data_scheme == "token_graph":
            return graph.sequence_to_token_graph(
                sample=sample
            ).to_heterograph()
        elif self.cfg.data_scheme == "word_graph":
            return graph.sequence_to_word_graph(
                sample=sample,
                tokenizer=self.input_tokenizer,
                dictionary=self.dictionary,
                add_word_features=self.cfg.add_word_features,
                add_dependency_info=self.cfg.add_dependency_info,
                add_num_neighbors=self.cfg.index_num_neighbors if self.cfg.index is not None else None,
                token_fully_connected=self.cfg.token_fully_connected,
                word_fully_connected=self.cfg.word_fully_connected,
                scheme=self.cfg.graph_scheme
            ).to_heterograph()
        elif self.cfg.data_scheme == "tensor":
            return torch.tensor(
                utils.flatten(sample.tokens),
                dtype=torch.long
            )
        else:
            raise ValueError(f"unknown data scheme {self.cfg.data_scheme}")

    def get_inputs(
            self,
            sequence: str,
            target_sequence: Optional[str] = None,
            is_inference: bool = False
    ) -> Tuple[
        Union[torch.Tensor, dgl.DGLHeteroGraph],
        Dict[str, Any]
    ]:
        self.cfg: SECNMTConfig

        input_sample = self.get_sample(sequence)

        info = {}
        if not is_inference:
            info["label"] = self.output_tokenizer.tokenize(target_sequence, add_bos_eos=True)
            info["pad_token_id"] = self.output_pad_token_id

        return self._construct_input(input_sample), info


@dataclass
class TokenizationRepairPlusConfig(TokenizationRepairConfig):
    type: DatasetVariants = DatasetVariants.TOKENIZATION_REPAIR_PLUS

    # one of {tokenization_repair_plus_sed, tokenization_repair_plus_sed_plus_sec}
    output_type: str = "tokenization_repair_plus_sed"
    # whether to tokenization repair is considered to be fixed (no labels and thereby no tokenization repair loss
    # and gradients will be calculated during training, should only be used together
    # with fix_tokenization_repair in model)
    fix_tokenization_repair: bool = False

    dictionary_file: Optional[str] = None
    add_word_features: bool = True

    # for tokenization_repair_plus_sed_plus_sec
    sec_tokenizer: Optional[TokenizerConfig] = None


class TokenizationRepairPlus(TokenizationRepair):
    def __init__(self, cfg: DatasetVariantConfig, seed: int):
        super().__init__(cfg, seed)
        self.cfg: TokenizationRepairPlusConfig
        assert self.cfg.output_type in {"tokenization_repair_plus_sed", "tokenization_repair_plus_sed_plus_sec"}

        if self.cfg.input_type != "char":
            raise NotImplementedError("only input type char is currently implemented for tokenization repair plus")

        if self.cfg.dictionary_file:
            self.dictionary = io.dictionary_from_file(self.cfg.dictionary_file)
        else:
            self.dictionary = None

        if self.cfg.sec_tokenizer is not None:
            self.sec_tokenizer = get_tokenizer_from_config(self.cfg.sec_tokenizer)
            self.sec_pad_token_id = self.sec_tokenizer.token_to_id(tokenization.PAD)
        else:
            self.sec_tokenizer = None

    def get_inputs(
            self,
            sequence: Union[str, utils.Sample],
            target_sequence: Optional[str] = None,
            is_inference: bool = False
    ) -> Tuple[
        Union[dgl.DGLHeteroGraph, graph.HeterogeneousDGLData],
        Dict[str, Any]
    ]:
        self.cfg: TokenizationRepairPlusConfig

        input_sample = self.get_sample(sequence)

        info = {"add_bos_eos": self.cfg.add_bos_eos}
        if not is_inference:
            if not self.cfg.fix_tokenization_repair:
                # get the whitespace operations to turn input_sample into target_sequence
                tokenization_repair_label = tokenization_repair.get_whitespace_operations(
                    str(input_sample), target_sequence
                )
                if self.cfg.add_bos_eos:
                    # -100 is the default ignore index for pytorch cross entropy loss
                    tokenization_repair_label = [-100] + tokenization_repair_label + [-100]
                info["tokenization_repair_label"] = torch.tensor(tokenization_repair_label, dtype=torch.long)

            org_words = input_sample.info.get("org_sequence", target_sequence).split()
            target_words = target_sequence.split()
            assert len(target_words) == len(org_words)

            repaired_words, repaired_doc = utils.tokenize_words(target_sequence, return_doc=True)
            word_groups = utils.get_character_groups_from_repaired_doc(list(str(input_sample)), repaired_doc)
            if self.cfg.add_bos_eos:
                # count bos to first group and eos to last group
                word_groups = [word_groups[0]] + word_groups + [word_groups[-1]]

            info["word_groups"] = {
                "stage": "char_to_word",
                "groups": torch.tensor(word_groups, dtype=torch.long)
            }

            if self.cfg.add_word_features:
                info["word_features"] = utils.get_word_features(repaired_doc, self.dictionary)

            input_group_lengths = [0] * len(target_words)
            word_group_lengths = [0] * len(target_words)
            word_ws_groups = []
            word_ws_idx = 0
            for word in repaired_doc:
                input_group_lengths[word_ws_idx] += len(word.text)
                word_group_lengths[word_ws_idx] += 1
                word_ws_groups.append(word_ws_idx)
                if word.whitespace_ == " ":
                    word_ws_idx += 1
            if self.cfg.add_bos_eos:
                # count bos to first group and eos to last group
                input_group_lengths[0] += 1
                input_group_lengths[-1] += 1

            info["word_ws_groups"] = [
                {
                    "stage": "word_to_word_ws",
                    "groups": torch.tensor(word_ws_groups, dtype=torch.long)
                }
            ]

            info["word_group_lengths"] = torch.tensor(word_group_lengths, dtype=torch.long)
            info["input_group_lengths"] = torch.tensor(input_group_lengths, dtype=torch.long)

            info["sed_label"] = torch.tensor(
                [
                    int(org_word != target_word)
                    for org_word, target_word in zip(org_words, target_words)
                ], dtype=torch.long
            )

            if self.cfg.output_type == "tokenization_repair_plus_sed_plus_sec":
                assert self.sec_tokenizer is not None, "sec tokenizer must be specified to use sec output"
                label = [
                    self.sec_tokenizer.tokenize(word, add_bos_eos=True)
                    for word in org_words
                ]
                info["sec_label"] = label
                info["sec_pad_token_id"] = self.sec_pad_token_id

        return self._construct_input(input_sample), info


def get_variant_from_config(
        cfg: Union[DatasetVariantConfig, omegaconf.DictConfig],
        seed: int
) -> DatasetVariant:
    # explicitly convert ot dict config first, this way we support both dictconfigs
    # and structured configs as input
    cfg: omegaconf.DictConfig = omegaconf.DictConfig(cfg)
    variant_type = DatasetVariants[cfg.type] if isinstance(cfg.type, str) else cfg.type
    if variant_type == DatasetVariants.SED_SEQUENCE:
        cfg = OmegaConf.structured(SEDSequenceConfig(**cfg))
        return SEDSequence(cfg, seed)
    elif variant_type == DatasetVariants.SED_WORDS:
        cfg = OmegaConf.structured(SEDWordsConfig(**cfg))
        return SEDWords(cfg, seed)
    elif variant_type == DatasetVariants.TOKENIZATION_REPAIR:
        cfg = OmegaConf.structured(TokenizationRepairConfig(**cfg))
        return TokenizationRepair(cfg, seed)
    elif variant_type == DatasetVariants.SEC_WORDS_NMT:
        cfg = OmegaConf.structured(SECWordsNMTConfig(**cfg))
        return SECWordsNMT(cfg, seed)
    elif variant_type == DatasetVariants.SEC_NMT:
        cfg = OmegaConf.structured(SECNMTConfig(**cfg))
        return SECNMT(cfg, seed)
    elif variant_type == DatasetVariants.TOKENIZATION_REPAIR_PLUS:
        cfg = OmegaConf.structured(TokenizationRepairPlusConfig(**cfg))
        return TokenizationRepairPlus(cfg, seed)
    else:
        raise ValueError(f"Unknown variant {cfg.type.name}")
