import enum
import hashlib
import os
from dataclasses import dataclass
from typing import Tuple, List, Union, Optional, Dict, Any

import dgl
import numpy as np
import omegaconf
import torch
from omegaconf import MISSING, OmegaConf

from gnn_lib.data import graph, tokenization, utils, index
from gnn_lib.data.preprocessing import get_preprocessing_from_config, PreprocessingConfig, get_preprocessing_fn
from gnn_lib.data.tokenization import TokenizerConfig, get_tokenizer_from_config
from gnn_lib.utils import tokenization_repair, io, BATCH, DATA_INPUT


class DatasetVariants(enum.IntEnum):
    SED_SEQUENCE = 1
    SED_WORDS = 2
    TOKENIZATION_REPAIR = 3
    SEC_WORDS_NMT = 4
    SEC_NMT = 5
    SED_PLUS_SEC = 6
    TOKENIZATION_REPAIR_PLUS_SED = 6
    TOKENIZATION_REPAIR_PLUS_SED_PLUS_SEC = 7


@dataclass
class DatasetVariantConfig:
    type: DatasetVariants  # = MISSING

    preprocessing: PreprocessingConfig = MISSING


class DatasetVariant:
    def __init__(
            self,
            cfg: DatasetVariantConfig,
            seed: int,
            tokenization_fn: utils.TOKENIZATION_FN,
            neighbor_fn: Optional[utils.NEIGHBOR_FN] = None,
            **preprocessing_kwargs: Dict[str, Any]
    ):
        self.cfg = cfg
        self.seed = seed
        self.rand = np.random.default_rng(seed)

        preprocessing = get_preprocessing_from_config(self.cfg.preprocessing, self.seed)
        self.preprocessing_fn: utils.PREPROCESSING_FN = get_preprocessing_fn(
            preprocessing,
            tokenization_fn,
            neighbor_fn,
            **preprocessing_kwargs
        )

    @property
    def name(self) -> str:
        return self.cfg.type.name

    @property
    def cfg_string(self) -> str:
        cfg_yaml = OmegaConf.to_yaml(self.cfg, resolve=True, sort_keys=True)
        return str(hashlib.sha1(f"{cfg_yaml}_{self.seed}".encode("utf8")).hexdigest())

    def _get_inputs(
            self,
            sequence: Union[str, utils.SAMPLE],
            target_sequence: Optional[str] = None,
            is_inference: bool = False
    ) -> Tuple[utils.SAMPLE, Optional[str]]:
        if is_inference:
            assert target_sequence is None, "target sequence must be None during inference"
            return (
                (sequence, None) if isinstance(sequence, utils.SAMPLE)
                else (self.preprocessing_fn([sequence], [None], is_inference)[0][0], None)
            )
        else:
            if isinstance(sequence, utils.SAMPLE) and target_sequence is not None:
                return sequence, target_sequence
            else:
                return self.preprocessing_fn([sequence], [target_sequence], is_inference)[0]

    def prepare_sequence(
            self,
            sequence: Union[str, utils.SAMPLE],
            target_sequence: Optional[str] = None,
            is_inference: bool = False
    ) -> Tuple[DATA_INPUT, Dict[str, Any]]:
        raise NotImplementedError

    def prepare_sequences_for_inference(
            self,
            sequences: List[str]
    ) -> BATCH:
        items = [self.prepare_sequence(s, is_inference=True) for s in sequences]
        return utils.collate(items)


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
    add_dependency_info: bool = True
    word_fully_connected: bool = True
    token_fully_connected: bool = False


class SEDSequence(DatasetVariant):
    def __init__(self, cfg: DatasetVariantConfig, seed: int):
        cfg: SEDSequenceConfig
        self.tokenizer = get_tokenizer_from_config(cfg.tokenizer)

        self.bos_token_id = self.tokenizer.token_to_id(tokenization.BOS)
        self.eos_token_id = self.tokenizer.token_to_id(tokenization.EOS)
        self.unk_token_id = self.tokenizer.token_to_id(tokenization.UNK)

        if cfg.dictionary_file is not None:
            self.dictionary = io.dictionary_from_file(cfg.dictionary_file)
        else:
            self.dictionary = None

        tok_fn = tokenization.get_tokenization_fn(self.tokenizer)
        if cfg.index is not None:
            neighbor_index = index.NNIndex(cfg.index)
            neighbor_fn = index.get_neighbor_fn(neighbor_index, cfg.index_num_neighbors)
        else:
            neighbor_fn = None
        preprocessing_sample_kwargs = {}
        if cfg.data_scheme == "word_graph":
            preprocessing_sample_kwargs["with_dep_parser"] = cfg.add_dependency_info

        super().__init__(cfg, seed, tok_fn, neighbor_fn, **preprocessing_sample_kwargs)

    def construct_input(self, sample: utils.SAMPLE) -> Union[torch.Tensor, dgl.DGLHeteroGraph]:
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

    def prepare_sequence(
            self,
            sequence: Union[str, utils.SAMPLE],
            target_sequence: Optional[Union[str, utils.SAMPLE]] = None,
            is_inference: bool = False
    ) -> Tuple[
        Union[dgl.DGLHeteroGraph, graph.HeterogeneousDGLData],
        Dict[str, Any]
    ]:
        self.cfg: SEDSequenceConfig

        input_sample, target_sequence = self._get_inputs(
            sequence,
            target_sequence,
            is_inference
        )
        input_sample = utils.sanitize_sample(
            input_sample,
            unk_token_id=self.unk_token_id
        )

        info = {}
        if not is_inference:
            assert target_sequence is not None
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
                        "stage": "word_to_sequence",
                        "groups": sequence_groups,
                        # features are added before grouping so this is the correct stage
                        "features": utils.get_word_features(input_sample.doc, self.dictionary)
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

        return self.construct_input(input_sample), info


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

        self.bos_token_id = self.tokenizer.token_to_id(tokenization.BOS)
        self.eos_token_id = self.tokenizer.token_to_id(tokenization.EOS)
        self.unk_token_id = self.tokenizer.token_to_id(tokenization.UNK)

        if cfg.dictionary_file is not None:
            self.dictionary = io.dictionary_from_file(cfg.dictionary_file)
        else:
            self.dictionary = None

        tok_fn = tokenization.get_tokenization_fn(self.tokenizer)
        if cfg.index is not None:
            neighbor_index = index.NNIndex(cfg.index)
            neighbor_fn = index.get_neighbor_fn(neighbor_index, cfg.index_num_neighbors)
        else:
            neighbor_fn = None
        preprocessing_sample_kwargs = {}
        if cfg.data_scheme == "word_graph":
            preprocessing_sample_kwargs["with_dep_parser"] = cfg.add_dependency_info

        super().__init__(cfg, seed, tok_fn, neighbor_fn, **preprocessing_sample_kwargs)

    def construct_input(self, sample: utils.SAMPLE) -> Union[torch.Tensor, dgl.DGLHeteroGraph]:
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

    def prepare_sequence(
            self,
            sequence: Union[str, utils.SAMPLE],
            target_sequence: Optional[Union[str, utils.SAMPLE]] = None,
            is_inference: bool = False
    ) -> Tuple[
        Union[dgl.DGLHeteroGraph, graph.HeterogeneousDGLData],
        Dict[str, Any]
    ]:
        self.cfg: SEDWordsConfig

        input_sample, target_sequence = self._get_inputs(
            sequence,
            target_sequence,
            is_inference
        )
        input_sample = utils.sanitize_sample(
            input_sample,
            unk_token_id=self.unk_token_id
        )

        info = {}
        if not is_inference:
            assert target_sequence is not None
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
            info["label"] = {
                "word": label
            }

        elif self.cfg.data_scheme == "token_graph":
            if self.cfg.add_word_features:
                # if we add word features, we first aggregate from tokens to words
                # and then from word to whitespace_words
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
                        "stage": "word_to_word_ws",
                        "groups": word_ws_groups,
                        # features are added before grouping so this is the correct stage
                        "features": utils.get_word_features(input_sample.doc, self.dictionary)
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
            info["label"] = label

        else:
            raise RuntimeError(f"unknown data scheme {self.cfg.data_scheme}")

        return self.construct_input(input_sample), info


@dataclass
class TokenizationRepairConfig(DatasetVariantConfig):
    type: DatasetVariants = DatasetVariants.TOKENIZATION_REPAIR

    data_scheme: str = "tensor"

    tokenization_level: str = "char"  # one of {char, byte}


class TokenizationRepair(DatasetVariant):
    def __init__(self, cfg: DatasetVariantConfig, seed: int):
        cfg: TokenizationRepairConfig

        if cfg.tokenization_level == "char":
            self.tokenizer = tokenization.CharTokenizer()
        elif cfg.tokenization_level == "byte":
            self.tokenizer = tokenization.ByteTokenizer()
        else:
            raise ValueError(f"unknown tokenization level {cfg.tokenization_level}, must be one of {{char, byte}}")

        tok_fn = tokenization.get_tokenization_fn(self.tokenizer, True)

        super().__init__(cfg, seed, tok_fn)

    def construct_input(self, sample: utils.SAMPLE) -> Union[dgl.DGLHeteroGraph, torch.Tensor]:
        self.cfg: TokenizationRepairConfig
        if self.cfg.data_scheme == "tensor":
            return torch.tensor(
                utils.flatten(sample.tokens), dtype=torch.long
            )
        else:
            raise ValueError(f"Unknown data scheme {self.cfg.data_scheme}")

    def prepare_sequence(
            self,
            sequence: Union[str, utils.SAMPLE],
            target_sequence: Optional[str] = None,
            is_inference: bool = False
    ) -> Tuple[
        Union[dgl.DGLHeteroGraph, graph.HeterogeneousDGLData],
        Dict[str, Any]
    ]:
        self.cfg: TokenizationRepairConfig

        input_sample, target_sequence = self._get_inputs(sequence, target_sequence, is_inference)
        input_sample = utils.sanitize_sample(
            input_sample,
            unk_token_id=self.tokenizer.token_to_id(tokenization.UNK)
        )

        info = {}
        if not is_inference:
            assert target_sequence is not None
            # get the whitespace operations to turn input_sample into target_sequence
            label = tokenization_repair.get_whitespace_operations(
                str(input_sample), target_sequence
            )
            info["label"] = torch.tensor(label, dtype=torch.long)

        return self.construct_input(input_sample), info


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
        cfg: SECNMTConfig
        self.input_tokenizer = get_tokenizer_from_config(cfg.input_tokenizer)
        self.output_tokenizer = get_tokenizer_from_config(cfg.output_tokenizer)

        self.input_bos_token_id = self.input_tokenizer.token_to_id(tokenization.BOS)
        self.input_eos_token_id = self.input_tokenizer.token_to_id(tokenization.EOS)
        self.input_unk_token_id = self.input_tokenizer.token_to_id(tokenization.UNK)

        self.output_pad_token_id = self.output_tokenizer.token_to_id(tokenization.PAD)

        if cfg.dictionary_file is not None:
            self.dictionary = io.dictionary_from_file(cfg.dictionary_file)
        else:
            self.dictionary = None

        tok_fn = tokenization.get_tokenization_fn(self.input_tokenizer)
        if cfg.index is not None:
            neighbor_index = index.NNIndex(cfg.index)
            neighbor_fn = index.get_neighbor_fn(neighbor_index, cfg.index_num_neighbors)
        else:
            neighbor_fn = None
        preprocessing_sample_kwargs = {}
        if cfg.data_scheme == "word_graph":
            preprocessing_sample_kwargs["with_dep_parser"] = cfg.add_dependency_info

        super().__init__(cfg, seed, tok_fn, neighbor_fn, **preprocessing_sample_kwargs)

    def construct_input(
            self,
            sample: utils.SAMPLE
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

    def prepare_sequence(
            self,
            sequence: str,
            target_sequence: Optional[str] = None,
            is_inference: bool = False
    ) -> Tuple[
        Union[torch.Tensor, dgl.DGLHeteroGraph],
        Dict[str, Any]
    ]:
        self.cfg: SECWordsNMTConfig

        input_sample, target_sequence = self._get_inputs(
            sequence,
            target_sequence,
            is_inference
        )
        input_sample = utils.sanitize_sample(
            input_sample,
            unk_token_id=self.input_unk_token_id
        )

        info = {"pad_token_id": self.output_pad_token_id}
        input_words = str(input_sample).split()
        if not is_inference:
            assert target_sequence is not None
            target_words = target_sequence.split()
            assert len(input_words) == len(target_words)
            label = [
                self.output_tokenizer.tokenize(word, add_bos_eos=True)
                for word in target_words
            ]
            label_splits = [len(labels) for labels in label]
            info["label"] = torch.tensor(utils.flatten(label), dtype=torch.long)
            info["label_splits"] = label_splits

        encoder_group_lengths = [0] * len(input_words)
        word_ws_idx = 0
        for tokens, word in zip(input_sample.tokens, input_sample.doc):
            encoder_group_lengths[word_ws_idx] += len(tokens)
            if word.whitespace_ == " ":
                word_ws_idx += 1
        info["encoder_group_lengths"] = torch.tensor(encoder_group_lengths, dtype=torch.long)

        return self.construct_input(input_sample), info


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

        self.input_bos_token_id = self.input_tokenizer.token_to_id(tokenization.BOS)
        self.input_eos_token_id = self.input_tokenizer.token_to_id(tokenization.EOS)
        self.input_unk_token_id = self.input_tokenizer.token_to_id(tokenization.UNK)

        self.output_pad_token_id = self.output_tokenizer.token_to_id(tokenization.PAD)

        if cfg.dictionary_file is not None:
            self.dictionary = io.dictionary_from_file(cfg.dictionary_file)
        else:
            self.dictionary = None

        tok_fn = tokenization.get_tokenization_fn(self.input_tokenizer)
        if cfg.index is not None:
            neighbor_index = index.NNIndex(cfg.index)
            neighbor_fn = index.get_neighbor_fn(neighbor_index, cfg.index_num_neighbors)
        else:
            neighbor_fn = None
        preprocessing_sample_kwargs = {}
        if cfg.data_scheme == "word_graph":
            preprocessing_sample_kwargs["with_dep_parser"] = cfg.add_dependency_info

        super().__init__(cfg, seed, tok_fn, neighbor_fn, **preprocessing_sample_kwargs)

    def construct_input(
            self,
            sample: utils.SAMPLE
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
                [self.input_bos_token_id] + utils.flatten(sample.tokens) + [self.input_eos_token_id],
                dtype=torch.long
            )
        else:
            raise ValueError(f"unknown data scheme {self.cfg.data_scheme}")

    def prepare_sequence(
            self,
            sequence: str,
            target_sequence: Optional[str] = None,
            is_inference: bool = False
    ) -> Tuple[
        Union[torch.Tensor, dgl.DGLHeteroGraph],
        Dict[str, Any]
    ]:
        self.cfg: SECNMTConfig

        input_sample, target_sequence = self._get_inputs(
            sequence,
            target_sequence,
            is_inference
        )
        input_sample = utils.sanitize_sample(
            input_sample,
            unk_token_id=self.input_unk_token_id
        )

        info = {"pad_token_id": self.output_pad_token_id}
        if not is_inference:
            assert target_sequence is not None
            label = torch.tensor(self.output_tokenizer.tokenize(target_sequence, add_bos_eos=True), dtype=torch.long)
            info["label"] = label

        return self.construct_input(input_sample), info


@dataclass
class TokenizationRepairPlusSEDConfig(TokenizationRepairConfig):
    type: DatasetVariants = DatasetVariants.TOKENIZATION_REPAIR_PLUS_SED

    dictionary_file: Optional[str] = None
    add_word_features: bool = True


class TokenizationRepairPlusSED(TokenizationRepair):
    def __init__(self, cfg: DatasetVariantConfig, seed: int):
        super().__init__(cfg, seed)
        self.cfg: TokenizationRepairPlusSEDConfig

        if self.cfg.dictionary_file is not None:
            self.dictionary = io.dictionary_from_file(self.cfg.dictionary_file)
        else:
            self.dictionary = None

    def prepare_sequence(
            self,
            sequence: Union[str, utils.SAMPLE],
            target_sequence: Optional[str] = None,
            is_inference: bool = False
    ) -> Tuple[
        Union[dgl.DGLHeteroGraph, graph.HeterogeneousDGLData],
        Dict[str, Any]
    ]:
        self.cfg: TokenizationRepairPlusSEDConfig

        input_sample, target_sequence = self._get_inputs(sequence, target_sequence, is_inference)
        input_sample = utils.sanitize_sample(
            input_sample,
            unk_token_id=self.tokenizer.token_to_id(tokenization.UNK)
        )

        info = {}
        if not is_inference:
            assert target_sequence is not None
            # get the whitespace operations to turn input_sample into target_sequence
            tokenization_repair_label = tokenization_repair.get_whitespace_operations(
                str(input_sample), target_sequence
            )

            assert "org_sequence" in input_sample.info
            org_words = input_sample.info["org_sequence"].split()
            target_words = target_sequence.split()
            assert len(target_words) == len(org_words)

            info["tokenization_repair_label"] = torch.tensor(tokenization_repair_label, dtype=torch.long)
            repaired_words, repaired_doc = utils.tokenize_words(target_sequence, return_doc=True)
            info["word_groups"] = [
                {
                    "stage": "char_to_word",
                    "groups": utils.get_character_groups_from_repaired_doc(list(str(input_sample)), repaired_doc)
                }
            ]
            word_ws_groups = [0] * len(target_words)
            word_ws_idx = 0
            for word in repaired_doc:
                word_ws_groups[word_ws_idx] += 1
                if word.whitespace_ == " ":
                    word_ws_idx += 1

            info["word_ws_groups"] = [
                {
                    "stage": "word_to_word_ws",
                    "groups": word_ws_groups,
                    "features": utils.get_word_features(repaired_doc, self.dictionary)
                }
            ]

            info["sed_label"] = torch.tensor(
                [
                    int(org_word != target_word)
                    for org_word, target_word in zip(org_words, target_words)
                ], dtype=torch.long
            )

        return self.construct_input(input_sample), info


@dataclass
class TokenizationRepairPlusSEDPlusSECConfig(TokenizationRepairPlusSEDConfig):
    type: DatasetVariants = DatasetVariants.TOKENIZATION_REPAIR_PLUS_SED_PLUS_SEC


@dataclass
class SEDPlusSECConfig(DatasetVariantConfig):
    type: DatasetVariants = DatasetVariants.SED_PLUS_SEC

    data_scheme: str = "tensor"

    dictionary_file: Optional[str] = None
    add_word_features: bool = True


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
    elif variant_type == DatasetVariants.SEC_WORDS_NMT:
        cfg = OmegaConf.structured(SECWordsNMTConfig(**cfg))
        return SECWordsNMT(cfg, seed)
    elif variant_type == DatasetVariants.SEC_NMT:
        cfg = OmegaConf.structured(SECNMTConfig(**cfg))
        return SECNMT(cfg, seed)
    elif variant_type == DatasetVariants.TOKENIZATION_REPAIR_PLUS_SED:
        cfg = OmegaConf.structured(TokenizationRepairPlusSEDConfig(**cfg))
        return TokenizationRepairPlusSED(cfg, seed)
    elif variant_type == DatasetVariants.TOKENIZATION_REPAIR_PLUS_SED_PLUS_SEC:
        cfg = OmegaConf.structured(TokenizationRepairPlusSEDPlusSECConfig(**cfg))
        return TokenizationRepairPlusSED(cfg, seed)
    elif variant_type == DatasetVariants.SED_PLUS_SEC:
        cfg = OmegaConf.structured(SEDPlusSECConfig(**cfg))
        return SEDPlusSEC(cfg, seed)
    else:
        raise ValueError(f"Unknown variant {cfg.type.name}")
