import enum
import hashlib
import pprint
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
from gnn_lib.utils import tokenization_repair, io, Batch, DataInput


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

    preprocessing: PreprocessingConfig = MISSING


class DatasetVariant:
    def __init__(
            self,
            cfg: DatasetVariantConfig,
            seed: int,
            tokenization_fn: utils.TokenizationFn,
            neighbor_fn: Optional[utils.NeighborFn] = None,
            **preprocessing_kwargs: Dict[str, Any]
    ):
        self.cfg = cfg
        self.seed = seed
        self.rand = np.random.default_rng(seed)

        preprocessing = get_preprocessing_from_config(self.cfg.preprocessing, self.seed)
        self.preprocessing_fn: utils.PreprocessingFn = get_preprocessing_fn(
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

    def _get_sample(
            self,
            sequence: Union[str, utils.Sample],
            target_sequence: Optional[str] = None,
            is_inference: bool = False,
            unk_token_id: Optional[int] = None
    ) -> Tuple[utils.Sample, Optional[str]]:
        sequence_is_sample = isinstance(sequence, utils.Sample)
        if is_inference and not sequence_is_sample:
            sequence = self.preprocessing_fn([sequence], [None], is_inference)[0][0]
        elif not is_inference and (not sequence_is_sample or target_sequence is None):
            sequence, target_sequence = self.preprocessing_fn([str(sequence)], [target_sequence], is_inference)[0]
        return utils.sanitize_sample(sequence, unk_token_id), target_sequence

    def get_inputs(
            self,
            sequence: Union[str, utils.Sample],
            target_sequence: Optional[str] = None,
            is_inference: bool = False
    ) -> Tuple[DataInput, Dict[str, Any]]:
        raise NotImplementedError

    def _split_sample_for_inference(
            self,
            sample: utils.Sample,
            max_length: int,
            context_length: int,
            **kwargs: Any
    ) -> List[Tuple[int, int, int, int]]:
        raise NotImplementedError

    def prepare_sequences_for_inference(
            self,
            sequences: List[str],
            max_length: int,
            context_length: int,
            **kwargs: Any
    ) -> Tuple[List[utils.Sample], List[utils.InferenceInfo]]:
        samples, _ = zip(*self.preprocessing_fn(sequences, [None] * len(sequences), True))
        all_samples = []
        all_infos = []
        for sample in samples:
            sequence = str(sample)
            token_lengths = [len(tokens) for tokens in sample.tokens]
            length = sum(token_lengths)
            if length <= max_length:
                all_samples.append(sample)
                all_infos.append(utils.InferenceInfo(
                    ctx_start=0,
                    ctx_end=len(sequence),
                    window_start=0,
                    window_end=0,
                    window_idx=0,
                    length=length
                ))
            else:
                windows = self._split_sample_for_inference(sample, max_length, context_length, **kwargs)
                for i, (ctx_start, ctx_end, window_start, window_end) in enumerate(windows):
                    sample, _ = self._get_sample(sequence[ctx_start:ctx_end], is_inference=True)
                    all_samples.append(sample)
                    all_infos.append(utils.InferenceInfo(
                        ctx_start=ctx_start,
                        ctx_end=ctx_end,
                        window_start=window_start,
                        window_end=window_end,
                        window_idx=i,
                        length=sum(len(t) for t in sample.tokens)
                    ))
        return all_samples, all_infos

    def batch_sequences_for_inference(
            self,
            sequences: List[Union[str, utils.Sample]]
    ) -> Batch:
        items = [self.get_inputs(s, is_inference=True) for s in sequences]
        return utils.collate(items)

    def _merge_inference_outputs(
            self,
            sequence: str,
            infos: List[utils.InferenceInfo],
            predictions: List[Any],
            **kwargs: Any
    ) -> Any:
        raise NotImplementedError

    def postprocess_inference_outputs(
            self,
            sequences: List[str],
            infos: List[utils.InferenceInfo],
            predictions: List[Any],
            **kwargs: Any
    ) -> List[Any]:
        grouped_predictions: List[List[int]] = []
        grouped_infos: List[List[utils.InferenceInfo]] = []
        prev_info = None
        for info, prediction in zip(infos, predictions):
            if info.window_idx == 0:
                grouped_predictions.append([prediction])
                grouped_infos.append([info])
            elif info.window_idx == prev_info.window_idx + 1:
                grouped_predictions[-1].append(prediction)
                grouped_infos[-1].append(info)
            else:
                raise RuntimeError("should not happen")

            prev_info = info

        assert len(sequences) == len(grouped_predictions) == len(grouped_infos)

        merged_predictions = []
        for sequence, predictions, infos in zip(sequences, grouped_predictions, grouped_infos):
            if len(predictions) == 1:
                merged_predictions.append(predictions[0])
            else:
                merged_predictions.append(self._merge_inference_outputs(sequence, infos, predictions))
        return merged_predictions


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
        preprocessing_sample_kwargs = {}
        if cfg.data_scheme == "word_graph":
            preprocessing_sample_kwargs["with_dep_parser"] = cfg.add_dependency_info

        super().__init__(cfg, seed, tok_fn, neighbor_fn, **preprocessing_sample_kwargs)

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

        input_sample, target_sequence = self._get_sample(
            sequence,
            target_sequence,
            is_inference,
            self.unk_token_id
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

    def _split_sample_for_inference(
            self,
            sample: utils.Sample,
            max_length: int,
            context_length: int,
            **kwargs: Any
    ) -> List[Tuple[int, int, int, int]]:
        return utils.get_word_windows(sample, max_length, context_length)

    def _merge_inference_outputs(
            self,
            sequence: str,
            infos: List[utils.InferenceInfo],
            predictions: List[int],
            **kwargs: Any
    ) -> int:
        assert all(p in {0, 1} for p in predictions)
        # for sed sequence, if for any part of the sequence an error was detected, the overall sequence has an error
        return int(any(p for p in predictions))


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
        preprocessing_sample_kwargs = {}
        if cfg.data_scheme == "word_graph":
            preprocessing_sample_kwargs["with_dep_parser"] = cfg.add_dependency_info

        super().__init__(cfg, seed, tok_fn, neighbor_fn, **preprocessing_sample_kwargs)

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

        input_sample, target_sequence = self._get_sample(
            sequence,
            target_sequence,
            is_inference,
            self.unk_token_id
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
            info["label"] = label

        else:
            raise RuntimeError(f"unknown data scheme {self.cfg.data_scheme}")

        return self._construct_input(input_sample), info

    def _split_sample_for_inference(
            self,
            sample: utils.Sample,
            max_length: int,
            context_length: int,
            **kwargs: Any
    ) -> List[Tuple[int, int, int, int]]:
        return utils.get_word_windows(sample, max_length, context_length)

    def _merge_inference_outputs(
            self,
            sequence: str,
            infos: List[utils.InferenceInfo],
            predictions: List[List[int]],
            **kwargs: Any
    ) -> List[int]:
        assert all(p in {0, 1} for prediction in predictions for p in prediction)
        merged_prediction = []
        for info, prediction in zip(infos, predictions):
            assert len(sequence[info.ctx_start:info.ctx_end].split()) == len(prediction)
            num_left_context_words = len(sequence[info.ctx_start:info.window_start].split())
            num_window_words = len(sequence[info.window_start:info.window_end].split())
            merged_prediction.extend(prediction[num_left_context_words:num_left_context_words + num_window_words])
        assert len(merged_prediction) == len(sequence.split())
        return merged_prediction


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

        self.unk_token_id = self.tokenizer.token_to_id(tokenization.UNK)
        tok_fn = tokenization.get_tokenization_fn(self.tokenizer)

        super().__init__(cfg, seed, tok_fn)

    def _construct_input(self, sample: utils.Sample) -> Union[dgl.DGLHeteroGraph, torch.Tensor]:
        self.cfg: TokenizationRepairConfig
        if self.cfg.data_scheme == "tensor":
            return torch.tensor(
                utils.flatten(sample.tokens), dtype=torch.long
            )
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

        input_sample, target_sequence = self._get_sample(
            sequence,
            target_sequence,
            is_inference,
            self.unk_token_id
        )

        info = {}
        if not is_inference:
            assert target_sequence is not None
            # get the whitespace operations to turn input_sample into target_sequence
            label = tokenization_repair.get_whitespace_operations(
                str(input_sample), target_sequence
            )
            info["label"] = torch.tensor(label, dtype=torch.long)

        return self._construct_input(input_sample), info

    def _split_sample_for_inference(
            self,
            sample: utils.Sample,
            max_length: int,
            context_length: int,
            **kwargs: Any
    ) -> List[Tuple[int, int, int, int]]:
        self.cfg: TokenizationRepairConfig
        if self.cfg.tokenization_level == "char":
            return utils.get_character_windows(sample, max_length, context_length)
        elif self.cfg.tokenization_level == "byte":
            return utils.get_byte_windows(sample, max_length, context_length)
        else:
            raise RuntimeError("should not happen")

    def _merge_inference_outputs(
            self,
            sequence: str,
            infos: List[utils.InferenceInfo],
            predictions: List[str],
            **kwargs: Any
    ) -> str:
        merged_prediction = ""
        for prediction, info in zip(predictions, infos):
            left_context = sequence[info.ctx_start:info.window_start]
            window = sequence[info.window_start:info.window_end]
            right_context = sequence[info.window_end:info.ctx_end]
            match_start, match_end = tokenization_repair.match_string_ignoring_space(
                prediction,
                left_context,
                window,
                right_context
            )
            merged_prediction += prediction[match_start:match_end]
        assert merged_prediction.replace(" ", "") == sequence.replace(" ", "")
        return merged_prediction


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

        self.input_unk_token_id = self.input_tokenizer.token_to_id(tokenization.UNK)

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
        preprocessing_sample_kwargs = {}
        if cfg.data_scheme == "word_graph":
            preprocessing_sample_kwargs["with_dep_parser"] = cfg.add_dependency_info

        super().__init__(cfg, seed, tok_fn, neighbor_fn, **preprocessing_sample_kwargs)

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

        input_sample, target_sequence = self._get_sample(
            sequence,
            target_sequence,
            is_inference,
            self.input_unk_token_id
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

        return self._construct_input(input_sample), info

    def _split_sample_for_inference(
            self,
            sample: utils.Sample,
            max_length: int,
            context_length: int,
            **kwargs: Any
    ) -> List[Tuple[int, int, int, int]]:
        return utils.get_word_windows(sample, max_length, context_length)

    def _merge_inference_outputs(
            self,
            sequence: str,
            infos: List[utils.InferenceInfo],
            predictions: List[List[str]],
            **kwargs: Any
    ) -> List[str]:
        min_num_predictions = min(len(prediction) for prediction in predictions)
        merged_predictions = [[] for _ in range(min_num_predictions)]
        for info, prediction in zip(infos, predictions):
            num_left_context_words = len(sequence[info.ctx_start:info.window_start].split())
            num_window_words = len(sequence[info.window_start:info.window_end].split())
            for i in range(min_num_predictions):
                predicted_words = prediction[i].split()
                merged_predictions[i].extend(
                    predicted_words[num_left_context_words:num_left_context_words + num_window_words]
                )
        merged_predictions = [" ".join(predicted_words) for predicted_words in merged_predictions]
        return merged_predictions


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

        self.input_unk_token_id = self.input_tokenizer.token_to_id(tokenization.UNK)

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
        preprocessing_sample_kwargs = {}
        if cfg.data_scheme == "word_graph":
            preprocessing_sample_kwargs["with_dep_parser"] = cfg.add_dependency_info

        super().__init__(cfg, seed, tok_fn, neighbor_fn, **preprocessing_sample_kwargs)

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

        input_sample, target_sequence = self._get_sample(
            sequence,
            target_sequence,
            is_inference,
            self.input_unk_token_id
        )

        info = {"pad_token_id": self.output_pad_token_id}
        if not is_inference:
            assert target_sequence is not None
            label = torch.tensor(self.output_tokenizer.tokenize(target_sequence, add_bos_eos=True), dtype=torch.long)
            info["label"] = label

        return self._construct_input(input_sample), info

    def _split_sample_for_inference(
            self,
            sample: utils.Sample,
            max_length: int,
            context_length: int,
            **kwargs: Any
    ) -> List[Tuple[int, int, int, int]]:
        return utils.get_word_windows(sample, max_length, 0)

    def _merge_inference_outputs(
            self,
            sequence: str,
            infos: List[utils.InferenceInfo],
            predictions: List[List[str]],
            **kwargs: Any
    ) -> Any:
        print(pprint.pformat(predictions))
        min_num_predictions = min(len(prediction) for prediction in predictions)
        merged_predictions = [[] for _ in range(min_num_predictions)]
        for prediction in predictions:
            for i in range(min_num_predictions):
                merged_predictions[i].append(prediction[i])
        merged_predictions = [" ".join(s.strip() for s in predictions) for predictions in merged_predictions]
        return merged_predictions


@dataclass
class TokenizationRepairPlusConfig(TokenizationRepairConfig):
    type: DatasetVariants = DatasetVariants.TOKENIZATION_REPAIR_PLUS

    # one of {tokenization_repair_plus_sed, tokenization_repair_plus_sed_plus_sec}
    output_type: str = "tokenization_repair_plus_sed"

    dictionary_file: Optional[str] = None
    add_word_features: bool = True

    # for tokenization_repair_plus_sed_plus_sec
    sec_tokenizer: Optional[TokenizerConfig] = None


class TokenizationRepairPlus(TokenizationRepair):
    def __init__(self, cfg: DatasetVariantConfig, seed: int):
        super().__init__(cfg, seed)
        self.cfg: TokenizationRepairPlusConfig
        assert self.cfg.output_type in {"tokenization_repair_plus_sed", "tokenization_repair_plus_sed_plus_sec"}

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

        input_sample, target_sequence = self._get_sample(
            sequence,
            target_sequence,
            is_inference,
            self.unk_token_id
        )

        info = {}
        if not is_inference:
            assert target_sequence is not None
            # get the whitespace operations to turn input_sample into target_sequence
            tokenization_repair_label = tokenization_repair.get_whitespace_operations(
                str(input_sample), target_sequence
            )
            info["tokenization_repair_label"] = torch.tensor(tokenization_repair_label, dtype=torch.long)

            assert "org_sequence" in input_sample.info
            org_words = input_sample.info["org_sequence"].split()
            target_words = target_sequence.split()
            assert len(target_words) == len(org_words)

            repaired_words, repaired_doc = utils.tokenize_words(target_sequence, return_doc=True)
            info["word_groups"] = {
                "stage": "char_to_word",
                "groups": utils.get_character_groups_from_repaired_doc(list(str(input_sample)), repaired_doc)
            }

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
                assert self.sec_tokenizer is not None, "output tokenizer must be specified to use sec output"
                label = [
                    self.sec_tokenizer.tokenize(word, add_bos_eos=True)
                    for word in org_words
                ]
                label_splits = [len(labels) for labels in label]
                info["sec_label"] = torch.tensor(utils.flatten(label), dtype=torch.long)
                info["sec_label_splits"] = label_splits
                info["sec_pad_token_id"] = self.sec_pad_token_id

        return self._construct_input(input_sample), info

    def _split_sample_for_inference(
            self,
            sample: utils.Sample,
            max_length: int,
            context_length: int,
            no_repair: bool = False,
            **kwargs: Any
    ) -> List[Tuple[int, int, int, int]]:
        if no_repair:
            return utils.get_word_windows(sample, max_length, context_length)
        else:
            return super()._split_sample_for_inference(sample, max_length, context_length, **kwargs)

    def postprocess_inference_outputs(
            self,
            sequences: List[str],
            infos: List[utils.InferenceInfo],
            predictions: List[Any],
            output_type: str = "all",
            no_repair: bool = False,
            **kwargs: Any
    ) -> List[Any]:
        if no_repair:
            pass
        return predictions


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
    elif variant_type == DatasetVariants.TOKENIZATION_REPAIR_PLUS:
        cfg = OmegaConf.structured(TokenizationRepairPlusConfig(**cfg))
        return TokenizationRepairPlus(cfg, seed)
    else:
        raise ValueError(f"Unknown variant {cfg.type.name}")
