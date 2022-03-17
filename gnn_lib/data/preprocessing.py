import enum
import hashlib
import json
from dataclasses import dataclass
from typing import Tuple, List, Dict, Optional, Union, Any

import numpy as np
import omegaconf
from omegaconf import MISSING, OmegaConf
from spacy.tokens import Doc

from gnn_lib.data import utils
from gnn_lib.data.utils import PREPROCESSING_FN, SAMPLE, TOKENIZATION_FN, NEIGHBOR_FN


def artificial_edits(word: str, num_edits: int, rand: np.random.Generator) -> str:
    exclude_indices = set()
    for _ in range(num_edits):
        word, edits, exclude_indices = utils.edit_token(
            token=word,
            rand=rand,
            exclude_indices=exclude_indices
        )
    return word


def realistic_edits(
        word: str,
        rand: np.random.Generator,
        word_misspellings: Dict[str, List[str]]) -> str:
    pot_misspellings = word_misspellings.get(word, word_misspellings.get(word.lower()))
    if pot_misspellings is not None and len(pot_misspellings) > 0:
        return pot_misspellings[rand.integers(len(pot_misspellings))]
    else:
        return word


def corrupt_sequence(
        words: List[str],
        doc: Doc,
        edit_token_p: float,
        num_edits_p: float,
        corrupt_method: str,
        rand: np.random.Generator,
        word_misspellings: Optional[Dict[str, List[str]]] = None,
        mixed_artificial_p: float = 0.2,
        re_weight_edit_token_p: bool = False
) -> List[str]:
    assert len(words) == len(doc) and len(words) > 0

    special_tokens_mask = np.array([utils.is_special_token(doc[i]) for i in range(len(words))], dtype=bool)

    artificial_p = realistic_p = 0
    if re_weight_edit_token_p:
        num_non_special_tokens = np.logical_not(special_tokens_mask).sum()
        if corrupt_method == "artificial":
            # compensate for the special tokens, because we do not edit them during artificial corruption
            artificial_p = edit_token_p * (len(words) / max(1, num_non_special_tokens))
        else:
            assert word_misspellings is not None, \
                "need word misspellings and char operations for realistic and mixed noise"
            # for the realistic and mixed methods we additionally compensate for the words that we do not
            # have misspellings for, because we leave them unchanged
            in_misspellings_mask = np.array([
                word in word_misspellings or word.lower() in word_misspellings
                for word in words
            ], dtype=bool)

            num_non_special_and_in_misspellings_tokens = np.logical_and(
                np.logical_not(special_tokens_mask),
                in_misspellings_mask
            ).sum()

            realistic_editable_frac = len(words) / max(1, num_non_special_and_in_misspellings_tokens)
            if corrupt_method == "realistic":
                realistic_p = edit_token_p * realistic_editable_frac
            else:
                artificial_p = edit_token_p * (len(words) / max(1, num_non_special_tokens)) * mixed_artificial_p
                realistic_p = artificial_p + edit_token_p * (
                        len(words) / max(1, num_non_special_and_in_misspellings_tokens)) * (1 - mixed_artificial_p)
    else:
        if corrupt_method == "artificial":
            artificial_p = edit_token_p
        elif corrupt_method == "realistic":
            realistic_p = edit_token_p
        else:
            artificial_p = edit_token_p * mixed_artificial_p
            # equal to edit_token_p but here for clarity
            realistic_p = artificial_p + edit_token_p * (1 - mixed_artificial_p)

    for word_idx in range(len(words)):
        if special_tokens_mask[word_idx]:  # do not edit special tokens
            continue

        word = words[word_idx]

        r = rand.random()
        if corrupt_method == "artificial":
            if r < artificial_p:
                num_edits = min(rand.geometric(num_edits_p), len(word))
                word = artificial_edits(word, num_edits, rand)
        elif corrupt_method == "realistic":
            if r < realistic_p:
                word = realistic_edits(word, rand, word_misspellings)
        elif corrupt_method == "mixed":
            if r < artificial_p:
                num_edits = min(rand.geometric(num_edits_p), len(word))
                word = artificial_edits(word, num_edits, rand)
            elif r < realistic_p:
                word = realistic_edits(word, rand, word_misspellings)
        else:
            raise ValueError(f"Unknown corrupt method {corrupt_method}")

        words[word_idx] = word

    return words


def corrupt_whitespace(
        sequence: str,
        iw_p: float,
        dw_p: float,
        no_ws_p: float,
        rand: np.random.Generator) -> str:
    if rand.random() < no_ws_p:
        return sequence.replace(" ", "")

    new_s = ""
    sequence_ptr = 0
    while sequence_ptr < len(sequence):
        char = sequence[sequence_ptr]
        prev_char = sequence[sequence_ptr - 1] if sequence_ptr > 0 else " "
        r = rand.random()

        if char == " ":
            if r < dw_p:
                pass
            else:
                new_s += char
        elif prev_char != " " and r < iw_p:
            new_s += " " + char
        else:
            new_s += char

        sequence_ptr += 1
    return new_s


class Preprocessings(enum.IntEnum):
    ARTIFICIAL_NOISE = 1
    REALISTIC_NOISE = 2
    MIXED_NOISE = 3
    WHITESPACE_NOISE = 4
    CHAINED = 5
    SWITCH = 6
    NONE = 7
    SUBSTRING = 8


@dataclass
class PreprocessingConfig:
    type: Preprocessings = MISSING


class Preprocessing:
    def __init__(self, cfg: PreprocessingConfig, seed: int) -> None:
        self.cfg = cfg
        self.rand = np.random.default_rng(seed)

    @property
    def name(self) -> str:
        return self.cfg.type.name

    def apply(
            self,
            sequences: List[str],
            target_sequences: List[str],
            infos: List[Dict[str, Any]],
            is_inference: bool = False
    ) -> Tuple[List[str], List[str], List[Dict[str, Any]]]:
        raise NotImplementedError

    @property
    def cfg_string(self) -> str:
        cfg_yaml = OmegaConf.to_yaml(self.cfg, resolve=True, sort_keys=True)
        return str(hashlib.sha1(cfg_yaml.encode("utf8")).hexdigest())


@dataclass
class ArtificialNoiseConfig(PreprocessingConfig):
    type: Preprocessings = Preprocessings.ARTIFICIAL_NOISE

    edit_token_p: float = MISSING
    num_edits_p: float = MISSING
    re_weight_edit_token_p: bool = True


class ArtificialNoise(Preprocessing):
    def apply(
            self,
            sequences: List[str],
            target_sequences: List[str],
            infos: List[Dict[str, Any]],
            is_inference: bool = False
    ) -> Tuple[List[str], List[str], List[Dict[str, Any]]]:
        if is_inference:
            return sequences, target_sequences, infos
        self.cfg: ArtificialNoiseConfig
        batch = utils.tokenize_words_batch(sequences, return_docs=True)
        batch_corrupted = [
            corrupt_sequence(
                words=words,
                doc=doc,
                edit_token_p=self.cfg.edit_token_p,
                num_edits_p=self.cfg.num_edits_p,
                corrupt_method="artificial",
                rand=self.rand,
                re_weight_edit_token_p=self.cfg.re_weight_edit_token_p
            )
            for words, doc in batch
        ]
        return (
            [utils.de_tokenize_words(words, doc) for words, (_, doc) in zip(batch_corrupted, batch)],
            target_sequences,
            infos
        )


@dataclass
class RealisticNoiseConfig(PreprocessingConfig):
    type: Preprocessings = Preprocessings.REALISTIC_NOISE

    edit_token_p: float = MISSING
    word_misspellings_file: str = MISSING
    re_weight_edit_token_p: bool = True


class RealisticNoise(Preprocessing):
    def __init__(self, cfg: PreprocessingConfig, seed: int) -> None:
        super().__init__(cfg, seed)
        self.cfg: RealisticNoiseConfig
        with open(self.cfg.word_misspellings_file, "r", encoding="utf8") as inf:
            self.word_misspellings = json.load(inf)

    def apply(
            self,
            sequences: List[str],
            target_sequences: List[str],
            infos: List[Dict[str, Any]],
            is_inference: bool = False
    ) -> Tuple[List[str], List[str], List[Dict[str, Any]]]:
        if is_inference:
            return sequences, target_sequences, infos
        self.cfg: RealisticNoiseConfig
        batch = utils.tokenize_words_batch(sequences, return_docs=True)
        batch_corrupted = [
            corrupt_sequence(
                words=words,
                doc=doc,
                edit_token_p=self.cfg.edit_token_p,
                num_edits_p=0,  # not used for realistic noise
                corrupt_method="realistic",
                rand=self.rand,
                word_misspellings=self.word_misspellings,
                re_weight_edit_token_p=self.cfg.re_weight_edit_token_p
            )
            for words, doc in batch
        ]
        return (
            [utils.de_tokenize_words(words, doc) for words, (_, doc) in zip(batch_corrupted, batch)],
            target_sequences,
            infos
        )


@dataclass
class MixedNoiseConfig(PreprocessingConfig):
    type: Preprocessings = Preprocessings.MIXED_NOISE

    edit_token_p: float = MISSING
    artificial_num_edits_p: float = MISSING
    word_misspellings_file: str = MISSING
    artificial_p: float = MISSING
    re_weight_edit_token_p: bool = True


class MixedNoise(Preprocessing):
    def __init__(self, cfg: PreprocessingConfig, seed: int) -> None:
        super().__init__(cfg, seed)
        self.cfg: MixedNoiseConfig
        with open(self.cfg.word_misspellings_file, "r", encoding="utf8") as inf:
            self.word_misspellings = json.load(inf)

    def apply(
            self,
            sequences: List[str],
            target_sequences: List[str],
            infos: List[Dict[str, Any]],
            is_inference: bool = False
    ) -> Tuple[List[str], List[str], List[Dict[str, Any]]]:
        if is_inference:
            return sequences, target_sequences, infos
        self.cfg: MixedNoiseConfig
        batch = utils.tokenize_words_batch(sequences, return_docs=True)
        batch_corrupted = [
            corrupt_sequence(
                words=words,
                doc=doc,
                num_edits_p=self.cfg.artificial_num_edits_p,
                edit_token_p=self.cfg.edit_token_p,
                corrupt_method="mixed",
                rand=self.rand,
                word_misspellings=self.word_misspellings,
                mixed_artificial_p=self.cfg.artificial_p,
                re_weight_edit_token_p=self.cfg.re_weight_edit_token_p
            )
            for words, doc in batch
        ]
        return (
            [utils.de_tokenize_words(words, doc) for words, (_, doc) in zip(batch_corrupted, batch)],
            target_sequences,
            infos
        )


@dataclass
class WhitespaceNoiseConfig(PreprocessingConfig):
    type: Preprocessings = Preprocessings.WHITESPACE_NOISE

    no_whitespace_p: float = MISSING
    insert_whitespace_p: float = MISSING
    delete_whitespace_p: float = MISSING


class Whitespace(Preprocessing):
    def apply(
            self,
            sequences: List[str],
            target_sequences: List[str],
            infos: List[Dict[str, Any]],
            is_inference: bool = False
    ) -> Tuple[List[str], List[str], List[Dict[str, Any]]]:
        if is_inference:
            return sequences, target_sequences, infos
        self.cfg: WhitespaceNoiseConfig
        return (
            [
                corrupt_whitespace(
                    sequence,
                    iw_p=self.cfg.insert_whitespace_p,
                    dw_p=self.cfg.delete_whitespace_p,
                    no_ws_p=self.cfg.no_whitespace_p,
                    rand=self.rand
                )
                for sequence in sequences
            ],
            target_sequences,
            infos
        )


@dataclass
class ChainedConfig(PreprocessingConfig):
    type: Preprocessings = Preprocessings.CHAINED

    cfgs: List[PreprocessingConfig] = MISSING
    overrides: List[bool] = MISSING


class Chained(Preprocessing):
    def __init__(self, cfg: PreprocessingConfig, seed: int) -> None:
        super().__init__(cfg, seed)
        self.cfg: ChainedConfig
        assert len(self.cfg.cfgs) == len(self.cfg.overrides), \
            "expected same number of noise configs and override flags"
        assert not self.cfg.overrides[-1], "last override flag must always be False"
        self.preprocessing = [get_preprocessing_from_config(cfg, seed) for cfg in self.cfg.cfgs]
        self.overrides = self.cfg.overrides

    def apply(
            self,
            sequences: List[str],
            target_sequences: List[str],
            infos: List[Dict[str, Any]],
            is_inference: bool = False
    ) -> Tuple[List[str], List[str], List[Dict[str, Any]]]:
        for override, preprocessing in zip(self.overrides, self.preprocessing):
            sequences, target_sequences, infos = preprocessing.apply(sequences, target_sequences, infos, is_inference)
            if override:
                target_sequences = sequences
        return sequences, target_sequences, infos


@dataclass
class SwitchConfig(PreprocessingConfig):
    type: Preprocessings = Preprocessings.SWITCH
    cfgs: List[PreprocessingConfig] = MISSING
    probabilities: List[float] = MISSING


class Switch(Preprocessing):
    def __init__(self, cfg: SwitchConfig, seed: int) -> None:
        super().__init__(cfg, seed)
        self.cfg: SwitchConfig
        assert len(self.cfg.cfgs) == len(self.cfg.probabilities), \
            "expected same number of preprocessing configs and probabilities"
        assert sum(self.cfg.probabilities) == 1 and all(0 < p < 1 for p in self.cfg.probabilities), \
            "probabilities must be between 0 and 1 and must sum to one"
        self.preprocessing = [get_preprocessing_from_config(cfg, seed) for cfg in self.cfg.cfgs]
        self.probabilities = self.cfg.probabilities

    def apply(
            self,
            sequences: List[str],
            target_sequences: List[str],
            infos: List[Dict[str, Any]],
            is_inference: bool = False
    ) -> Tuple[List[str], List[str], List[Dict[str, Any]]]:
        idx = self.rand.choice(np.arange(len(self.preprocessing)), p=self.probabilities)
        return self.preprocessing[idx].apply(sequences, target_sequences, infos, is_inference)


@dataclass
class SubstringConfig(PreprocessingConfig):
    type: Preprocessings = Preprocessings.SUBSTRING
    max_length: int = 512


class Substring(Preprocessing):
    def apply(
            self,
            sequences: List[str],
            target_sequences: List[str],
            infos: List[Dict[str, Any]],
            is_inference: bool = False
    ) -> Tuple[List[str], List[str], List[Dict[str, Any]]]:
        self.cfg: SubstringConfig

        substring_sequences = []
        substring_target_sequences = []
        for sequence, target_sequence in zip(sequences, target_sequences):
            assert len(sequence) == len(target_sequence), \
                f"substring preprocessing should only be used for sequences that have the same length"
            if is_inference:
                sequence = sequence[:self.cfg.max_length]
                target_sequence = target_sequence[:self.cfg.max_length]
            else:
                start_idx = self.rand.integers(0, max(1, len(sequence) - self.cfg.max_length + 1))
                sequence = sequence[start_idx:start_idx + self.cfg.max_length]
                target_sequence = target_sequence[start_idx:start_idx + self.cfg.max_length]
            substring_sequences.append(sequence)
            substring_target_sequences.append(target_sequence)
        return substring_sequences, substring_target_sequences, infos


@dataclass
class NoneConfig(PreprocessingConfig):
    type: Preprocessings = Preprocessings.NONE


class NoPreprocessing(Preprocessing):
    def apply(
            self,
            sequences: List[str],
            target_sequences: List[str],
            infos: List[Dict[str, Any]],
            is_inference: bool = False
    ) -> Tuple[List[str], List[str], List[Dict[str, Any]]]:
        return sequences, target_sequences, infos


def get_preprocessing_from_config(cfg: Union[PreprocessingConfig, omegaconf.DictConfig], seed: int) -> Preprocessing:
    preprocessing_type = cfg.type if isinstance(cfg, PreprocessingConfig) else Preprocessings[cfg.type]
    if preprocessing_type == Preprocessings.ARTIFICIAL_NOISE:
        cfg = OmegaConf.structured(ArtificialNoiseConfig(**cfg))
        return ArtificialNoise(cfg, seed)
    elif preprocessing_type == Preprocessings.REALISTIC_NOISE:
        cfg = OmegaConf.structured(RealisticNoiseConfig(**cfg))
        return RealisticNoise(cfg, seed)
    elif preprocessing_type == Preprocessings.MIXED_NOISE:
        cfg = OmegaConf.structured(MixedNoiseConfig(**cfg))
        return MixedNoise(cfg, seed)
    elif preprocessing_type == Preprocessings.WHITESPACE_NOISE:
        cfg = OmegaConf.structured(WhitespaceNoiseConfig(**cfg))
        return Whitespace(cfg, seed)
    elif preprocessing_type == Preprocessings.CHAINED:
        cfg = OmegaConf.structured(ChainedConfig(**cfg))
        return Chained(cfg, seed)
    elif preprocessing_type == Preprocessings.SWITCH:
        cfg = OmegaConf.structured(SwitchConfig(**cfg))
        return Switch(cfg, seed)
    elif preprocessing_type == Preprocessings.NONE:
        cfg = OmegaConf.structured(NoneConfig(**cfg))
        return NoPreprocessing(cfg, seed)
    elif preprocessing_type == Preprocessings.SUBSTRING:
        cfg = OmegaConf.structured(SubstringConfig(**cfg))
        return Substring(cfg, seed)
    else:
        raise ValueError(f"Unknown noise {cfg.type.name}")


def get_preprocessing_fn(
        preprocessing: Preprocessing,
        tokenization_fn: TOKENIZATION_FN,
        neighbor_fn: Optional[NEIGHBOR_FN] = None,
        split_only_on_ws: bool = False,
        with_pos_tags: bool = False,
        with_ner: bool = False,
        with_dep_parser: bool = False,
        batch_size: Optional[int] = None
) -> PREPROCESSING_FN:
    def _preprocessing_fn(
            sequences: List[str],
            target_sequences: List[Optional[str]],
            is_inference: bool = False
    ) -> List[Tuple[SAMPLE, str]]:
        # initialize target sequences with target sequences if given else with input sequences
        target_sequences = [
            target_sequence if target_sequence is not None else sequence
            for sequence, target_sequence in zip(sequences, target_sequences)
        ]
        infos = [dict()] * len(sequences)
        sequences, target_sequences, infos = preprocessing.apply(sequences, target_sequences, infos, is_inference)
        samples = utils.prepare_samples(
            sequences,
            infos,
            tokenization_fn,
            neighbor_fn,
            split_only_on_ws=split_only_on_ws,
            with_pos_tags=with_pos_tags,
            with_ner=with_ner,
            with_dep_parser=with_dep_parser,
            batch_size=batch_size
        )
        return list(zip(samples, target_sequences))

    return _preprocessing_fn
