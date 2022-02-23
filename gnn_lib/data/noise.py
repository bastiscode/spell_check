import enum
import hashlib
import json
from dataclasses import dataclass
from typing import Tuple, List, Dict, Optional, Union

import numpy as np
import omegaconf
from omegaconf import MISSING, OmegaConf
from spacy.tokens import Doc

from gnn_lib.data import utils


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


class NoiseVariants(enum.IntEnum):
    ARTIFICIAL = 1
    REALISTIC = 2
    MIXED = 3
    WHITESPACE = 4
    CHAINED = 5
    SWITCH = 6
    NONE = 7


@dataclass
class NoiseVariantConfig:
    type: NoiseVariants = MISSING


class Noise:
    def __init__(self, cfg: NoiseVariantConfig, seed: int) -> None:
        self.cfg = cfg
        self.rand = np.random.default_rng(seed)

    @property
    def name(self) -> str:
        return self.cfg.type.name

    def apply(self, sequence: Union[str, utils.SAMPLE]) -> Tuple[str, str]:
        raise NotImplementedError

    @property
    def cfg_string(self) -> str:
        cfg_yaml = OmegaConf.to_yaml(self.cfg, resolve=True, sort_keys=True)
        return str(hashlib.sha1(cfg_yaml.encode("utf8")).hexdigest())


@dataclass
class ArtificialConfig(NoiseVariantConfig):
    type: NoiseVariants = NoiseVariants.ARTIFICIAL
    edit_token_p: float = MISSING
    num_edits_p: float = MISSING
    re_weight_edit_token_p: bool = True


class Artificial(Noise):
    def apply(self, sequence: Union[str, utils.SAMPLE]) -> Tuple[str, str]:
        self.cfg: ArtificialConfig
        if isinstance(sequence, str):
            words, doc = utils.tokenize_words(sequence, return_doc=True)
        else:
            words, doc = [w.text for w in sequence.doc], sequence.doc
        words = corrupt_sequence(
            words=words,
            doc=doc,
            edit_token_p=self.cfg.edit_token_p,
            num_edits_p=self.cfg.num_edits_p,
            corrupt_method="artificial",
            rand=self.rand,
            re_weight_edit_token_p=self.cfg.re_weight_edit_token_p
        )
        return str(sequence), utils.de_tokenize_words(words, doc)


@dataclass
class RealisticConfig(NoiseVariantConfig):
    type: NoiseVariants = NoiseVariants.REALISTIC
    edit_token_p: float = MISSING
    word_misspellings_file: str = MISSING
    re_weight_edit_token_p: bool = True


class Realistic(Noise):
    def __init__(self, cfg: NoiseVariantConfig, seed: int) -> None:
        super().__init__(cfg, seed)
        self.cfg: RealisticConfig
        with open(self.cfg.word_misspellings_file, "r", encoding="utf8") as inf:
            self.word_misspellings = json.load(inf)

    def apply(self, sequence: Union[str, utils.SAMPLE]) -> Tuple[str, str]:
        self.cfg: RealisticConfig
        if isinstance(sequence, str):
            words, doc = utils.tokenize_words(sequence, return_doc=True)
        else:
            words, doc = [w.text for w in sequence.doc], sequence.doc
        words = corrupt_sequence(
            words=words,
            doc=doc,
            num_edits_p=0,  # not used for realistic noise
            edit_token_p=self.cfg.edit_token_p,
            corrupt_method="realistic",
            rand=self.rand,
            word_misspellings=self.word_misspellings,
            re_weight_edit_token_p=self.cfg.re_weight_edit_token_p
        )
        return str(sequence), utils.de_tokenize_words(words, doc)


@dataclass
class MixedConfig(NoiseVariantConfig):
    type: NoiseVariants = NoiseVariants.MIXED
    edit_token_p: float = MISSING
    artificial_num_edits_p: float = MISSING
    word_misspellings_file: str = MISSING
    artificial_p: float = MISSING
    re_weight_edit_token_p: bool = True


class Mixed(Noise):
    def __init__(self, cfg: NoiseVariantConfig, seed: int) -> None:
        super().__init__(cfg, seed)
        self.cfg: MixedConfig
        with open(self.cfg.word_misspellings_file, "r", encoding="utf8") as inf:
            self.word_misspellings = json.load(inf)

    def apply(self, sequence: Union[str, utils.SAMPLE]) -> Tuple[str, str]:
        self.cfg: MixedConfig
        if isinstance(sequence, str):
            words, doc = utils.tokenize_words(sequence, return_doc=True)
        else:
            words, doc = [w.text for w in sequence.doc], sequence.doc
        words = corrupt_sequence(
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
        return str(sequence), utils.de_tokenize_words(words, doc)


@dataclass
class WhitespaceConfig(NoiseVariantConfig):
    type: NoiseVariants = NoiseVariants.WHITESPACE
    no_whitespace_p: float = MISSING
    insert_whitespace_p: float = MISSING
    delete_whitespace_p: float = MISSING


class Whitespace(Noise):
    def apply(self, sequence: Union[str, utils.SAMPLE]) -> Tuple[str, str]:
        self.cfg: WhitespaceConfig
        return sequence, corrupt_whitespace(
            str(sequence),
            iw_p=self.cfg.insert_whitespace_p,
            dw_p=self.cfg.delete_whitespace_p,
            no_ws_p=self.cfg.no_whitespace_p,
            rand=self.rand
        )


@dataclass
class ChainedConfig(NoiseVariantConfig):
    type: NoiseVariants = NoiseVariants.CHAINED
    noise_cfgs: List[NoiseVariantConfig] = MISSING
    overrides: List[bool] = MISSING


class Chained(Noise):
    def __init__(self, cfg: NoiseVariantConfig, seed: int) -> None:
        super().__init__(cfg, seed)
        self.cfg: ChainedConfig
        assert len(self.cfg.noise_cfgs) == len(self.cfg.overrides), \
            "expected same number of noise configs and override flags"
        assert not self.cfg.overrides[-1], "last override flag must always be False"
        self.noises = [get_noise_from_config(cfg, seed) for cfg in self.cfg.noise_cfgs]
        self.overrides = self.cfg.overrides

    def apply(self, sequence: Union[str, utils.SAMPLE]) -> Tuple[str, str]:
        seq = str(sequence)
        corr_seq = str(sequence)
        for override, noise in zip(self.overrides, self.noises):
            _, corr_seq = noise.apply(corr_seq)
            if override:
                seq = corr_seq
        return seq, corr_seq


@dataclass
class SwitchConfig(NoiseVariantConfig):
    type: NoiseVariants = NoiseVariants.SWITCH
    noise_cfgs: List[NoiseVariantConfig] = MISSING
    probabilities: List[float] = MISSING


class Switch(Noise):
    def __init__(self, cfg: SwitchConfig, seed: int) -> None:
        super().__init__(cfg, seed)
        self.cfg: SwitchConfig
        assert len(self.cfg.noise_cfgs) == len(self.cfg.probabilities), \
            "expected same number of noise configs and probabilities"
        assert sum(self.cfg.probabilities) == 1 and all(0 < p < 1 for p in self.cfg.probabilities), \
            "probabilities must be between 0 and 1 and must sum to one"
        self.noises = [get_noise_from_config(cfg, seed) for cfg in self.cfg.noise_cfgs]
        self.probabilities = self.cfg.probabilities

    def apply(self, sequence: Union[str, utils.SAMPLE]) -> Tuple[str, str]:
        noise_idx = self.rand.choice(np.arange(len(self.noises)), p=self.probabilities)
        return self.noises[noise_idx].apply(sequence)


@dataclass
class NoneConfig(NoiseVariantConfig):
    type: NoiseVariants = NoiseVariants.NONE


class NoneNoise(Noise):
    def __init__(self, cfg: NoiseVariantConfig, seed: int) -> None:
        super().__init__(cfg, seed)

    def apply(self, sequence: Union[str, utils.SAMPLE]) -> Tuple[str, str]:
        return str(sequence), str(sequence)


def get_noise_from_config(cfg: omegaconf.DictConfig, seed: int) -> Noise:
    noise_type = NoiseVariants[cfg.type]
    if noise_type == NoiseVariants.ARTIFICIAL:
        cfg = OmegaConf.structured(ArtificialConfig(**cfg))
        return Artificial(cfg, seed)
    elif noise_type == NoiseVariants.REALISTIC:
        cfg = OmegaConf.structured(RealisticConfig(**cfg))
        return Realistic(cfg, seed)
    elif noise_type == NoiseVariants.MIXED:
        cfg = OmegaConf.structured(MixedConfig(**cfg))
        return Mixed(cfg, seed)
    elif noise_type == NoiseVariants.WHITESPACE:
        cfg = OmegaConf.structured(WhitespaceConfig(**cfg))
        return Whitespace(cfg, seed)
    elif noise_type == NoiseVariants.CHAINED:
        cfg = OmegaConf.structured(ChainedConfig(**cfg))
        return Chained(cfg, seed)
    elif noise_type == NoiseVariants.SWITCH:
        cfg = OmegaConf.structured(SwitchConfig(**cfg))
        return Switch(cfg, seed)
    elif noise_type == NoiseVariants.NONE:
        cfg = OmegaConf.structured(NoneConfig(**cfg))
        return NoneNoise(cfg, seed)
    else:
        raise ValueError(f"Unknown noise {cfg.type.name}")
