import copy
import enum
import hashlib
import json
import re
import string
from dataclasses import dataclass
from typing import Tuple, List, Dict, Optional, Union, Any, Set

import numpy as np
import omegaconf
from omegaconf import MISSING, OmegaConf
from spacy.tokens import Doc

from nsc.data import utils
from nsc.data.utils import PreprocessingFn, Sample, TokenizationFn, NeighborFn

_INCLUDE_ALL = tuple(i for i in range(4))
_EDIT_CHARS = tuple(string.ascii_letters)


def edit_token(
        token: str,
        rand: np.random.Generator,
        include: Tuple[int, ...] = _INCLUDE_ALL,
        edit_chars: Tuple[str, ...] = _EDIT_CHARS,
        exclude_indices: Optional[Set[int]] = None
) -> Tuple[str, Set[int]]:
    """

    Perform a random edit operation from
    {insert, delete, swap, replace} with the token.

    :param token: token string
    :param rand: random state
    :param include: list of integers that represent
    edit operations from which should be chosen
    :param edit_chars: list of strings to choose
    from for inserting and replacing
    :param exclude_indices: list of indices to not consider for editing,
    useful if you want to prevent that indices are edited multiple times
    :return: token with one random edit, list of ints
    indicating the edits for the positions in the token, updated exclude indices
    """
    exclude_indices = exclude_indices or set()
    if len(token) == 0:
        return token, set()

    # edit methods: 0 -> insert, 1 -> delete, 2 -> swap, 3 -> replace
    # pre-filter edit methods based on token length
    def _valid_edit_method(m: int) -> bool:
        assert m in {0, 1, 2, 3}, f"invalid edit method {m}, must be in {{0, 1, 2, 3}}"
        # only delete if there is more than one character
        if m == 1 and len(token) <= 1:
            return False
        # only swap if there is more than one character
        elif m == 2 and len(token) <= 1:
            return False
        # we can always replace or insert
        return True

    include = list(filter(lambda m: _valid_edit_method(m), include))
    edit_method = rand.choice(include)

    if edit_method == 0:
        insert_indices = set()
        # only insert between two not excluded characters (or at beginning or at end)
        for i in range(-1, len(token)):
            if i in exclude_indices or i + 1 in exclude_indices:
                continue
            insert_indices.add(i + 1)

        if len(insert_indices) > 0:
            char_idx = rand.integers(len(edit_chars))
            _insert_char = edit_chars[char_idx]
            assert len(_insert_char) > 0, "dont insert empty string, this is equal to leaving the token unchanged"

            token_idx = rand.choice(list(insert_indices))
            assert token_idx not in exclude_indices

            token = token[:token_idx] + _insert_char + token[token_idx:]

            # we insert some chars, so the length of the token changed
            # adjust exclude indices accordingly: indices right of the inserted chars must be increased by
            # the number of inserted chars
            exclude_indices = set(idx + len(_insert_char) if idx >= token_idx else idx for idx in exclude_indices)
            for i in range(len(_insert_char)):
                exclude_indices.add(token_idx + i)

    elif edit_method == 1:
        delete_indices = set(range(len(token))) - exclude_indices

        if len(delete_indices) > 0:
            token_idx = rand.choice(list(delete_indices))
            assert token_idx not in exclude_indices
            token = token[:token_idx] + token[token_idx + 1:]

            # we deleted a character, so the length of the token changed
            # adjust exclude indices accordingly: indices right of the deleted char must be decreased by 1
            exclude_indices = set(idx - 1 if idx > token_idx else idx for idx in exclude_indices)

    elif edit_method == 2:
        swap_indices = set()
        for i in range(len(token) - 1):
            if i in exclude_indices or i + 1 in exclude_indices:
                continue
            swap_indices.add(i)

        if len(swap_indices) > 0:
            token_idx = rand.choice(list(swap_indices))
            assert token_idx not in exclude_indices and token_idx + 1 not in exclude_indices

            token = token[:token_idx] + token[token_idx + 1] + token[token_idx] + token[token_idx + 2:]

            # we swapped two characters, length of the token did not change, just add the token indices to the
            # exclude indices
            exclude_indices.add(token_idx)
            exclude_indices.add(token_idx + 1)

    else:
        replace_indices = set(range(len(token))) - exclude_indices

        if len(replace_indices) > 0:
            token_idx = rand.choice(list(replace_indices))
            assert token_idx not in exclude_indices

            new_char = token[token_idx]
            while new_char == token[token_idx]:
                new_char = edit_chars[rand.integers(len(edit_chars))]
            assert len(new_char) > 0, "dont replace chars with empty strings, delete should be used for that"

            token = token[:token_idx] + new_char + token[token_idx + 1:]

            exclude_indices = set(idx + len(new_char) - 1 if idx >= token_idx + 1 else idx for idx in exclude_indices)
            for i in range(len(new_char)):
                exclude_indices.add(token_idx + i)

    return token, exclude_indices


def find_substring_ignoring_spaces(
        substring: str,
        search_str: str
) -> Tuple[int, int]:
    pattern = r"\s?".join(re.escape(char) for char in substring.replace(" ", ""))
    match = re.search(pattern, search_str)
    assert match is not None
    return match.start(), match.end()


def find_word_boundaries_ignoring_space(
        sequence: str,
        search_str: str
) -> List[Tuple[int, int]]:
    pattern = r"\s?"
    for word in sequence.split():
        pattern += "(" + r"\s?".join(re.escape(char) for char in word) + r")\s?"
    match = re.fullmatch(pattern, search_str)
    assert match is not None
    word_boundaries = []
    for g in range(len(match.groups())):
        word_boundaries.append((match.start(g + 1), match.end(g + 1)))
    return word_boundaries


def artificial_edits(word: str, num_edits: int, rand: np.random.Generator) -> str:
    exclude_indices = set()
    for _ in range(num_edits):
        word, exclude_indices = edit_token(
            token=word,
            rand=rand,
            exclude_indices=exclude_indices
        )
    return word


def realistic_edits(
        word: str,
        rand: np.random.Generator,
        word_misspellings: Dict[str, List[str]]
) -> str:
    pot_misspellings = word_misspellings.get(word, word_misspellings.get(word.lower()))
    if pot_misspellings is not None and len(pot_misspellings) > 0:
        return pot_misspellings[rand.integers(len(pot_misspellings))]
    else:
        return word


# tokens_edited = 0
# tokens_seen = 0


def corrupt_words(
        words: List[str],
        doc: Doc,
        edit_token_p: float,
        num_edits_p: float,
        corrupt_method: str,
        rand: np.random.Generator,
        word_misspellings: Optional[Dict[str, List[str]]] = None,
        mixed_artificial_p: float = 0.2,
        re_weight_edit_token_p: bool = False,
        min_edit_tokens: Optional[int] = None
) -> List[str]:
    assert len(words) == len(doc) and len(words) > 0

    special_tokens_mask = np.array([utils.is_special_token(doc[i]) for i in range(len(words))], dtype=bool)

    artificial_p = 0
    realistic_p = 0
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
                        len(words) / max(1, num_non_special_and_in_misspellings_tokens)
                ) * (1 - mixed_artificial_p)
    else:
        if corrupt_method == "artificial":
            artificial_p = edit_token_p
        elif corrupt_method == "realistic":
            realistic_p = edit_token_p
        else:
            artificial_p = edit_token_p * mixed_artificial_p
            # equal to edit_token_p but here for clarity
            realistic_p = artificial_p + edit_token_p * (1 - mixed_artificial_p)

    edited_words = copy.deepcopy(words)
    for word_idx in range(len(words)):
        if special_tokens_mask[word_idx]:  # do not edit special tokens
            continue

        word = edited_words[word_idx]

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
            raise ValueError(f"unknown corrupt method {corrupt_method}")

        edited_words[word_idx] = word

    if min_edit_tokens is not None:
        assert min_edit_tokens > 0
        min_edit_tokens = min(min_edit_tokens, len(words))
        edited = [edited_word != word for edited_word, word in zip(edited_words, words)]
        if sum(edited) < min_edit_tokens:
            editable_indices = [
                i for i, was_edited in enumerate(edited)
                if not was_edited and not special_tokens_mask[i]
            ]
            edit_indices = rand.choice(
                editable_indices,
                size=min(len(editable_indices), min_edit_tokens - sum(edited)),
                replace=False
            )
            for idx in edit_indices:
                edited_words[idx] = artificial_edits(edited_words[idx], 1, rand)

    return edited_words


def corrupt_whitespace(
        sequence: str,
        iw_p: float,
        dw_p: float,
        no_ws_p: float,
        full_ws_p: float,
        rand: np.random.Generator) -> str:
    r = rand.random()
    if r < no_ws_p:
        return sequence.replace(" ", "")
    elif r < no_ws_p + full_ws_p:
        return " ".join(sequence.replace(" ", ""))
    else:
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
    SAVE = 9
    REPLACE = 10


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
            infos: List[Dict[str, Any]]
    ) -> Tuple[List[str], List[str], List[Dict[str, Any]]]:
        raise NotImplementedError

    @property
    def cfg_string(self) -> str:
        cfg_yaml = OmegaConf.to_yaml(self.cfg, resolve=True, sort_keys=True)
        return str(hashlib.sha1(cfg_yaml.encode("utf8")).hexdigest())


@dataclass
class SaveConfig(PreprocessingConfig):
    save_sequence_as: Optional[str] = None
    save_target_sequence_as: Optional[str] = None


class Save(Preprocessing):
    def apply(
            self,
            sequences: List[str],
            target_sequences: List[str],
            infos: List[Dict[str, Any]]
    ) -> Tuple[List[str], List[str], List[Dict[str, Any]]]:
        self.cfg: SaveConfig
        if self.cfg.save_sequence_as is None and self.cfg.save_target_sequence_as is None:
            return sequences, target_sequences, infos

        new_infos = []
        for sequence, target_sequence, info in zip(sequences, target_sequences, infos):
            info = copy.deepcopy(info)
            if self.cfg.save_sequence_as is not None:
                info[self.cfg.save_sequence_as] = sequence
            if self.cfg.save_target_sequence_as is not None:
                info[self.cfg.save_target_sequence_as] = target_sequence
            new_infos.append(info)
        return sequences, target_sequences, new_infos


@dataclass
class NoiseConfig(PreprocessingConfig):
    edit_token_p: float = MISSING
    min_edit_tokens: Optional[int] = None
    re_weight_edit_token_p: bool = True


class Noise(Preprocessing):
    def _noise_words(self, words: List[str], doc: Doc) -> List[str]:
        raise NotImplementedError

    def apply(
            self,
            sequences: List[str],
            target_sequences: List[str],
            infos: List[Dict[str, Any]]
    ) -> Tuple[List[str], List[str], List[Dict[str, Any]]]:
        self.cfg: NoiseConfig
        batch_corrupted = [
            utils.de_tokenize_words(self._noise_words(words, doc), doc)
            for words, doc in utils.tokenize_words_batch(sequences, return_docs=True)
        ]
        return batch_corrupted, target_sequences, infos


@dataclass
class ArtificialNoiseConfig(NoiseConfig):
    type: Preprocessings = Preprocessings.ARTIFICIAL_NOISE

    num_edits_p: float = MISSING


class ArtificialNoise(Noise):
    def _noise_words(self, words: List[str], doc: Doc) -> List[str]:
        self.cfg: ArtificialNoiseConfig
        return corrupt_words(
            words=words,
            doc=doc,
            edit_token_p=self.cfg.edit_token_p,
            min_edit_tokens=self.cfg.min_edit_tokens,
            num_edits_p=self.cfg.num_edits_p,
            corrupt_method="artificial",
            rand=self.rand,
            re_weight_edit_token_p=self.cfg.re_weight_edit_token_p
        )


@dataclass
class RealisticNoiseConfig(NoiseConfig):
    type: Preprocessings = Preprocessings.REALISTIC_NOISE

    word_misspellings_file: str = MISSING


class RealisticNoise(Noise):
    def __init__(self, cfg: PreprocessingConfig, seed: int) -> None:
        super().__init__(cfg, seed)
        self.cfg: RealisticNoiseConfig
        with open(self.cfg.word_misspellings_file, "r", encoding="utf8") as inf:
            self.word_misspellings = json.load(inf)

    def _noise_words(self, words: List[str], doc: Doc) -> List[str]:
        self.cfg: RealisticNoiseConfig
        return corrupt_words(
            words=words,
            doc=doc,
            edit_token_p=self.cfg.edit_token_p,
            min_edit_tokens=self.cfg.min_edit_tokens,
            num_edits_p=0,  # not used for realistic noise
            corrupt_method="realistic",
            rand=self.rand,
            word_misspellings=self.word_misspellings,
            re_weight_edit_token_p=self.cfg.re_weight_edit_token_p
        )


@dataclass
class MixedNoiseConfig(NoiseConfig):
    type: Preprocessings = Preprocessings.MIXED_NOISE

    artificial_num_edits_p: float = MISSING
    word_misspellings_file: str = MISSING
    artificial_p: float = MISSING


class MixedNoise(Noise):
    def __init__(self, cfg: PreprocessingConfig, seed: int) -> None:
        super().__init__(cfg, seed)
        self.cfg: MixedNoiseConfig
        with open(self.cfg.word_misspellings_file, "r", encoding="utf8") as inf:
            self.word_misspellings = json.load(inf)

    def _noise_words(self, words: List[str], doc: Doc) -> List[str]:
        self.cfg: MixedNoiseConfig
        return corrupt_words(
            words=words,
            doc=doc,
            edit_token_p=self.cfg.edit_token_p,
            min_edit_tokens=self.cfg.min_edit_tokens,
            num_edits_p=self.cfg.artificial_num_edits_p,
            corrupt_method="mixed",
            rand=self.rand,
            word_misspellings=self.word_misspellings,
            mixed_artificial_p=self.cfg.artificial_p,
            re_weight_edit_token_p=self.cfg.re_weight_edit_token_p
        )


@dataclass
class WhitespaceNoiseConfig(PreprocessingConfig):
    type: Preprocessings = Preprocessings.WHITESPACE_NOISE

    no_whitespace_p: float = MISSING
    full_whitespace_p: float = MISSING
    insert_whitespace_p: float = MISSING
    delete_whitespace_p: float = MISSING


class WhitespaceNoise(Preprocessing):
    def apply(
            self,
            sequences: List[str],
            target_sequences: List[str],
            infos: List[Dict[str, Any]]
    ) -> Tuple[List[str], List[str], List[Dict[str, Any]]]:
        self.cfg: WhitespaceNoiseConfig
        return (
            [
                corrupt_whitespace(
                    sequence,
                    iw_p=self.cfg.insert_whitespace_p,
                    dw_p=self.cfg.delete_whitespace_p,
                    no_ws_p=self.cfg.no_whitespace_p,
                    full_ws_p=self.cfg.full_whitespace_p,
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


class Chained(Preprocessing):
    def __init__(self, cfg: PreprocessingConfig, seed: int) -> None:
        super().__init__(cfg, seed)
        self.cfg: ChainedConfig
        self.preprocessing = [get_preprocessing_from_config(cfg, seed) for cfg in self.cfg.cfgs]

    def apply(
            self,
            sequences: List[str],
            target_sequences: List[str],
            infos: List[Dict[str, Any]]
    ) -> Tuple[List[str], List[str], List[Dict[str, Any]]]:
        for preprocessing in self.preprocessing:
            sequences, target_sequences, infos = preprocessing.apply(sequences, target_sequences, infos)
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
            infos: List[Dict[str, Any]]
    ) -> Tuple[List[str], List[str], List[Dict[str, Any]]]:
        idx = self.rand.choice(np.arange(len(self.preprocessing)), p=self.probabilities)
        return self.preprocessing[idx].apply(sequences, target_sequences, infos)


@dataclass
class SubstringConfig(PreprocessingConfig):
    type: Preprocessings = Preprocessings.SUBSTRING
    max_length: int = 512
    unit: str = "char"  # one of {byte, char}
    respect_word_boundaries: bool = False


class Substring(Preprocessing):
    def _get_start_end(
            self,
            sequence: str,
            target_sequence: str
    ) -> Optional[Tuple[Tuple[int, int], Tuple[int, int]]]:
        self.cfg: SubstringConfig
        if self.cfg.respect_word_boundaries:
            word_boundaries = find_word_boundaries_ignoring_space(target_sequence, sequence)
            if self.cfg.unit == "char":
                num_elements_per_word = [int(i > 0) + word_end - word_start
                                         for i, (word_start, word_end) in enumerate(word_boundaries)]
            elif self.cfg.unit == "byte":
                num_elements_per_word = [len((" " * (i > 0) + sequence[word_start:word_end]).encode("utf8"))
                                         for i, (word_start, word_end) in enumerate(word_boundaries)]
            else:
                raise RuntimeError(f"unknown unit {self.cfg.unit}, must be one of {{char, byte}}")

            if any(num_elements > self.cfg.max_length for num_elements in num_elements_per_word):
                return

            starting_indices = [0]
            if sum(num_elements_per_word) > self.cfg.max_length:
                for i in range(1, len(num_elements_per_word)):
                    if sum(num_elements_per_word[i:]) >= self.cfg.max_length:
                        starting_indices.append(i)
                    else:
                        break

            word_start_idx = starting_indices[self.rand.integers(0, len(starting_indices))]
            word_end_idx = word_start_idx + 1
            while (
                    word_end_idx < len(num_elements_per_word)
                    and sum(num_elements_per_word[word_start_idx:word_end_idx])
                    + num_elements_per_word[word_end_idx] <= self.cfg.max_length
            ):
                word_end_idx += 1

            assert sum(num_elements_per_word[word_start_idx:word_end_idx]) <= self.cfg.max_length

            start_idx = word_boundaries[word_start_idx][0]
            end_idx = word_boundaries[word_end_idx - 1][1]

        else:
            if self.cfg.unit == "char":
                start_idx = self.rand.integers(0, max(0, len(sequence) - self.cfg.max_length) + 1)
                end_idx = start_idx + self.cfg.max_length

            elif self.cfg.unit == "byte":
                byte_lengths = [len(char.encode("utf8")) for char in sequence]

                starting_indices = [0]
                if sum(byte_lengths) > self.cfg.max_length:
                    for i in range(1, len(byte_lengths)):
                        if sum(byte_lengths[i:]) >= self.cfg.max_length:
                            starting_indices.append(i)
                        else:
                            break

                start_idx = starting_indices[self.rand.integers(0, len(starting_indices))]
                end_idx = start_idx + 1
                while (
                        end_idx < len(byte_lengths)
                        and sum(byte_lengths[start_idx:end_idx]) + byte_lengths[end_idx] <= self.cfg.max_length
                ):
                    end_idx += 1

                assert sum(byte_lengths[start_idx:end_idx]) <= self.cfg.max_length

            else:
                raise RuntimeError(f"unknown unit {self.cfg.unit}, must be one of {{char, byte}}")

        target_start_idx, target_end_idx = find_substring_ignoring_spaces(
            sequence[start_idx: end_idx], target_sequence
        )

        return (start_idx, end_idx), (target_start_idx, target_end_idx)

    def apply(
            self,
            sequences: List[str],
            target_sequences: List[str],
            infos: List[Dict[str, Any]]
    ) -> Tuple[List[str], List[str], List[Dict[str, Any]]]:
        self.cfg: SubstringConfig

        substring_sequences = []
        substring_target_sequences = []
        new_infos = []
        for sequence, target_sequence, info in zip(sequences, target_sequences, infos):
            indices = self._get_start_end(sequence, target_sequence)
            if indices is None:
                print("found invalid inputs to substring preprocessing")
                continue

            (start, end), (target_start, target_end) = indices

            substring_sequences.append(sequence[start: end])
            substring_target_sequences.append(target_sequence[target_start: target_end])
            if "org_sequence" in info:
                assert self.cfg.respect_word_boundaries, \
                    "org_sequence is mapped to target sequence using words, so respect_word_boundaries must be true"
                info = copy.deepcopy(info)
                org_sequence = info["org_sequence"]
                org_words = org_sequence.split()
                assert len(target_sequence.split()) == len(org_sequence.split())
                # filter original sequence
                num_words_before = len(target_sequence[:target_start].split())
                num_window_words = len(target_sequence[target_start:target_end].split())
                num_words = len(target_sequence[:target_end].split())
                assert num_words_before + num_window_words == num_words
                org_words = org_words[num_words_before:num_words]
                assert len(org_words) == num_window_words
                info["org_sequence"] = " ".join(org_words)
            new_infos.append(info)

        return substring_sequences, substring_target_sequences, new_infos


@dataclass
class ReplaceConfig(PreprocessingConfig):
    type: Preprocessings = Preprocessings.REPLACE

    replace: str = MISSING


class Replace(Preprocessing):
    def __init__(self, cfg: ReplaceConfig, seed: int) -> None:
        super().__init__(cfg, seed)
        self.cfg: ReplaceConfig
        assert self.cfg.replace in {"target_with_input", "input_with_target"}, \
            f"replace must be one of {{target_with_input, input_with_target}}, but got {self.cfg.replace}"

    def apply(
            self,
            sequences: List[str],
            target_sequences: List[str],
            infos: List[Dict[str, Any]]
    ) -> Tuple[List[str], List[str], List[Dict[str, Any]]]:
        self.cfg: ReplaceConfig

        if self.cfg.replace == "input_with_target":
            sequences = target_sequences
        else:
            target_sequences = sequences

        return sequences, target_sequences, infos


@dataclass
class NoneConfig(PreprocessingConfig):
    type: Preprocessings = Preprocessings.NONE


class NoPreprocessing(Preprocessing):
    def apply(
            self,
            sequences: List[str],
            target_sequences: List[str],
            infos: List[Dict[str, Any]]
    ) -> Tuple[List[str], List[str], List[Dict[str, Any]]]:
        return sequences, target_sequences, infos


def get_preprocessing_from_config(cfg: Union[PreprocessingConfig, omegaconf.DictConfig], seed: int) -> Preprocessing:
    # explicitly convert to dict config first, this way we support both dictconfigs
    # and structured configs as input
    cfg: omegaconf.DictConfig = omegaconf.DictConfig(cfg)
    preprocessing_type = Preprocessings[cfg.type] if isinstance(cfg.type, str) else cfg.type
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
        return WhitespaceNoise(cfg, seed)
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
    elif preprocessing_type == Preprocessings.SAVE:
        cfg = OmegaConf.structured(SaveConfig(**cfg))
        return Save(cfg, seed)
    elif preprocessing_type == Preprocessings.REPLACE:
        cfg = OmegaConf.structured(ReplaceConfig(**cfg))
        return Replace(cfg, seed)
    else:
        raise ValueError(f"unknown noise {cfg.type.name}")


def get_preprocessing_fn(
        preprocessing: Preprocessing,
        tokenization_fn: TokenizationFn,
        neighbor_fn: Optional[NeighborFn] = None,
        with_pos_tags: bool = False,
        with_ner: bool = False,
        with_dep_parser: bool = False,
        batch_size: Optional[int] = None
) -> PreprocessingFn:
    def _preprocessing_fn(
            sequences: List[str],
            target_sequences: List[Optional[str]],
            is_inference: bool = False
    ) -> List[Tuple[Sample, str]]:
        # initialize target sequences with target sequences if given else with input sequences
        target_sequences = [
            None if is_inference else (target_sequence or sequence)
            for sequence, target_sequence in zip(sequences, target_sequences)
        ]
        infos = [dict()] * len(sequences)

        if not is_inference:
            sequences, target_sequences, infos = preprocessing.apply(sequences, target_sequences, infos)

        samples = utils.prepare_samples(
            sequences,
            infos,
            tokenization_fn,
            neighbor_fn,
            with_pos_tags=with_pos_tags,
            with_ner=with_ner,
            with_dep_parser=with_dep_parser,
            batch_size=batch_size
        )
        return list(zip(samples, target_sequences))

    return _preprocessing_fn
