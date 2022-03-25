import collections
import json
import random
import re
import string
from collections import Counter
from typing import Union, List, Tuple, Optional, Iterator, Any, Dict, Iterable, Set, Callable

import dgl
import ftfy
import lmdb
import numpy as np
import spacy
import torch
from spacy import Vocab
from spacy.tokens import Token, Doc
from torch import distributed as dist
from torch.utils.data import Sampler, DistributedSampler, Dataset

from gnn_lib.utils import DATA_INPUT, BATCH
from gnn_lib.utils import common
from gnn_lib.utils.distributed import DistributedDevice


class NEIGHBORS(collections.namedtuple("NEIGHBORS", ["words", "left_contexts", "right_contexts", "distances"])):
    __slots__ = ()

    def __str__(self) -> str:
        return "\n".join(l + w + r for l, w, r in zip(self.left_contexts, self.words, self.right_contexts))


class SAMPLE(collections.namedtuple("SAMPLE", ["tokens", "doc", "neighbors_list", "info"])):
    __slots__ = ()

    def __str__(self) -> str:
        return str(self.doc)


PREPROCESSING_FN = Callable[[List[str], List[Optional[str]], bool], List[Tuple[SAMPLE, str]]]


def tokenize_words_regex(sequence: str) -> Tuple[List[str], List[bool]]:
    pattern = re.compile(r"\w+\S*\w+|\w+|\S")
    words = []
    whitespaces = []
    last_end = -1
    for match in pattern.finditer(sequence.strip()):
        words.append(match.group())
        if last_end >= 0:
            whitespaces.append(last_end < match.start())
        last_end = match.end()
    if len(words) > 0:
        whitespaces.append(sequence.endswith(" "))
    return words, whitespaces


class SpacyRegexTokenizer:
    def __init__(self, vocab: Vocab) -> None:
        self.vocab = vocab

    def __call__(self, text: str) -> Doc:
        words, whitespaces = tokenize_words_regex(text)
        return Doc(self.vocab, words=words, spaces=whitespaces)


class SpacyWhitespaceTokenizer:
    def __init__(self, vocab: Vocab) -> None:
        self.vocab = vocab

    def __call__(self, text: str) -> Doc:
        words = text.strip().split()
        whitespaces = [True] * len(words)
        whitespaces[-1] = False
        return Doc(self.vocab, words=words, spaces=whitespaces)


SPACY_TOKENIZER_REGEX = spacy.load("en_core_web_lg")
SPACY_TOKENIZER_REGEX.tokenizer = SpacyRegexTokenizer(SPACY_TOKENIZER_REGEX.vocab)
SPACY_TOKENIZER_WS = spacy.load("en_core_web_lg")
SPACY_TOKENIZER_WS.tokenizer = SpacyWhitespaceTokenizer(SPACY_TOKENIZER_WS.vocab)

SPACY_NER_MAP = {label: i for i, label in enumerate(SPACY_TOKENIZER_REGEX.get_pipe("ner").labels)}
# Universal pos tags
_UNIVERSAL_POS_TAGS = ("ADJ", "ADP", "ADV", "AUX", "CCONJ", "DET", "INTJ", "NOUN",
                       "NUM", "PART", "PRON", "PROPN", "PUNCT", "SCONJ", "SYM", "VERB", "X")
SPACY_POS_TAG_MAP = {tag: i for i, tag in enumerate(_UNIVERSAL_POS_TAGS)}
# Uncomment this for more fine grained english pos tags, but then also change graph.py .pos_ to .tag_
# SPACY_POS_TAG_MAP = {tag: i for i, tag in enumerate(SPACY_TOKENIZER_REGEX.get_pipe("tagger").labels)}
SPACY_DEP_TAG_MAP = {tag: i for i, tag in enumerate(SPACY_TOKENIZER_REGEX.get_pipe("parser").labels)}
SPACY_NUM_NER_TYPES = len(SPACY_NER_MAP)
SPACY_NUM_POS_TAGS = len(SPACY_POS_TAG_MAP)
SPACY_NUM_DEP_TAGS = len(SPACY_DEP_TAG_MAP)


def tokenize_words_batch(
        sequences: Iterable[str],
        return_docs: bool = False,
        split_only_on_ws: bool = False,
        with_pos_tags: bool = False,
        with_ner: bool = False,
        with_dep_parser: bool = False,
        batch_size: Optional[int] = None,
        num_processes: int = 1
) -> List[Union[List[str], Tuple[List[str], Doc]]]:
    tok = SPACY_TOKENIZER_WS if split_only_on_ws else SPACY_TOKENIZER_REGEX
    disable = []
    if not with_pos_tags:
        disable.append("tagger")
        disable.append("lemmatizer")
    if not with_ner:
        disable.append("ner")
    if not with_dep_parser:
        disable.append("parser")
    outputs = []
    for doc in tok.pipe(
            sequences,
            disable=disable,
            batch_size=batch_size,
            n_process=num_processes
    ):
        words = [w.text for w in doc]
        if return_docs:
            outputs.append((words, doc))
        else:
            outputs.append(words)
    return outputs


def tokenize_words(
        sequence: str,
        return_doc: bool = False,
        split_only_on_ws: bool = False,
        with_pos_tags: bool = False,
        with_ner: bool = False,
        with_dep_parser: bool = False
) -> Union[List[str], Tuple[List[str], Doc]]:
    return tokenize_words_batch(
        [sequence],
        return_doc,
        split_only_on_ws,
        with_pos_tags,
        with_ner,
        with_dep_parser
    )[0]


TOKENIZATION_FN = Callable[[Doc], List[List[int]]]
NEIGHBOR_FN = Callable[[List[Doc]], List[List[NEIGHBORS]]]
NOISE_FN = Callable[[Union[str, SAMPLE]], Tuple[str, str]]


def prepare_samples(
        sequences: List[str],
        infos: List[Dict[str, Any]],
        tokenization_fn: TOKENIZATION_FN,
        neighbor_fn: Optional[NEIGHBOR_FN] = None,
        **tokenize_words_kwargs: Any,
) -> List[SAMPLE]:
    outputs = tokenize_words_batch(sequences, return_docs=True, **tokenize_words_kwargs)
    docs: List[Doc] = [doc for _, doc in outputs]
    if neighbor_fn is not None:
        neighbors_lists = neighbor_fn(docs)
    else:
        neighbors_lists = [None] * len(docs)
    samples = []
    for doc, neighbors_list, info in zip(docs, neighbors_lists, infos):
        tokens = tokenization_fn(doc)
        sample = SAMPLE(
            tokens=tokens,
            doc=doc,
            neighbors_list=None if neighbors_list is None else neighbors_list,
            info=info
        )
        samples.append(sample)
    return samples


def serialize_samples(
        samples: List[SAMPLE]
) -> List[bytes]:
    import pickle
    import lz4.frame
    from spacy.tokens import DocBin

    outputs = []
    for sample in samples:
        doc_bin = DocBin()
        doc_bin.add(sample.doc)
        outputs.append(
            lz4.frame.compress(
                pickle.dumps((sample.tokens, doc_bin.to_bytes(), sample.neighbors_list, sample.info)),
                compression_level=lz4.frame.COMPRESSIONLEVEL_MAX)
        )
    return outputs


def deserialize_samples(inputs: List[bytes]) -> List[SAMPLE]:
    import pickle
    import lz4.frame
    from spacy.tokens import DocBin

    outputs = []
    for ipt in inputs:
        tokens, doc_bin_bytes, neighbors_list, info = pickle.loads(lz4.frame.decompress(ipt))
        doc_bin = DocBin().from_bytes(doc_bin_bytes)
        docs = list(doc_bin.get_docs(SPACY_TOKENIZER_REGEX.vocab))
        assert len(docs) == 1
        outputs.append(SAMPLE(tokens=tokens, doc=docs[0], neighbors_list=neighbors_list, info=info))
    return outputs


def sanitize_sample(sample: SAMPLE, unk_token_id: int) -> SAMPLE:
    for i, tokens in enumerate(sample.tokens):
        if len(tokens) == 0:
            sample.tokens[i] = [unk_token_id]
    return sample


def de_tokenize_words(words: List[str], doc: Union[List[bool], Optional[Doc]] = None) -> str:
    if doc is None:
        return " ".join(words)
    else:
        if isinstance(doc, Doc):
            whitespaces = [token.whitespace_ == " " for token in doc]
        else:
            whitespaces = doc

        assert len(words) == len(whitespaces), f"Got {len(words)} words but {len(whitespaces)} whitespace infos"

        sequence = ""

        for word, has_trailing_ws in zip(words, whitespaces):
            sequence += word + " " * has_trailing_ws

        return sequence


def is_valid_word(word: str) -> bool:
    pattern = re.compile(r"^[^\W\d_]+$")
    return pattern.fullmatch(word) is not None


def get_word_frequencies_from_file(file: str) -> Counter:
    dictionary = Counter()
    with open(file, "r", encoding="utf8") as f:
        lines = f.readlines()
        lines = [clean_sequence(line) for line in lines]
        lines = [line for line in lines if line != ""]

    if file.endswith(".jsonl"):
        lines = [json.loads(line)["sequence"] for line in lines]
    elif file.endswith(".txt"):
        pass
    else:
        raise ValueError(f"Unknown file ending of in file, must be either .jsonl or .txt")

    for line in lines:
        for w in tokenize_words_regex(line)[0]:
            dictionary[w] += 1

    return dictionary


def dictionary_token_flags(token: Token, dictionary: Dict[str, int]) -> List[bool]:
    return [token.text in dictionary, token.text.lower() in dictionary]


def special_token_flags(token: Token) -> List[bool]:
    return [token.is_punct,
            token.is_currency,
            token.is_digit or token.like_num,
            token.like_url,
            token.like_email]


def additional_token_flags(token: Token) -> List[bool]:
    return [token.whitespace_ == " ",
            token.is_title,
            token.is_upper,
            token.is_lower,
            token.is_stop,
            token.is_alpha]


def token_flags(token: Token, dictionary: Optional[Dict[str, Any]]) -> List[bool]:
    features = special_token_flags(token) + additional_token_flags(token)
    if dictionary is not None:
        features += dictionary_token_flags(token, dictionary)
    return features


def parser_token_flags(token: Token) -> List[bool]:
    return [token.is_sent_start or False,
            token.is_sent_end or False]


def is_special_token(token: Token) -> bool:
    return any(special_token_flags(token))


def one_hot_encode(idx: int, num_items: int) -> List[bool]:
    one_hot = [False] * num_items
    if 0 <= idx < num_items:
        one_hot[idx] = True
    return one_hot


def flatten(inputs: Union[List, Any]) -> List:
    if not isinstance(inputs, list):
        return [inputs]
    return [elem for ipt in inputs for elem in flatten(ipt)]


def open_lmdb(lmdb_path: str, write: bool = False) -> lmdb.Environment:
    return lmdb.open(
        lmdb_path,
        map_size=5e10,
        subdir=False,
        readonly=not write,
        lock=False
    )


def collate(items: List[Tuple[DATA_INPUT, Dict[str, Any]]]) -> BATCH:
    data = []
    info = {}
    for item in items:
        data.append(item[0])
        for key, val in item[1].items():
            if key not in info:
                info[key] = [val]
            else:
                info[key].append(val)
    if len(data) > 0 and isinstance(data[0], dgl.DGLHeteroGraph):
        data = dgl.batch(data)
    return BATCH(data, info)


# modified version of
# https://catalyst-team.github.io/catalyst/_modules/catalyst/data/dataset/torch.html#DatasetFromSampler
class SamplerDataset(Dataset):
    def __init__(self, sampler: Sampler) -> None:
        super().__init__()
        self.sampler = sampler
        self.sampler_indices = None

    def __getitem__(self, idx: int) -> Any:
        if self.sampler_indices is None:
            self.sampler_indices = list(self.sampler)
        return self.sampler_indices[idx]

    def __len__(self) -> int:
        return len(self.sampler)


# modified version of
# https://catalyst-team.github.io/catalyst/_modules/catalyst/data/sampler.html#DistributedSamplerWrapper
class DistributedDynamicSampler(DistributedSampler):
    def __init__(self,
                 sampler: Sampler,
                 device: DistributedDevice,
                 seed: int,
                 drop_last: bool = False,
                 shuffle: bool = True) -> None:
        super().__init__(SamplerDataset(sampler),
                         device.world_size,
                         device.rank,
                         shuffle,
                         seed,
                         drop_last)
        self.sampler = sampler
        self.steps_to_fast_forward = 0

    def __iter__(self) -> List[int]:
        self.dataset = SamplerDataset(self.sampler)

        dist_indices = list(super().__iter__())
        sampler_indices = self.dataset

        for idx in dist_indices[self.steps_to_fast_forward:]:
            yield sampler_indices[idx]

    def set_epoch(self, epoch: int) -> None:
        super().set_epoch(epoch)
        if hasattr(self.sampler, "set_epoch"):
            self.sampler.set_epoch(epoch)

    def set_steps_to_fast_forward(self, steps: int) -> None:
        self.steps_to_fast_forward = steps

    def __len__(self) -> int:
        return super().__len__() - self.steps_to_fast_forward


class BucketSampler(Sampler):
    def __init__(self,
                 dataset: dgl.data.DGLDataset,
                 values: List[int],
                 batch_max_value: int,
                 seed: int,
                 shuffle: bool = False,
                 bucket_span: Optional[int] = None,
                 max_value: int = 512) -> None:
        super().__init__(None)
        self.logger = common.get_logger("BUCKET_SAMPLER")
        self.dataset = dataset
        self.values = values
        self.batch_max_value = batch_max_value
        self.seed = seed
        self.shuffle = shuffle
        self.bucket_span = bucket_span
        self.max_value = max_value
        self.batches = []
        self.rand = random.Random(seed)

        if bucket_span is None:
            self._build_batches()
        else:
            self._build_batch_buckets()

        if not dist.is_initialized() or (dist.is_initialized() and dist.get_rank() == 0):
            self.logger.info(
                f"Generated {len(self.batches)} batches with {sum(len(b) for b in self.batches)} items in total "
                f"(batch_max_value={self.batch_max_value}, bucket_span={self.bucket_span}, max_value={self.max_value})"
            )

    def _build_batches(self) -> None:
        indices = list(range(len(self.dataset)))
        if self.shuffle:
            self.rand.shuffle(indices)
        batches = []
        running_sum = 0
        batch = []
        for idx in indices:
            value = self.values[idx]
            if running_sum + value > self.batch_max_value and len(batch) > 0:
                batches.append(batch)
                batch = []
                running_sum = 0
            batch.append(idx)
            running_sum += value
        if len(batch) > 0:
            batches.append(batch)

        self.batches = batches

    def _build_batch_buckets(self) -> None:
        num_buckets = self.max_value // self.bucket_span + 1
        bucket_max_lengths = [min((bucket_idx + 1) * self.bucket_span - 1, self.max_value)
                              for bucket_idx in range(num_buckets)]
        bucket_max_batch_samples = [self.batch_max_value // max(1, bucket_max_lengths[bucket_idx])
                                    for bucket_idx in range(num_buckets)]
        batch_buckets = [[] for _ in range(num_buckets)]

        indices = list(range(len(self.dataset)))
        if self.shuffle:
            self.rand.shuffle(indices)

        for idx in indices:
            value = self.values[idx]
            bucket_idx = value // self.bucket_span

            if len(batch_buckets[bucket_idx]) == 0:
                batch_buckets[bucket_idx].append([])

            if len(batch_buckets[bucket_idx][-1]) < bucket_max_batch_samples[bucket_idx]:
                batch_buckets[bucket_idx][-1].append(idx)
            else:
                batch_buckets[bucket_idx].append([idx])

        batches = [batch for bucket in batch_buckets for batch in bucket if len(batch) > 0]
        if self.shuffle:
            self.rand.shuffle(batches)

        self.batches = batches

    def __iter__(self) -> Iterator:
        for b in self.batches:
            yield b

    def __len__(self) -> int:
        return len(self.batches)


def get_word_whitespace_groups(
        sample: SAMPLE
) -> torch.Tensor:
    word_whitespace_groups = []
    word_ws_group = 0
    for i, (tokens, word) in enumerate(zip(sample.tokens, sample.doc)):
        word_whitespace_groups.extend([word_ws_group] * len(tokens))
        if word.whitespace_ == " ":
            word_ws_group += 1
    return torch.tensor(word_whitespace_groups, dtype=torch.long)


def get_word_and_word_whitespace_groups(
        sample: SAMPLE
) -> Tuple[torch.Tensor, torch.Tensor]:
    word_groups = []
    word_whitespace_groups = []
    word_ws_group = 0
    for i, (tokens, word) in enumerate(zip(sample.tokens, sample.doc)):
        word_groups.extend([i] * len(tokens))
        word_whitespace_groups.append(word_ws_group)
        if word.whitespace_ == " ":
            word_ws_group += 1
    return torch.tensor(word_groups, dtype=torch.long), torch.tensor(word_whitespace_groups, dtype=torch.long)


def get_sequence_groups(
        sample: SAMPLE
) -> torch.Tensor:
    return torch.tensor([0] * len(flatten(sample.tokens)), dtype=torch.long)


def get_word_and_sequence_groups(
        sample: SAMPLE
) -> Tuple[torch.Tensor, torch.Tensor]:
    word_groups = []
    for i, tokens in enumerate(sample.tokens):
        word_groups.extend([i] * len(tokens))
    sequence_groups = [0] * len(sample.doc)
    return torch.tensor(word_groups, dtype=torch.long), torch.tensor(sequence_groups, dtype=torch.long)


def get_word_features(doc: Doc, dictionary: Optional[Dict[str, int]]) -> torch.Tensor:
    features = []
    for word in doc:
        features.append(token_flags(word, dictionary))
    return torch.tensor(features, dtype=torch.float)


def get_character_groups_from_repaired_doc(
        input_characters: List[str],
        repaired_doc: Doc
) -> torch.Tensor:
    character_groups = []
    char_idx = 0
    for i, word in enumerate(repaired_doc):
        running_word = ""
        while running_word != word.text and len(running_word) < len(word.text):
            if input_characters[char_idx] == " ":
                character_groups.append(-1)
            else:
                running_word += input_characters[char_idx]
                character_groups.append(i)
            char_idx += 1
        assert running_word == word.text

    assert char_idx == len(input_characters)
    return torch.tensor(character_groups, dtype=torch.long)


def clean_sequence(sequence: str, fix_unicode_errors: bool = False) -> str:
    """

    Replace all multiple whitespaces, tabs,
    linebreaks etc. with single whitespaces.

    :param sequence: string
    :param fix_unicode_errors: bool
    :return: cleaned string
    """
    if fix_unicode_errors:
        sequence = fix_unicode(sequence)
    return " ".join(sequence.strip().split())


def fix_unicode(sequence: str) -> str:
    """

    Fixes quotes and unicode issues using ftfy.

    :param sequence: string
    :return: cleaned string
    """
    # sequence = "".join(ch for ch in sequence if unicodedata.category(ch)[0] != "C")
    return ftfy.fix_text(sequence)


def is_valid_sequence(sequence: str, min_length: int = 0, max_length: int = -1, min_words: int = 0) -> bool:
    """
    Check if a string is a valid sequence in the
    sense that it is a proper sentence/expression.

    :param sequence: string
    :param min_length: minimum length of string
    :param max_length: maximum length of string
    :param min_words: minimum number of words in string
    :return: bool whether string is valid
    """
    if max_length < 0:
        max_length = float("inf")
    # from tokenization repair repo
    f = re.compile(r" [.,;]( |$)|<|>|\"\"|\(\)| ' |\([,;]|colspan")
    if f.search(sequence) is not None:
        return False
    # if sequence is smaller than min_length characters its invalid
    if len(sequence) < min_length or len(sequence) > max_length or len(sequence.split()) < min_words:
        return False
    # sequence must contain at least one standard character to be valid
    contains_chars = re.compile(r"[a-zA-Z]+")
    if contains_chars.search(sequence) is None:
        return False
    # check if sequence contains some xml/html markup
    contains_markup = re.compile(r"<.*?>|</.*?>")
    if contains_markup.search(sequence) is not None:
        return False
    # if sequence passes all the tests its valid
    return True


_INCLUDE_ALL = tuple(i for i in range(4))
_EDIT_CHARS = tuple(string.ascii_letters)


def edit_token(token: str,
               rand: np.random.Generator,
               include: Tuple[int, ...] = _INCLUDE_ALL,
               edit_chars: Tuple[str, ...] = _EDIT_CHARS,
               exclude_indices: Optional[Set[int]] = None) -> \
        Tuple[str, List[int], Set[int]]:
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
    exclude_indices = set() if exclude_indices is None else exclude_indices
    if len(token) == 0:
        return token, [], set()

    # edit methods: 0 -> insert, 1 -> delete, 2 -> swap, 3 -> replace
    edit_method = rand.choice(include)
    edits = [-1] * len(token)

    if edit_method == 0:
        insert_indices = set(range(len(token) + 1)) - exclude_indices
        if len(insert_indices) > 0:
            char_idx = rand.integers(len(edit_chars))
            _insert_char = edit_chars[char_idx]
            assert len(_insert_char) > 0, "dont insert empty string, this is equal to leaving the token unchanged"

            token_idx = rand.choice(list(insert_indices))

            token = token[:token_idx] + _insert_char + token[token_idx:]

            edits.extend([-1] * len(_insert_char))
            for i in range(len(_insert_char)):
                edits[token_idx + i] = 0
                exclude_indices.add(token_idx + i)

    elif edit_method == 1 and len(token) > 1:
        delete_indices = set(range(len(token))) - exclude_indices
        if len(delete_indices) > 0:
            token_idx = rand.choice(list(delete_indices))
            token = token[:token_idx] + token[token_idx + 1:]

            edits.pop()
            edits[max(token_idx - 1, 0)] = 1
            edits[min(token_idx, len(token) - 1)] = 1

            # for delete we dont add anything to exclude indices

    elif edit_method == 2 and len(token) > 1:
        swap_indices = set()
        for i in range(len(token) - 1):
            if i in exclude_indices or i + 1 in exclude_indices:
                continue
            swap_indices.add(i)

        if len(swap_indices) > 0:
            token_idx = rand.choice(list(swap_indices))

            token = token[:token_idx] + token[token_idx + 1] + token[token_idx] + token[token_idx + 2:]

            edits[token_idx] = 2
            edits[token_idx + 1] = 2
            exclude_indices.add(token_idx)
            exclude_indices.add(token_idx + 1)

    else:
        replace_indices = set(range(len(token))) - exclude_indices
        if len(replace_indices) > 0:
            token_idx = rand.choice(list(replace_indices))

            new_char = token[token_idx]
            while new_char == token[token_idx]:
                new_char = edit_chars[rand.integers(len(edit_chars))]
            assert len(new_char) > 0, "dont replace chars with empty strings, delete should be used for that"

            token = token[:token_idx] + new_char + token[token_idx + 1:]

            edits.extend([-1] * (len(new_char) - 1))
            for i in range(len(new_char)):
                edits[token_idx + i] = 3
                exclude_indices.add(token_idx + i)

    assert len(token) == len(edits)
    return token, edits, exclude_indices


def find_substring_ignoring_spaces(
        substring: str,
        search_str: str
) -> Tuple[int, int]:
    pattern = r"\s*".join(re.escape(char) for char in substring.replace(" ", ""))
    match = re.search(pattern, search_str)
    assert match is not None
    return match.start(), match.end()
