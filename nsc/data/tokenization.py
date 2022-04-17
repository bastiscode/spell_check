import enum
import hashlib
import multiprocessing as mp
import os
import pickle
import string
from collections import Counter
from dataclasses import dataclass
from typing import List, Optional, Any, Iterator, Callable, Union

import omegaconf
import tokenizers
from omegaconf import MISSING
from spacy.tokens import Doc
from tokenizers import trainers, pre_tokenizers, models, normalizers, decoders
from tqdm import tqdm

from nsc.data import utils
from nsc.utils import io

ALL_CHARS = string.ascii_letters + string.digits + string.punctuation + " "

UNK = "<unk>"
BOS = "<bos>"
EOS = "<eos>"
PAD = "<pad>"
SPECIAL_TOKENS = [UNK, BOS, EOS, PAD]


class Tokenizers(enum.IntEnum):
    CHAR = 1
    WORD = 2
    BPE = 3
    TOKENIZATION_REPAIR = 4
    BYTE = 5


@dataclass
class TokenizerConfig:
    type: Tokenizers = MISSING
    file_path: Optional[str] = None


class Tokenizer:
    @property
    def vocab_size(self) -> int:
        raise NotImplementedError

    def normalize(self, sequence: str) -> str:
        raise NotImplementedError

    def split(self, sequence: str) -> List[str]:
        raise NotImplementedError

    def token_to_id(self, token: str) -> int:
        raise NotImplementedError

    def id_to_token(self, token_id: int) -> str:
        raise NotImplementedError

    def tokenize(self, sequence: str, add_bos_eos: bool = False) -> List[int]:
        raise NotImplementedError

    def de_tokenize(self, token_ids: List[int], **kwargs: Any) -> str:
        raise NotImplementedError

    @property
    def name(self) -> str:
        raise NotImplementedError


def get_tokenizer_from_config(cfg: Union[TokenizerConfig, omegaconf.DictConfig]) -> Tokenizer:
    cfg = omegaconf.OmegaConf.structured(
        cfg if isinstance(cfg, TokenizerConfig) else TokenizerConfig(**cfg)
    )
    if cfg.type == Tokenizers.CHAR:
        return CharTokenizer()
    elif cfg.type == Tokenizers.WORD:
        return WordTokenizer(cfg)
    elif cfg.type == Tokenizers.BPE:
        return BPETokenizer(cfg)
    elif cfg.type == Tokenizers.TOKENIZATION_REPAIR:
        return TokenizationRepairTokenizer()
    elif cfg.type == Tokenizers.BYTE:
        return ByteTokenizer()
    else:
        raise ValueError("Unknown tokenizer type")


def get_tokenization_fn(
        tokenizer: Tokenizer,
        respect_leading_whitespaces: bool = True
) -> Callable[[Doc], List[List[int]]]:
    def tok(doc: Doc) -> List[List[int]]:
        if respect_leading_whitespaces:
            # transform spacy trailing whitespaces saved in doc into leading whitespaces for our tokenizers such as BPE
            # also add leading and trailing whitespaces before/after first/last word if necessary
            word_tokens = [
                tokenizer.tokenize(
                    (doc[i - 1].whitespace_ if i > 0 else (doc.user_data.get("leading_whitespace", False) * " "))
                    + word.text
                    + (doc[i].whitespace_ if i == len(doc) - 1 else "")
                )
                for i, word in enumerate(doc)
            ]
        else:
            # just tokenize the raw words each on their own with all whitespaces removed
            word_tokens = [tokenizer.tokenize(word.text) for word in doc]
        return word_tokens

    return tok


class ByteTokenizer(Tokenizer):
    def __init__(self) -> None:
        self._generate_byte_vocab()
        self.reverse_vocab = {
            v: k for k, v in self.vocab.items()
        }
        self.bos_id = self.vocab[BOS]
        self.eos_id = self.vocab[EOS]

    def _generate_byte_vocab(self) -> None:
        vocab = {chr(i): i for i in range(256)}
        special_vocab = {
            st: 256 + i for i, st in
            enumerate(SPECIAL_TOKENS)
        }  # unk should be never needed with bytes, but we still add it because some functions using tokenizers
        # expect every tokenizer to have an unk token
        # put special tokens at end of vocab such that byte == token_id
        self.vocab = {**vocab, **special_vocab}

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def normalize(self, sequence: str) -> str:
        return sequence

    def split(self, sequence: str) -> List[str]:
        return list(chr(b) for b in self.normalize(sequence).encode("utf8"))

    def token_to_id(self, token: str) -> int:
        return self.vocab[token]

    def id_to_token(self, token_id: int) -> str:
        return self.reverse_vocab[token_id]

    def tokenize(self, sequence: str, add_bos_eos: bool = False) -> List[int]:
        token_ids = [self.token_to_id(c) for c in self.split(sequence)]
        if add_bos_eos:
            token_ids = [self.bos_id] + token_ids + [self.eos_id]
        return token_ids

    def de_tokenize(self, token_ids: List[int], **kwargs: Any) -> str:
        return bytes(filter(lambda token_id: token_id < 256, token_ids)).decode("utf8")

    @property
    def name(self) -> str:
        return "ByteTokenizer"


class CharTokenizer(Tokenizer):
    def __init__(self) -> None:
        self._generate_char_vocab()
        self.reverse_vocab = {
            v: k for k, v in self.vocab.items()
        }
        self.unk_id = self.vocab[UNK]
        self.bos_id = self.vocab[BOS]
        self.eos_id = self.vocab[EOS]

    def _generate_char_vocab(self) -> None:
        special_vocab = {st: i for i, st in enumerate(SPECIAL_TOKENS)}
        vocab = {char: i + len(SPECIAL_TOKENS) for i, char in enumerate(ALL_CHARS)}
        self.vocab = {**special_vocab, **vocab}

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def normalize(self, sequence: str) -> str:
        return sequence

    def split(self, sequence: str) -> List[str]:
        return list(self.normalize(sequence))

    def token_to_id(self, token: str) -> int:
        return self.vocab.get(token, self.unk_id)

    def id_to_token(self, token_id: int) -> str:
        return self.reverse_vocab.get(token_id, UNK)

    def tokenize(self, sequence: str, add_bos_eos: bool = False) -> List[int]:
        token_ids = [self.token_to_id(c) for c in self.split(sequence)]
        if add_bos_eos:
            token_ids = [self.bos_id] + token_ids + [self.eos_id]
        return token_ids

    def de_tokenize(self, token_ids: List[int], **kwargs: Any) -> str:
        return "".join(self.id_to_token(token_id)
                       for token_id in token_ids)

    @property
    def name(self) -> str:
        return "CharTokenizer"


class TokenizationRepairTokenizer(CharTokenizer):
    def _generate_char_vocab(self) -> None:
        vocab = {"#": 0, "_": 1, "x": 2}
        special_vocab = {st: i + len(vocab) for i, st in enumerate(SPECIAL_TOKENS)}
        self.vocab = {**vocab, **special_vocab}

    @property
    def name(self) -> str:
        return "TokenizationRepairTokenizer"


class WordTokenizer(Tokenizer):
    def __init__(self,
                 cfg: TokenizerConfig):
        with open(cfg.file_path, "rb") as f:
            self.vocab = pickle.load(f)

        self.file_path = cfg.file_path

        self.reverse_vocab = {
            v: k for k, v in self.vocab.items()
        }
        self.unk_id = self.vocab[UNK]
        self.bos_id = self.vocab[BOS]
        self.eos_id = self.vocab[EOS]

    @staticmethod
    def train(files: List[str], save_path: str, vocab_size: int, max_sequences: Optional[int] = None) -> None:
        vocab = {st: i for i, st in enumerate(SPECIAL_TOKENS)}

        ctx = mp.get_context("spawn")

        if max_sequences is not None and max_sequences > 0:
            running_sum = 0
            num_files = 0
            for file in files:
                running_sum += io.line_count(file)
                num_files += 1
                if running_sum >= max_sequences:
                    break
            files = files[:num_files]

        word_freq = Counter()
        with ctx.Pool(
                processes=min(int(os.getenv("NSC_NUM_PROCESSES", min(len(os.sched_getaffinity(0)), 8))), len(files))
        ) as pool:
            for i, d in tqdm(enumerate(pool.imap_unordered(utils.get_word_frequencies_from_file, files, chunksize=16)),
                             total=len(files),
                             desc="Calculating word frequencies from files"):
                word_freq += d

        token_id = len(vocab)
        for word, _ in word_freq.most_common(vocab_size - len(vocab)):
            vocab[word] = token_id
            token_id += 1

        os.makedirs(os.path.dirname(save_path),
                    exist_ok=True)
        with open(save_path, "wb") as f:  # type: ignore
            pickle.dump(vocab, f)  # type: ignore

    def normalize(self, sequence: str) -> str:
        return sequence

    def split(self, sequence: str) -> List[str]:
        return utils.tokenize_words_regex(self.normalize(sequence))[0]

    def token_to_id(self, token: str) -> int:
        return self.vocab.get(token, self.unk_id)

    def id_to_token(self, token_id: int) -> str:
        return self.reverse_vocab.get(token_id, UNK)

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def tokenize(self, sequence: str, add_bos_eos: bool = False) -> List[int]:
        token_ids = [self.token_to_id(w) for w in self.split(sequence)]
        if add_bos_eos:
            token_ids = [self.bos_id] + token_ids + [self.eos_id]
        return token_ids

    def de_tokenize(self, token_ids: List[int], **kwargs: Any) -> str:
        return utils.de_tokenize_words(
            [self.id_to_token(token_id) for token_id in token_ids],
            kwargs.get("whitespaces", kwargs.get("doc"))
        )

    @property
    def name(self) -> str:
        return "WordTokenizer"


class BPETokenizer(Tokenizer):
    def __init__(self, cfg: TokenizerConfig) -> None:
        self.file_path = cfg.file_path
        self.tokenizer = tokenizers.Tokenizer.from_file(cfg.file_path)
        self.bos_id = self.token_to_id(BOS)
        self.eos_id = self.token_to_id(EOS)
        self.unk_id = self.token_to_id(UNK)

    @staticmethod
    def train(
            files: List[str],
            save_path: str,
            vocab_size: int,
            max_sequences: Optional[int] = None,
            add_prefix_space: bool = False
    ) -> None:
        tokenizer = tokenizers.Tokenizer(models.BPE(unk_token=UNK))
        tokenizer.normalizer = normalizers.Sequence([
            normalizers.NFD(),
            normalizers.StripAccents()
        ])
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=add_prefix_space)
        tokenizer.decoder = decoders.ByteLevel()
        tokenizer.add_special_tokens(SPECIAL_TOKENS)
        initial_alphabet = pre_tokenizers.ByteLevel.alphabet()
        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=2,
            show_progress=True,
            special_tokens=SPECIAL_TOKENS,
            initial_alphabet=initial_alphabet
        )
        if len(initial_alphabet) < vocab_size:
            tokenizer.train_from_iterator(
                BPETokenizer._training_iterator(files, max_sequences),
                trainer
            )
        tokenizer.save(save_path)

    @staticmethod
    def _training_iterator(files: List[str], max_sequences: Optional[int]) -> Iterator[str]:
        num_sequences = 0
        for file in files:
            with open(file, "r", encoding="utf8") as in_f:
                for line in in_f:
                    yield line.strip()
                    num_sequences += 1

                    if max_sequences is not None and num_sequences >= max_sequences:
                        return

    @property
    def vocab_size(self) -> int:
        return self.tokenizer.get_vocab_size()

    def normalize(self, sequence: str) -> str:
        return self.tokenizer.normalizer.normalize_str(sequence)

    def split(self, sequence: str) -> List[str]:
        return self.tokenizer.encode(sequence).tokens

    def token_to_id(self, token: str) -> int:
        token_id = self.tokenizer.token_to_id(token)
        return self.unk_id if token_id is None else token_id

    def id_to_token(self, token_id: int) -> str:
        token = self.tokenizer.id_to_token(token_id)
        return UNK if token is None else token

    def tokenize(self, sequence: str, add_bos_eos: bool = False) -> List[int]:
        token_ids = self.tokenizer.encode(sequence).ids
        if add_bos_eos:
            token_ids = [self.bos_id] + token_ids + [self.eos_id]
        return token_ids

    def de_tokenize(self, token_ids: List[int], **kwargs: Any) -> str:
        de_tokenized = self.tokenizer.decode(
            token_ids,
            skip_special_tokens=False
        )
        if self.tokenizer.pre_tokenizer.add_prefix_space:
            return de_tokenized.lstrip()
        else:
            return de_tokenized

    @property
    def name(self) -> str:
        return "BPETokenizer"
