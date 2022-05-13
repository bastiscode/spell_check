import enum
import multiprocessing as mp
import os
import pickle
import string
from collections import Counter
from dataclasses import dataclass
from typing import List, Optional, Any, Iterator, Callable, Union, Dict

import omegaconf
import tokenizers
from omegaconf import MISSING
from spacy.tokens import Doc
from tokenizers import trainers, pre_tokenizers, models, normalizers, decoders
from tqdm import tqdm

from nsc.data import utils
from nsc.utils import io, common

os.environ["TOKENIZERS_PARALLELISM"] = "false"

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
    def __init__(self, unk_id: int, bos_id: int, eos_id: int) -> None:
        self.bos_id = bos_id
        self.eos_id = eos_id
        self.unk_id = unk_id

    @property
    def vocab_size(self) -> int:
        raise NotImplementedError

    def normalize(self, sequence: str) -> str:
        return self.normalize_batch([sequence])[0]

    def normalize_batch(self, sequences: List[str]) -> List[str]:
        return sequences

    def split(self, sequence: str) -> List[str]:
        return self.split_batch([sequence])[0]

    def split_batch(self, sequences: List[str]) -> List[List[str]]:
        raise NotImplementedError

    def token_to_id(self, token: str) -> int:
        raise NotImplementedError

    def id_to_token(self, token_id: int) -> str:
        raise NotImplementedError

    def tokenize(self, sequence: str, add_bos_eos: bool = False) -> List[int]:
        return self.tokenize_batch([sequence], add_bos_eos)[0]

    def tokenize_batch(self, sequences: List[str], add_bos_eos: bool = False) -> List[List[int]]:
        batch_token_ids = []
        for tokens in self.split_batch(sequences):
            token_ids = [self.token_to_id(t) for t in tokens]
            if add_bos_eos:
                token_ids = [self.bos_id] + token_ids + [self.eos_id]
            batch_token_ids.append(token_ids)
        return batch_token_ids

    def de_tokenize(self, token_ids: List[int], **kwargs: Any) -> str:
        return self.de_tokenize_batch([token_ids], **kwargs)[0]

    def de_tokenize_batch(self, token_ids: List[List[int]], **kwargs: Any) -> List[str]:
        raise NotImplementedError

    @property
    def name(self) -> str:
        raise NotImplementedError


def get_tokenizer_from_config(cfg: Union[TokenizerConfig, omegaconf.DictConfig]) -> Tokenizer:
    # explicitly convert to dict config first, this way we support both dictconfigs
    # and structured configs as input
    cfg: omegaconf.DictConfig = omegaconf.DictConfig(cfg)
    cfg: TokenizerConfig = omegaconf.OmegaConf.structured(TokenizerConfig(**cfg))
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
        self.vocab = self._generate_byte_vocab()
        self.reverse_vocab = {
            v: k for k, v in self.vocab.items()
        }
        super().__init__(self.vocab[UNK], self.vocab[BOS], self.vocab[EOS])

    def _generate_byte_vocab(self) -> Dict[str, int]:
        vocab = {chr(i): i for i in range(256)}
        special_vocab = {
            st: 256 + i for i, st in
            enumerate(SPECIAL_TOKENS)
        }  # unk should be never needed with bytes, but we still add it because some functions using tokenizers
        # expect every tokenizer to have an unk token
        # put special tokens at end of vocab such that byte == token_id
        return {**vocab, **special_vocab}

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def split_batch(self, sequences: List[str]) -> List[List[str]]:
        return [list(chr(b) for b in s.encode("utf8")) for s in self.normalize_batch(sequences)]

    def token_to_id(self, token: str) -> int:
        return self.vocab[token]

    def id_to_token(self, token_id: int) -> str:
        return self.reverse_vocab[token_id]

    def de_tokenize_batch(self, token_ids: List[List[int]], **kwargs: Any) -> List[str]:
        strings = []
        for ids in token_ids:
            strings.append(bytes(filter(lambda token_id: token_id < 256, ids)).decode("utf8"))
        return strings

    @property
    def name(self) -> str:
        return "ByteTokenizer"


class CharTokenizer(Tokenizer):
    def __init__(self) -> None:
        self.vocab = self._generate_char_vocab()
        self.reverse_vocab = {
            v: k for k, v in self.vocab.items()
        }
        super().__init__(self.vocab[UNK], self.vocab[BOS], self.vocab[EOS])

    def _generate_char_vocab(self) -> Dict[str, int]:
        special_vocab = {st: i for i, st in enumerate(SPECIAL_TOKENS)}
        vocab = {char: i + len(SPECIAL_TOKENS) for i, char in enumerate(ALL_CHARS)}
        return {**special_vocab, **vocab}

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def split_batch(self, sequences: List[str]) -> List[List[str]]:
        return [list(s) for s in self.normalize_batch(sequences)]

    def token_to_id(self, token: str) -> int:
        return self.vocab.get(token, self.unk_id)

    def id_to_token(self, token_id: int) -> str:
        return self.reverse_vocab.get(token_id, UNK)

    def de_tokenize_batch(self, token_ids: List[List[int]], **kwargs: Any) -> List[str]:
        strings = []
        for ids in token_ids:
            strings.append("".join(self.id_to_token(token_id) for token_id in ids))
        return strings

    @property
    def name(self) -> str:
        return "CharTokenizer"


class TokenizationRepairTokenizer(CharTokenizer):
    def _generate_char_vocab(self) -> Dict[str, int]:
        vocab = {"#": 0, "_": 1, "x": 2}
        special_vocab = {st: i + len(vocab) for i, st in enumerate(SPECIAL_TOKENS)}
        return {**vocab, **special_vocab}

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
        super().__init__(self.vocab[UNK], self.vocab[BOS], self.vocab[EOS])

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
        with ctx.Pool(min(int(os.getenv("NSC_NUM_PROCESSES", len(os.sched_getaffinity(0)))), len(files))) as pool:
            for d in tqdm(
                    pool.imap_unordered(utils.get_word_frequencies_from_file, files),
                    total=len(files),
                    desc="calculating word frequencies from files",
                    leave=False,
                    disable=common.disable_tqdm()
            ):
                word_freq += d

        token_id = len(vocab)
        for word, _ in word_freq.most_common(vocab_size - len(vocab)):
            vocab[word] = token_id
            token_id += 1

        os.makedirs(os.path.dirname(save_path),
                    exist_ok=True)
        with open(save_path, "wb") as f:  # type: ignore
            pickle.dump(vocab, f)  # type: ignore

    def split_batch(self, sequences: List[str]) -> List[List[str]]:
        return [utils.tokenize_words_regex(s)[0] for s in self.normalize_batch(sequences)]

    def token_to_id(self, token: str) -> int:
        return self.vocab.get(token, self.unk_id)

    def id_to_token(self, token_id: int) -> str:
        return self.reverse_vocab.get(token_id, UNK)

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def de_tokenize(self, token_ids: List[int], **kwargs: Any) -> str:
        batch_whitespaces = [kwargs.get("whitespaces", kwargs.get("doc"))]
        return super().de_tokenize(token_ids, whitespaces=batch_whitespaces)

    def de_tokenize_batch(self, token_ids: List[List[int]], **kwargs: Any) -> List[str]:
        strings = []
        batch_whitespaces = kwargs.get("whitespaces", kwargs.get("doc", [None for _ in range(len(token_ids))]))
        for ids, ws in zip(token_ids, batch_whitespaces):
            strings.append(utils.de_tokenize_words(
                [self.id_to_token(token_id) for token_id in ids],
                ws
            ))
        return strings

    @property
    def name(self) -> str:
        return "WordTokenizer"


class BPETokenizer(Tokenizer):
    def __init__(self, cfg: TokenizerConfig) -> None:
        self.file_path = cfg.file_path
        self.tokenizer = tokenizers.Tokenizer.from_file(cfg.file_path)
        super().__init__(self.token_to_id(UNK), self.token_to_id(BOS), self.token_to_id(EOS))

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

    def normalize_batch(self, sequences: List[str]) -> List[str]:
        return [self.tokenizer.normalizer.normalize_str(s) for s in sequences]

    def split_batch(self, sequences: List[str]) -> List[List[str]]:
        return [enc.tokens for enc in self.tokenizer.encode_batch(sequences)]

    def token_to_id(self, token: str) -> int:
        token_id = self.tokenizer.token_to_id(token)
        return self.unk_id if token_id is None else token_id

    def id_to_token(self, token_id: int) -> str:
        token = self.tokenizer.id_to_token(token_id)
        return UNK if token is None else token

    def tokenize_batch(self, sequences: List[str], add_bos_eos: bool = False) -> List[List[int]]:
        batch_token_ids = []
        for enc in self.tokenizer.encode_batch(sequences):
            token_ids = enc.ids
            if add_bos_eos:
                token_ids = [self.bos_id] + token_ids + [self.eos_id]
            batch_token_ids.append(token_ids)
        return batch_token_ids

    def de_tokenize_batch(self, token_ids: List[List[int]], **kwargs: Any) -> List[str]:
        batch_decoded = self.tokenizer.decode_batch(
            token_ids,
            skip_special_tokens=False
        )
        if self.tokenizer.pre_tokenizer.add_prefix_space:
            # remove prefix space added by tokenizer
            return [s.lstrip() for s in batch_decoded]
        else:
            return batch_decoded

    @property
    def name(self) -> str:
        return "BPETokenizer"
