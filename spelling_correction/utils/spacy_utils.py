import argparse
from typing import Tuple, List

from tqdm import tqdm
import spacy
from spacy.lang.en.tokenizer_exceptions import TOKENIZER_EXCEPTIONS
from spacy.tokens import Doc

from gnn_lib.api.utils import load_text_file, save_text_file
from gnn_lib.data import utils


class SpacyWhitespaceTokenizer:
    def __init__(self, vocab: spacy.Vocab) -> None:
        self.vocab = vocab

    def __call__(self, text: str) -> Doc:
        words = text.split()
        whitespaces = [True] * len(words)
        whitespaces[-1] = False
        return Doc(self.vocab, words=words, spaces=whitespaces)


SPACY_TOKENIZER = spacy.load("en_core_web_lg")

SPACY_TOKENIZER_WS = spacy.load("en_core_web_lg")
SPACY_TOKENIZER_WS.tokenizer = SpacyWhitespaceTokenizer(SPACY_TOKENIZER_WS.vocab)


def tokenize_words_ws(sequence: str) -> Tuple[List[str], Doc]:
    doc = SPACY_TOKENIZER_WS(sequence)
    return [w.text for w in doc], doc


def fix_sequence(sequence: str) -> Tuple[str, List[bool]]:
    whitespaces = []
    words, doc = tokenize_words_ws(sequence)
    num_quotes = 0
    for i, token in enumerate(doc):
        next_token = doc[i + 1] if i < len(doc) - 1 else None

        add_with_ws = True

        if next_token is not None:
            if token.is_quote:
                num_quotes += 1
                if num_quotes % 2 == 1:
                    add_with_ws = False

            if next_token.is_quote and num_quotes % 2 == 1:
                add_with_ws = False

            if token.text + next_token.text in TOKENIZER_EXCEPTIONS:
                add_with_ws = False

            if (token.is_left_punct and not token.is_quote) or token.text == "-":
                add_with_ws = False

            if (next_token.is_right_punct and not next_token.is_quote) \
                    or (next_token.is_punct and not next_token.is_right_punct and not next_token.is_left_punct):
                add_with_ws = False

            if next_token.text == "'s":
                add_with_ws = False

        else:
            add_with_ws = False

        whitespaces.append(add_with_ws)

    return utils.de_tokenize_words(words, whitespaces), whitespaces


def fix_sequences(correct_sequence: str, corrupt_sequence: str) -> Tuple[str, str]:
    assert len(correct_sequence.split()) == len(corrupt_sequence.split())
    correct_sequence, whitespaces = fix_sequence(correct_sequence)
    corrupt_sequence = utils.de_tokenize_words(corrupt_sequence.split(), whitespaces)
    return correct_sequence, corrupt_sequence


def split_sequence(sequence: str) -> str:
    doc = SPACY_TOKENIZER(sequence)
    return " ".join(token.text for token in doc)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-file", type=str, required=True)
    parser.add_argument("--out-file", type=str, required=True)
    parser.add_argument("--action", choices=["fix", "split"], required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    inputs = load_text_file(args.in_file)
    if args.action == "fix":
        outputs = [fix_sequence(line)[0] for line in tqdm(inputs)]
    else:
        outputs = [split_sequence(line) for line in tqdm(inputs)]
    save_text_file(args.out_file, outputs)
