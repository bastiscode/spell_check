from typing import Tuple

from spacy.lang.en.tokenizer_exceptions import TOKENIZER_EXCEPTIONS

from gnn_lib.data import utils


def clean_sequences(correct_sequence: str, corrupt_sequence: str) -> Tuple[str, str]:
    words = []
    whitespaces = []
    _words, doc = utils.tokenize_words(correct_sequence, return_doc=True, split_only_on_ws=True)
    assert _words == correct_sequence.split(), f"{_words} <--> {correct_sequence.split()}"
    assert len(correct_sequence.split()) == len(corrupt_sequence.split())

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

        words.append(token.text)
        whitespaces.append(add_with_ws)

    new_correct_sequence = utils.de_tokenize_words(words, whitespaces)
    new_corrupt_sequence = utils.de_tokenize_words(corrupt_sequence.split(), whitespaces)
    return new_correct_sequence, new_corrupt_sequence
