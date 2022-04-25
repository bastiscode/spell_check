import collections
import json
import os
import re
from typing import List, Optional, Dict, Iterable, Any, Tuple

import numpy as np
import pkg_resources
import requests
import torch
from spacy.tokens import Doc

from nsc.data import utils, index
from nsc.utils import io

from spell_checking import DICTIONARIES_DIR, SPELL_CHECK_INDEX_DIR
from spell_checking.baselines import Baseline
from spell_checking.utils.edit import edit_distance


class SECCTDBaseline(Baseline):
    def __init__(self, seed: int):
        super().__init__(seed)
        self.dictionary = io.dictionary_from_file(os.path.join(DICTIONARIES_DIR, "merged_train_100k.txt"))
        self.rand = np.random.RandomState(seed)

    @property
    def name(self) -> str:
        return "ctd"

    def _word_in_dict(self, word: str) -> bool:
        return word.lower() in self.dictionary or word in self.dictionary

    def _closest_words_in_dict(self, word: str) -> List[str]:
        min_ed = float("inf")

        closest_words: Dict[str, int] = {}
        for w, freq in self.dictionary.items():
            edit_dist = edit_distance(w, word)
            if edit_dist < min_ed:
                closest_words = {w: freq}
                min_ed = edit_dist
            elif edit_dist == min_ed:
                closest_words[w] = freq

        max_freq = max(closest_words.values())
        return [w for w, freq in closest_words.items() if freq == max_freq]

    def inference(self, sequences: List[str], **kwargs: Dict[str, Any]) -> List[str]:
        predictions = []
        for seq in sequences:
            predicted_words = []
            words, doc = utils.tokenize_words(seq, return_doc=True)
            for word in doc:
                if utils.is_special_token(word) or self._word_in_dict(word.text):
                    predicted_words.append(word.text)
                else:
                    closest_words = self._closest_words_in_dict(word.text)
                    cw_idx = np.random.randint(len(closest_words))
                    predicted_words.append(closest_words[cw_idx])
            predictions.append(utils.de_tokenize_words(predicted_words, doc))
        return predictions


class SECSpellCheckIndexBaseline(Baseline):
    def __init__(
            self,
            seed: Optional[int] = None,
            index_name: str = "ctx_1_euclidean_custom",
            num_neighbors: int = 10
    ) -> None:
        super().__init__(seed)
        self.index_dir = os.path.join(
            SPELL_CHECK_INDEX_DIR, os.environ.get("BASELINE_SPELL_CHECK_INDEX", index_name)
        )
        self.index = index.NNIndex(self.index_dir)
        self.num_neighbors = num_neighbors

    @property
    def name(self) -> str:
        return f"sci_{os.path.basename(self.index_dir)}"

    def inference(self, sequences: List[str], **kwargs: Dict[str, Any]) -> List[str]:
        batch: List[Tuple[List[str], Doc]] = utils.tokenize_words_batch(sequences, return_docs=True)
        docs = [b[1] for b in batch]
        neighbors_lists = self.index.batch_retrieve_from_docs(docs, self.num_neighbors)
        predictions = []
        for doc, neighbor_list in zip(docs, neighbors_lists):
            assert len(doc) == len(neighbor_list)
            predicted_words = []
            for word, neighbors in zip(doc, neighbor_list):
                if utils.is_special_token(word):
                    predicted_words.append(word.text)
                else:
                    joined_neighbor_words = collections.defaultdict(int)

                    for word_list in neighbors.word_lists:
                        for w, freq in word_list:
                            joined_neighbor_words[w] += freq

                    neighbor_words = [
                        (w, freq, edit_distance(w, word.text))
                        for w, freq in joined_neighbor_words.items()
                    ]
                    neighbor_words = sorted(neighbor_words, key=lambda item: (item[2], -item[1]))
                    if len(neighbor_words):
                        predicted_words.append(neighbor_words[0][0])
                    else:
                        predicted_words.append(word.text)

            predictions.append(utils.de_tokenize_words(predicted_words, doc))
        return predictions


class SECDummyBaseline(Baseline):
    @property
    def name(self) -> str:
        return "dummy"

    def inference(self, sequences: List[str], **kwargs: Dict[str, Any]) -> List[str]:
        return sequences


class SECJamspellBaseline(Baseline):
    def __init__(self, seed: Optional[int] = None):
        super().__init__(seed)
        import jamspell
        self.spell_checker = jamspell.TSpellCorrector()
        self.spell_checker.LoadLangModel(os.path.join(os.path.dirname(__file__), "jamspell", "en.bin"))

    @property
    def name(self) -> str:
        return "jamspell"

    def inference(self, sequences: List[str], **kwargs: Dict[str, Any]) -> List[str]:
        return [self.spell_checker.FixFragment(sequence) for sequence in sequences]


class SECHunspellBaseline(Baseline):
    def __init__(self, seed: Optional[int] = None):
        super().__init__(seed)
        import hunspell
        self.spell_checker = hunspell.HunSpell("/usr/share/hunspell/en_US.dic", "/usr/share/hunspell/en_US.aff")

    @property
    def name(self) -> str:
        return "hunspell"

    def _correct_sequence(self, sequence: str) -> str:
        words, doc = utils.tokenize_words(sequence, return_doc=True)
        corrections = []
        for word in doc:
            if utils.is_special_token(word):
                corrections.append(word.text)
                continue

            if not self.spell_checker.spell(word.text):
                suggestions = self.spell_checker.suggest(word.text)
                if len(suggestions) > 0 and " " not in suggestions[0] and suggestions[0] != "":
                    corrections.append(suggestions[0])
                    continue

            corrections.append(word.text)
        return utils.de_tokenize_words(corrections, doc)

    def inference(self, sequences: List[str], **kwargs: Dict[str, Any]) -> List[str]:
        return [self._correct_sequence(sequence) for sequence in sequences]


class SECAspellBaseline(Baseline):
    def __init__(self, seed: Optional[int] = None):
        super().__init__(seed)
        import aspell
        self.spell_checker = aspell.Speller("lang", "en")
        self.spell_checker.setConfigKey("sug-mode", "normal")

    @property
    def name(self) -> str:
        return "aspell"

    def _correct_sequence(self, sequence: str) -> str:
        words, doc = utils.tokenize_words(sequence, return_doc=True)
        corrections = []
        for word in doc:
            if utils.is_special_token(word):
                corrections.append(word.text)
                continue

            if not self.spell_checker.check(word.text):
                suggestions = self.spell_checker.suggest(word.text)
                if len(suggestions) > 0 and " " not in suggestions[0] and suggestions[0] != "":
                    corrections.append(suggestions[0])
                    continue

            corrections.append(word.text)
        return utils.de_tokenize_words(corrections, doc)

    def inference(self, sequences: List[str], **kwargs: Dict[str, Any]) -> List[str]:
        return [self._correct_sequence(sequence) for sequence in sequences]


class SECNeuspellBaseline(Baseline):
    def __init__(self, model_name: str, seed: Optional[int] = None):
        super().__init__(seed)
        import neuspell
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if model_name == "bert":
            self.spell_checker = neuspell.BertChecker(
                pretrained=True,
                device=device
            )
        elif model_name == "sclstm_elmo":
            self.spell_checker = neuspell.SclstmelmoChecker(
                pretrained=True,
                device=device
            )
        else:
            raise ValueError(f"Unknown Neuspell spell checker {model_name}")

        self.model_name = model_name

        # bert checker just cuts off sequences longer than 510 tokens so have to do the splitting ourselves
        self.max_num_words = 128
        self.word_context = self.max_num_words // 8
        self.word_window = self.max_num_words - 2 * self.word_context

    @property
    def name(self) -> str:
        return f"neuspell_{self.model_name}"

    @staticmethod
    def match_tokens_backtracking(a: List[str], b: List[str]) -> List[bool]:
        assert len(a) <= len(b)
        a_str = " ".join(a)
        b_esc = [re.escape(tok) if tok != "[UNK]" else "\\S+?" for tok in b]

        whitespaces: List[Optional[bool]] = [None] * len(b)

        def prefix_match(idx: int) -> bool:
            b_str = "^"
            for tok, ws_ in zip(b_esc[:idx + 1], whitespaces[:idx + 1]):
                b_str += tok + " " * ws_
            is_match = re.match(b_str, a_str, re.UNICODE) is not None
            return is_match

        def match(idx: int) -> bool:
            for ws in [True, False]:
                whitespaces[idx] = ws
                if prefix_match(idx):
                    if idx >= len(b) - 1 or match(idx + 1):
                        return True
            if idx == 0:
                raise RuntimeError(f"should not happen:\n{a}\n{b}")
            return False

        match(0)

        assert all(w is not None for w in whitespaces), f"{whitespaces}\n{a}\n{b}"
        whitespaces[-1] = False
        return whitespaces

    def inference(self, sequences: List[str], **kwargs: Dict[str, Any]) -> List[str]:
        sequences = [utils.clean_sequence(seq, fix_unicode_errors=True) for seq in sequences]

        batch_detections = kwargs.get("detections", [[1] * len(seq.split()) for seq in sequences])

        input_sequences = []
        input_windows = []
        input_detections = []
        for sequence, detections in zip(sequences, batch_detections):
            words = sequence.split()
            if len(words) <= self.max_num_words:
                input_sequences.append(sequence)
                input_windows.append((0, len(words), 0))
                input_detections.append(detections)
            else:
                for window_idx, i in enumerate(range(0, len(words), self.word_window)):
                    input_sequences.append(
                        " ".join(words[max(0, i - self.word_context):i + self.word_window + self.word_context]))
                    start_idx = 0 if i == 0 else self.word_context
                    input_windows.append(
                        (start_idx, start_idx + self.word_window, window_idx)
                    )
                    input_detections.append(
                        detections[max(0, i - self.word_context):i + self.word_window + self.word_context]
                    )

        tokenized_sequences, corrected_sequences = self.spell_checker.correct_strings(input_sequences, return_all=True)

        outputs = []
        for sequence, tokenized, corrected, detections in zip(
                input_sequences, tokenized_sequences, corrected_sequences, input_detections
        ):
            sequence_tokens = sequence.split()
            assert len(detections) == len(sequence_tokens)
            tokenized_tokens = tokenized.split()
            corrected_tokens = corrected.split()
            assert len(sequence_tokens) <= len(tokenized_tokens) == len(corrected_tokens)
            assert re.fullmatch(
                "".join(re.escape(tok) if tok != "[UNK]" else "\\S+?" for tok in tokenized_tokens),
                "".join(sequence_tokens)
            ), (sequence_tokens, tokenized_tokens)

            whitespaces = SECNeuspellBaseline.match_tokens_backtracking(sequence_tokens, tokenized_tokens)

            output_str = ""
            ws_idx = 0
            for input_token, corrected_token, whitespace in zip(tokenized_tokens, corrected_tokens, whitespaces):
                output_str += (corrected_token if detections[ws_idx] else input_token) + " " * whitespace
                if whitespace:
                    ws_idx += 1
            assert ws_idx == len(sequence_tokens) - 1, (ws_idx, len(sequence_tokens))
            outputs.append(output_str)

        merged_outputs = []
        for output, input_sequence, (from_word, to_word, window_idx) in zip(outputs, input_sequences, input_windows):
            output_words = output.split()
            assert len(input_sequence.split()) == len(output_words)
            if window_idx == 0:
                merged_outputs.append(output_words[from_word:to_word])
            else:
                merged_outputs[-1].extend(output_words[from_word:to_word])

        assert [len(merged) == len(ipt.split()) for ipt, merged in zip(sequences, merged_outputs)]

        return [" ".join(output) for output in merged_outputs]


class SECNorvigBaseline(Baseline):
    @property
    def name(self) -> str:
        return "norvig"

    @staticmethod
    def _correct_sequence(sequence: str) -> str:
        from spell_checking.baselines.norvig import spell as norvig
        corrections = []
        words, doc = utils.tokenize_words(sequence, return_doc=True)
        for word in doc:
            if utils.is_special_token(word):
                corrections.append(word.text)
            else:
                corrections.append(norvig.correction(word.text))
        return utils.de_tokenize_words(corrections, doc)

    def inference(self, sequences: List[str], **kwargs: Dict[str, Any]) -> List[str]:
        return [self._correct_sequence(sequence) for sequence in sequences]


class SECSymSpellBaseline(Baseline):
    def __init__(self, seed: Optional[int] = None):
        super().__init__(seed)
        import symspellpy
        self.symspell = symspellpy.SymSpell(max_dictionary_edit_distance=2)
        dictionary_path = pkg_resources.resource_filename("symspellpy", "frequency_dictionary_en_82_765.txt")
        self.symspell.load_dictionary(dictionary_path, term_index=0, count_index=1)

    @property
    def name(self) -> str:
        return "symspell"

    def _correct_sequence(self, sequence: str) -> str:
        suggestions = self.symspell.lookup_compound(sequence, max_edit_distance=2)
        return suggestions[0].term

    def inference(self, sequences: List[str], **kwargs: Dict[str, Any]) -> List[str]:
        return [self._correct_sequence(sequence) for sequence in sequences]


class SECLanguagetoolBaseline(Baseline):
    def __init__(self, edit_whitespaces: bool = True, seed: Optional[int] = None):
        super().__init__(seed)
        port = int(os.getenv("LANGUAGE_TOOL_PORT", 8010))
        host = str(os.getenv("LANGUAGE_TOOL_HOST", "localhost"))
        self.base_url = f"http://{host}:{port}/v2"
        self.edit_whitespaces = edit_whitespaces
        try:
            # try if language tool server is reachable
            requests.get(f"{self.base_url}/languages")
        except Exception as e:
            raise RuntimeError(f"Error making request to language tool server at {self.base_url}:\n{e}")

    @property
    def name(self) -> str:
        return "languagetool"

    def _correct_with_language_tool(self, sequence: str) -> str:
        org_len = len(sequence)
        try:
            response = requests.post(f"{self.base_url}/check", data={"language": "en-US", "text": sequence})
            if response.status_code != 200:
                raise RuntimeError(f"Got status code {response.status_code} with response text '{response.text}'")
            response_data = json.loads(response.text)

            # keep track of the change in length of the sequence because the indices all refer to the original string
            change = 0
            for match in response_data["matches"]:
                if match["ignoreForIncompleteSentence"] or len(match["replacements"]) == 0:
                    continue
                replacement = match["replacements"][0]["value"]
                start_idx = match["offset"] + change
                end_idx = start_idx + match["length"]
                if (" " in replacement or " " in sequence[start_idx:end_idx]) and not self.edit_whitespaces:
                    continue
                sequence = sequence[:start_idx] + replacement + sequence[end_idx:]
                change = len(sequence) - org_len
        except Exception as e:
            raise RuntimeError(f"Error making request to language tool server at {self.base_url}:\n{e}")
        return sequence

    def inference(self, sequences: Iterable[str], **kwargs: Dict[str, Any]) -> List[str]:
        return [self._correct_with_language_tool(seq) for seq in sequences]
