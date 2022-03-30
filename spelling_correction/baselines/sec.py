import json
import os
import re
from typing import List, Optional, Dict, Iterable, Any

import Levenshtein
import numpy as np
import pkg_resources
import requests
import torch

from gnn_lib.data import utils
from gnn_lib.utils import io

from spelling_correction import DICTIONARIES_DIR
from spelling_correction.baselines import Baseline


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
            edit_dist = Levenshtein.distance(w, word)
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

    @property
    def name(self) -> str:
        return f"neuspell_{self.model_name}"

    @staticmethod
    def match_tokens_backtracking(a: List[str], b: List[str]) -> List[bool]:
        assert len(a) <= len(b)
        a_str = " ".join(a)
        b_esc = [re.escape(tok) if tok != "[UNK]" else ".*" for tok in b]

        whitespaces: List[Optional[bool]] = [None] * len(b)

        def prefix_match(idx: int) -> bool:
            b_str = "^"
            for tok, ws_ in zip(b_esc[:idx + 1], whitespaces[:idx + 1]):
                b_str += tok + " " * ws_
            is_match = re.match(b_str, a_str) is not None
            return is_match

        def match(idx: int) -> int:
            for ws in [True, False]:
                whitespaces[idx] = ws
                if prefix_match(idx):
                    if idx >= len(b) - 1 or match(idx + 1):
                        return 1
            return 0

        match(0)

        for i in range(len(whitespaces)):
            if whitespaces[i] is None:
                whitespaces[i] = False

        assert all(w is not None for w in whitespaces), f"{whitespaces}\n{a}\n{b}"
        return whitespaces

    def inference(self, sequences: List[str], **kwargs: Dict[str, Any]) -> List[str]:
        sequences = [utils.clean_sequence(seq, fix_unicode_errors=True) for seq in sequences]
        tokenized_sequences, corrected_sequences = self.spell_checker.correct_strings(sequences, return_all=True)

        batch_detections = kwargs.get("detections", [[1] * len(seq.split()) for seq in sequences])

        outputs = []
        for sequence, tokenized, corrected, detections in zip(
                sequences, tokenized_sequences, corrected_sequences, batch_detections
        ):
            sequence_tokens = sequence.split()
            assert len(detections) == len(sequence_tokens)
            tokenized_tokens = tokenized.split()
            corrected_tokens = corrected.split()
            # assert len(sequence_tokens) <= len(tokenized_tokens) == len(corrected_tokens)
            # assert re.fullmatch(
            #     "".join(re.escape(tok) if tok != "[UNK]" else ".*" for tok in tokenized_tokens),
            #     "".join(sequence_tokens)
            # )

            whitespaces = SECNeuspellBaseline.match_tokens_backtracking(sequence_tokens, tokenized_tokens)

            output_str = ""
            ws_idx = 0
            for input_token, corrected_token, whitespace in zip(tokenized_tokens, corrected_tokens, whitespaces):
                output_str += (corrected_token if detections[ws_idx] else input_token) + " " * whitespace
                if whitespace:
                    ws_idx += 1
            outputs.append(output_str)

        return outputs


class SECNorvigBaseline(Baseline):
    @property
    def name(self) -> str:
        return "norvig"

    @staticmethod
    def _correct_sequence(sequence: str) -> str:
        from spelling_correction.baselines.norvig import spell as norvig
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
    def __init__(self, edit_whitespaces: bool = False, seed: Optional[int] = None):
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
