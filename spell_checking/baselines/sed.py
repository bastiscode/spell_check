import os
from typing import List, Optional, Any, Dict

from spacy.tokens import Token

from nsc.data import utils
from nsc.utils import io
from spell_checking import DICTIONARIES_DIR
from spell_checking.baselines import Baselines, Baseline


class SEDSequenceOODBaseline(Baseline):
    def __init__(self, seed: Optional[int] = None):
        super().__init__(seed)
        self.dictionary = io.dictionary_from_file(os.path.join(DICTIONARIES_DIR, "merged_train_100k.txt"))

    @property
    def name(self) -> str:
        return "ood"

    def _word_in_dict(self, word: Token) -> bool:
        return word.text.lower() in self.dictionary or word.text in self.dictionary

    def inference(self, sequences: List[str], **kwargs: Dict[str, Any]) -> List[int]:
        predictions = []
        for sequence in sequences:
            prediction = 0
            _, doc = utils.tokenize_words(sequence, return_doc=True)
            for word in doc:
                if not utils.is_special_token(word) and not self._word_in_dict(word):
                    prediction = 1
                    break
            predictions.append(prediction)
        return predictions


class SEDSequenceFromSECBaseline(Baseline):
    def __init__(self, baseline: Baselines, seed: Optional[int]):
        super().__init__(seed)
        assert baseline.name.startswith("SEC"), f"baseline must be a sec baseline, but got {baseline.name}"
        from spell_checking.baselines import get_baseline
        self.sec = get_baseline(baseline, seed, **{"languagetool_edit_whitespaces": False})

    @property
    def name(self) -> str:
        return self.sec.name

    def inference(self, sequences: List[str], **kwargs: Dict[str, Any]) -> List[int]:
        predictions = self.sec.inference(sequences)
        return [int(p != s) for p, s in zip(predictions, sequences)]


class SEDWordsOODBaseline(SEDSequenceOODBaseline):
    def inference(self, sequences: List[str], **kwargs: Dict[str, Any]) -> List[List[int]]:
        predictions = []
        for sequence in sequences:
            prediction = []
            _, doc = utils.tokenize_words(sequence, return_doc=True)
            for word in doc:
                if not utils.is_special_token(word) and not self._word_in_dict(word):
                    prediction.append(1)
                else:
                    prediction.append(0)

            # complete full predictions
            merged_predictions = []
            merged_prediction = 0
            for i in range(len(doc)):
                merged_prediction += prediction[i]
                if doc[i].whitespace_ == " " or i == len(doc) - 1:
                    merged_predictions.append(int(merged_prediction > 0))
                    merged_prediction = 0
            predictions.append(merged_predictions)
        return predictions


class SEDWordsFromSECBaseline(SEDSequenceFromSECBaseline):
    def inference(self, sequences: List[str], **kwargs: Dict[str, Any]) -> List[List[int]]:
        corrections = self.sec.inference(sequences)
        predictions = []
        for c, s in zip(corrections, sequences):
            c_ = c.split()
            s_ = s.split()
            assert len(c_) == len(s_), f"\n{c_}\n{s_}"
            predictions.append([int(c != s) for c, s in zip(c_, s_)])
        return predictions
