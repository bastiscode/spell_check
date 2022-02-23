import enum
from typing import Any, Iterable, Optional


class Baselines(enum.IntEnum):
    # Spelling error detection sequence
    SED_SEQUENCE_OOD = 1
    SED_SEQUENCE_FROM_SEC = 2

    # Spelling error detection words
    SED_WORDS_OOD = 3
    SED_WORDS_FROM_SEC = 4

    # Spelling error correction
    SEC_CTD = 5
    SEC_DUMMY = 6
    SEC_JAMSPELL = 7
    SEC_NEUSPELL_BERT = 8
    SEC_NEUSPELL_ELMO = 9
    SEC_HUNSPELL = 10
    SEC_ASPELL = 11
    SEC_NORVIG = 12
    SEC_SYMSPELL = 13
    SEC_LANGUAGETOOL = 14


class Baseline:
    def __init__(self, seed: Optional[int] = None):
        self.seed = seed

    def inference(self, sequences: Iterable[str]) -> Any:
        raise NotImplementedError

    @property
    def name(self) -> str:
        raise NotImplementedError


def get_baseline(baseline: Baselines, seed: int, **kwargs: Any) -> Baseline:
    from spelling_correction.baselines.sed import (
        SEDSequenceOODBaseline,
        SEDWordsOODBaseline,
        SEDSequenceFromSECBaseline,
        SEDWordsFromSECBaseline
    )
    from spelling_correction.baselines.sec import (
        SECCTDBaseline,
        SECAspellBaseline,
        SECHunspellBaseline,
        SECJamspellBaseline,
        SECNeuspellBaseline,
        SECNorvigBaseline,
        SECDummyBaseline,
        SECLanguagetoolBaseline
    )

    if baseline == Baselines.SED_SEQUENCE_OOD:
        return SEDSequenceOODBaseline(seed)
    elif baseline == Baselines.SED_SEQUENCE_FROM_SEC:
        return SEDSequenceFromSECBaseline(kwargs["sec_baseline"], seed)
    elif baseline == Baselines.SED_WORDS_OOD:
        return SEDWordsOODBaseline(seed)
    elif baseline == Baselines.SED_WORDS_FROM_SEC:
        return SEDWordsFromSECBaseline(kwargs["sec_baseline"], seed)
    elif baseline == Baselines.SEC_CTD:
        return SECCTDBaseline(seed)
    elif baseline == Baselines.SEC_ASPELL:
        return SECAspellBaseline()
    elif baseline == Baselines.SEC_HUNSPELL:
        return SECHunspellBaseline()
    elif baseline == Baselines.SEC_NORVIG:
        return SECNorvigBaseline()
    elif baseline == Baselines.SEC_DUMMY:
        return SECDummyBaseline()
    elif baseline == Baselines.SEC_JAMSPELL:
        return SECJamspellBaseline()
    elif baseline == Baselines.SEC_NEUSPELL_BERT:
        return SECNeuspellBaseline(model_name="bert")
    elif baseline == Baselines.SEC_NEUSPELL_ELMO:
        return SECNeuspellBaseline(model_name="elmo_scrnn")
    elif baseline == Baselines.SEC_LANGUAGETOOL:
        return SECLanguagetoolBaseline(edit_whitespaces=False)
    else:
        raise ValueError(f"Unknown baseline {baseline.name}")

