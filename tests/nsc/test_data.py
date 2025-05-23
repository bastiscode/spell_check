import string
from typing import Tuple

import numpy as np
import pytest

from nsc.data import utils
from nsc.data import preprocessing


class TestData:
    @staticmethod
    def randomly_insert_whitespaces(s: str, p: float, seed: int) -> str:
        rand = np.random.RandomState(seed)
        new_s = ""
        for i, c in enumerate(s):
            if rand.random() < p and c != " " and (s[i - 1] != " " if i > 0 else True):
                new_s += " " + c
            else:
                new_s += c
        return new_s

    @staticmethod
    def add_noise(s: str, p: float, seed: int) -> str:
        rand = np.random.RandomState(seed)
        noise_chars = ["\n", "\t"]
        new_s = ""
        for c in s:
            if rand.random() < p and c == " ":
                if rand.random() < 0.33:
                    # multiple whitespaces
                    new_s += rand.randint(1, 5) * " " + c
                else:
                    new_s += rand.choice(noise_chars)
            else:
                new_s += c
        return new_s

    @pytest.mark.parametrize("execution", list(range(100)))
    @pytest.mark.parametrize("seed", list(range(20)))
    def test_clean_sequence(self, execution: int, seed: int) -> None:
        cleaned_sequence = TestData.randomly_insert_whitespaces(string.ascii_letters, p=0.2, seed=seed).strip()
        uncleaned_sequence = TestData.add_noise(cleaned_sequence, p=0.2, seed=seed)

        assert utils.clean_sequence(uncleaned_sequence) == cleaned_sequence

    @pytest.mark.parametrize("sequence", [
        ("This is a valid sequence", True),
        ("He was born 1992", True),
        ("123899213", False),
        ("This <html> is an invalid sequence", False),
        (", .. 1++", False)
    ])
    @pytest.mark.parametrize("min_length", [0, 5, 10])
    def test_is_valid_sequence(self, sequence: Tuple[str, int], min_length: int) -> None:
        assert utils.is_valid_sequence(sequence[0], min_length=min_length) == \
               (sequence[1] if len(sequence[0]) >= min_length else False)

    @pytest.mark.parametrize("token", ["this", "Test", "something"])
    @pytest.mark.parametrize("include", [
        (0,),
        (1,),
        (2,),
        (3,),
        (0, 1, 2, 3)
    ])
    @pytest.mark.parametrize("seed", list(range(20)))
    def test_edit_token(self, token: str, include: Tuple[int], seed: int) -> None:
        edited_token = preprocessing.edit_token(token, include=include, rand=np.random.default_rng(seed))[0]
        if include == 0:
            # insert
            assert len(edited_token) == len(token) + 1
        elif include == 1:
            # delete
            assert len(edited_token) == len(token) - 1
        elif include == 2:
            # swap
            assert len(token) == len(edited_token)
            assert len(set(token) - set(edited_token)) == 0
        elif include == 3:
            # replace
            assert len(token) == len(edited_token)
            assert len(set(token) - set(edited_token)) == 1

        assert edited_token != token
