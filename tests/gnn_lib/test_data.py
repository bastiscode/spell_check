import json
import logging
import os
import time
import random
from typing import Any, List, Dict

import pytest

from gnn_lib.data import tokenization, variants
from gnn_lib.utils import common
from tests import TOKENIZER_DIR, DATA_DIR


@pytest.fixture()
def variant(request: Any) -> variants.DatasetVariant:
    word_whitespace_cfg = get_tokenizer_cfg(tokenization.Tokenizers.WORD)
    bpe_cfg = get_tokenizer_cfg(tokenization.Tokenizers.BPE)
    variant = request.param

    if variant == variants.DatasetVariants.SED_SEQUENCE_TOKEN:
        cfg = variants.SEDSequenceTokenConfig(tokenizer=tokenization.CharTokenizerConfig())
    elif variant == variants.DatasetVariants.SED_SEQUENCE_WORD:
        cfg = variants.SEDSequenceConfig(tokenizer=word_whitespace_cfg)
    elif variant == variants.DatasetVariants.SED_WORDS:
        cfg = variants.SEDWordsConfig(tokenizer=word_whitespace_cfg)
    elif variant == variants.DatasetVariants.TOKENIZATION_REPAIR:
        cfg = variants.TokenizationRepairConfig()
    elif variant == variants.DatasetVariants.TOKENIZATION_REPAIR_GROUPS:
        cfg = variants.TokenizationRepairGroupsConfig()
    elif variant == variants.DatasetVariants.SEC_WORDS_NMT:
        cfg = variants.SECWordsNMTConfig(word_tokenizer=word_whitespace_cfg,
                                         sub_word_tokenizer=tokenization.CharTokenizerConfig())
    elif variant == variants.DatasetVariants.SEC_NMT:
        cfg = variants.SECNMTConfig(input_tokenizer=bpe_cfg, output_tokenizer=bpe_cfg)
    else:
        raise ValueError(f"Unknown variant {variant.name}")

    return variants.get_variant_from_config(cfg, 22)


def get_tokenizer_cfg(tokenizer: tokenization.Tokenizers) -> tokenization.TokenizerConfig:
    if tokenizer == tokenization.Tokenizers.CHAR:
        return tokenization.CharTokenizerConfig()
    elif tokenizer == tokenization.Tokenizers.WORD:
        return tokenization.WordWhitespaceTokenizerConfig(
            file_path=os.path.join(TOKENIZER_DIR, "word_whitespace", "test.pkl")
        )
    elif tokenizer == tokenization.Tokenizers.BPE:
        return tokenization.BPETokenizerConfig(
            file_path=os.path.join(TOKENIZER_DIR, "bpe", "test.pkl")
        )
    else:
        raise ValueError(f"Unknown tokenizer {tokenizer.name}")


class TestVariants:
    logger: logging.Logger
    test_sequences: List[Dict[str, str]]

    @classmethod
    def setup_class(cls) -> None:
        cls.logger = common.get_logger("TEST_DATA_UTILS")
        test_sequences = []
        with open(os.path.join(DATA_DIR, "cleaned", "wikidump_excerpt.jsonl"), "r", encoding="utf8") as test_file:
            for line in test_file:
                test_sequences.append(json.loads(line))
        cls.test_sequences = test_sequences

    @pytest.mark.parametrize("iterations", [1000])
    @pytest.mark.parametrize("variant", [v for v in variants.DatasetVariants], indirect=True)
    @pytest.mark.parametrize("is_inference", [True, False])
    def test_variants_speed(self,
                            iterations: int,
                            variant: variants.DatasetVariant,
                            is_inference: bool) -> None:
        rand = random.Random(22)
        total_time = 0.
        for i in range(iterations):
            sequence = rand.choice(self.test_sequences)["sequence"]
            start = time.monotonic()
            new_sequence, g, label = variant.prepare_sequence(sequence, is_inference=is_inference)
            end = time.monotonic()
            if is_inference:
                assert new_sequence == sequence
            total_time += end - start
        self.logger.info(
            f"Variant {variant.name} takes "
            f"{(total_time / iterations) * 1e3:.2f}ms on avg over {iterations} iterations "
            f"when {'' if is_inference else 'not '}in inference mode"
        )
