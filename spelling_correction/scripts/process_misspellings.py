import collections
import glob
import json
import os
import pickle
from functools import partial
from typing import Dict, List, Set, Callable, Tuple, Iterable, Optional

import Levenshtein
import aspell
import hunspell
import numpy as np
from tqdm import tqdm

import gnn_lib.data.utils
from gnn_lib.data import utils
from gnn_lib.utils import common, io
from spelling_correction import MISSPELLINGS_DIR, DATA_DIR, DICTIONARIES_DIR

HUNSPELL = hunspell.HunSpell("/usr/share/hunspell/en_US.dic", "/usr/share/hunspell/en_US.aff")

ASPELL = aspell.Speller("lang", "en")
ASPELL.setConfigKey("sug-mode", "normal")


def invalid_word(word: str) -> bool:
    return not utils.is_valid_word(word)


def invalid_pair(correct: str, misspelled: str, max_ed: int = 5) -> bool:
    return (
            misspelled == correct or
            Levenshtein.distance(correct, misspelled) > max_ed
    )


def hunspell_suggest(word: str, top_k: int) -> Set[str]:
    return set(w for w in HUNSPELL.suggest(word)[:top_k])


def aspell_suggest(word: str, top_k: int) -> Set[str]:
    return set(w for w in ASPELL.suggest(word)[:top_k])


def suggest_top_k(suggestion_fn: Callable, top_k: int) -> Callable:
    return partial(suggestion_fn, top_k=top_k)


def get_suggestions_for_file(filepath: str,
                             suggestion_fn: Callable[[str], Set[str]],
                             is_dictionary: bool = False) -> Dict[str, Set[str]]:
    suggestions_dict = {}
    with open(filepath, "r", encoding="utf8") as f:
        for line in tqdm(f, total=io.line_count(filepath), desc=f"Getting suggestions for file {filepath}"):
            word = line.strip()
            if is_dictionary:
                word = word.split("\t")[0]
            if word == "":
                continue
            if word not in suggestions_dict:
                suggestions = suggestion_fn(word)
                if len(suggestions) == 0:
                    continue
                suggestions_dict[word] = suggestions
    return suggestions_dict


def get_suggestions_for_list(words: List[str], suggestion_fn: Callable[[str], Set[str]]) -> Dict[str, Set[str]]:
    suggestions_dict = {}
    for word in tqdm(words, desc=f"Getting suggestion for word list"):
        if word == "":
            continue
        if word not in suggestions_dict:
            suggestions = suggestion_fn(word)
            if len(suggestions) == 0:
                continue
            suggestions_dict[word] = suggestions
    return suggestions_dict


def process_birkbeck(path: str) -> Dict[str, Set[str]]:
    files = glob.glob(os.path.join(path, "*.dat"))
    misspellings = collections.defaultdict(set)
    for file in files:
        with open(file, "r", encoding="utf8") as f:
            right_spelling = ""
            for line in tqdm(f, total=io.line_count(file), desc=f"Processing Birkbeck file at {file}"):
                line = gnn_lib.data.utils.clean_sequence(line)
                if line == "":
                    continue
                if line.startswith("$"):
                    right_spelling = gnn_lib.data.utils.clean_sequence(line[1:])
                    if right_spelling == "?":
                        right_spelling = ""
                        continue
                else:
                    misspellings[right_spelling].add(line)
    logger.info(f"Birkbeck: Found {len(misspellings)} correct words "
                f"with {sum(len(v) for v in misspellings.values())} misspellings in total")
    return misspellings


def process_wikipedia(path: str) -> Dict[str, Set[str]]:
    misspellings = collections.defaultdict(set)
    with open(path, "r", encoding="utf8") as f:
        for line in tqdm(f, total=io.line_count(path), desc=f"Processing Wikipedia misspellings at {path}"):
            line = line.strip()
            if line == "":
                continue
            misspelling, correct_spellings = line.split("->")
            for correct_spelling in correct_spellings.split(","):
                correct_spelling = gnn_lib.data.utils.clean_sequence(correct_spelling)
                misspelling = gnn_lib.data.utils.clean_sequence(misspelling)
                misspellings[correct_spelling].add(misspelling)
    logger.info(f"Wikipedia: Found {len(misspellings)} correct words "
                f"with {sum(len(v) for v in misspellings.values())} misspellings in total")
    return misspellings


def process_toefl_spell(path: str) -> Dict[str, Set[str]]:
    misspellings = collections.defaultdict(set)
    with open(path, "r", encoding="utf8") as f:
        for i, line in tqdm(enumerate(f), total=io.line_count(path), desc=f"Processing TOEFL spell file at {path}"):
            if i == 0:
                continue
            line = line.strip()
            if line == "":
                continue
            _, _, misspelling, _, correction = line.split("\t")
            correction = gnn_lib.data.utils.clean_sequence(correction)
            misspelling = gnn_lib.data.utils.clean_sequence(misspelling)
            misspellings[correction].add(misspelling)
    logger.info(f"TOEFL Spell: Found {len(misspellings)} correct words "
                f"with {sum(len(v) for v in misspellings.values())} misspellings in total")
    return misspellings


def process_moe(path: str) -> Dict[str, Set[str]]:
    misspellings = collections.defaultdict(set)
    with open(path, "r", encoding="utf8") as f:
        for line in tqdm(f, total=io.line_count(path), desc=f"Processing MOE misspellings at {path}"):
            split = line.split("\t")
            if len(split) != 2:
                continue
            misspelled = utils.clean_sequence(split[0])
            correct = utils.clean_sequence(split[1])
            misspellings[correct].add(misspelled)
    logger.info(f"MOE: Found {len(misspellings)} correct words with "
                f"{sum(len(v) for v in misspellings.values())} misspellings in total")
    return misspellings


def process_neuspell(path: str) -> Dict[str, Set[str]]:
    misspellings = collections.defaultdict(set)
    with open(os.path.join(path, "combined_data"), "r", encoding="utf8") as correct_f, \
            open(os.path.join(path, "combined_data.noise"), "r", encoding="utf8") as noise_f:
        for corr, noise in zip(correct_f, noise_f):
            corr = utils.clean_sequence(corr)
            noise = utils.clean_sequence(noise)
            misspellings[corr].add(noise)
    logger.info(f"Neuspell: Found {len(misspellings)} correct words with "
                f"{sum(len(v) for v in misspellings.values())} misspellings in total")
    return misspellings


def process_bea(
        m2_files: List[str],
        gec_error_types: Tuple[str, ...] = (
                "R:SPELL", "R:ORTH",
                "R:NOUN:INFL", "R:NOUN:NUM",
                "R:VERB:FORM", "R:VERB:INFL", "R:VERB:SVA", "R:VERB:TENSE")
) -> Dict[str, Set[str]]:
    misspellings = collections.defaultdict(set)
    assert all(e[0] == "R" for e in gec_error_types), "only replacement errors are allowed"
    error_types = set(gec_error_types)
    error_types_count = collections.Counter()
    sequence = ""
    for m2_file in tqdm(m2_files, "Extracting misspellings from BEA m2 files"):
        with open(m2_file, "r", encoding="utf8") as inf:
            for line in tqdm(inf, f"Extracting from {m2_file}", total=io.line_count(m2_file), leave=False):
                line = line.strip()
                if line.startswith("S"):
                    sequence = gnn_lib.data.utils.clean_sequence(line)
                elif line.startswith("A"):
                    words = sequence.split(" ")[1:]
                    positions, error_type, replacement = line.split("|||")[:3]
                    from_, to_ = positions.split(" ")[1:]
                    from_ = int(from_)
                    to_ = int(to_)
                    if from_ + 1 != to_ or from_ >= len(words):
                        continue
                    misspelled = words[from_].strip()
                    replacement = replacement.strip()
                    if invalid_word(replacement) or invalid_word(misspelled):
                        continue
                    if error_type in error_types:
                        error_types_count[error_type] += 1
                        misspellings[replacement].add(misspelled)
    logger.info(f"BEA: Found {len(misspellings)} correct words with "
                f"{sum(len(v) for v in misspellings.values())} misspellings in total for GEC error types "
                f"{gec_error_types} ({error_types_count})")
    return misspellings


def process_tweet(path: str) -> Dict[str, Set[str]]:
    misspellings = collections.defaultdict(set)
    with open(path, "r", encoding="utf8") as inf:
        for line in inf:
            line = line.strip()
            split = line.split("\t")
            corrupt, correct = split[0], split[1]
            misspellings[correct].add(corrupt)
    logger.info(f"Tweet: Found {len(misspellings)} correct words with "
                f"{sum(len(v) for v in misspellings.values())} misspellings in total")
    return misspellings


def process_homophones(path: str) -> Dict[str, Set[str]]:
    misspellings = collections.defaultdict(set)
    skip = True
    with open(path, "r", encoding="utf8") as inf:
        for line in inf:
            line = line.strip()
            if line.startswith("---"):
                # next line is the first with content so set skip to false
                skip = False
                continue
            elif skip:
                continue
            split = line.split(",")
            if len(split) <= 1:
                continue
            for w1 in split:
                for w2 in split:
                    if w1 == w2:
                        continue
                    misspellings[w1].add(w2)
    logger.info(f"Homophones: Found {len(misspellings)} correct words with "
                f"{sum(len(v) for v in misspellings.values())} homophones in total")
    return misspellings


def merge_dicts(*args: Dict) -> Dict[str, List[str]]:
    merged = {}
    for key in merge_keys(*args):
        values = merge_values(*args, key=key)
        if len(values) == 0:
            continue
        merged[key] = values
    return merged


def merge_keys(*args: Dict) -> List[str]:
    keys = set(args[0].keys())
    for i in range(1, len(args)):
        keys = keys.union(set(args[i].keys()))
    return list(keys)


def merge_values(*args: Dict, key: str) -> List[str]:
    values = args[0].get(key, set())
    for i in range(1, len(args)):
        values = values.union(args[i].get(key, set()))
    values.discard(key)
    return list(values)


def filter_misspellings(misspellings: Dict[str, Iterable[str]],
                        dictionary: Optional[Dict[str, int]] = None,
                        min_freq: int = 2) -> Dict[str, List[str]]:
    def in_dict_and_frequent_enough(w: str) -> bool:
        return dictionary is not None and dictionary.get(w, -1) >= min_freq

    if len(misspellings) == 0:
        return {}
    filtered = collections.defaultdict(list)
    invalid_correct = 0
    invalid_missp = 0
    for correct, missp in misspellings.items():
        if invalid_word(correct) or not in_dict_and_frequent_enough(correct):
            invalid_correct += 1
            continue
        for m in missp:
            if not invalid_word(m) and not invalid_pair(correct, m, max_ed=10):
                filtered[correct].append(m)
            else:
                invalid_missp += 1
    logger.info(f"Filtered misspellings: {100 * invalid_correct / len(misspellings.keys()):.2f}% of "
                f"words were invalid, {100 * invalid_missp / sum(len(v) for v in misspellings.values()):.2f}% of "
                f"misspellings were invalid")
    return filtered


def train_test_split_misspellings(
        misspellings: Dict[str, List[str]],
        split: Tuple[float, float],
        seed: int) -> Tuple[Dict[str, List[str]], ...]:
    assert sum(split) == 1.0 and all(0 < s < 1 for s in split)

    train_missp = collections.defaultdict(list)
    test_missp = collections.defaultdict(list)

    rand = np.random.RandomState(seed)

    for correct, missp in tqdm(misspellings.items(), "Splitting misspellings into train and test"):
        for m in missp:
            if rand.random_sample() < split[0]:
                # train
                train_missp[correct].append(m)
            else:
                # test
                test_missp[correct].append(m)
    return train_missp, test_missp


if __name__ == "__main__":
    logger = common.get_logger("PROCESS_MISSPELLINGS")
    os.makedirs(MISSPELLINGS_DIR, exist_ok=True)

    dictionary = io.dictionary_from_file(os.path.join(DICTIONARIES_DIR, "merged_train_100k.txt"))
    out_file = os.path.join(MISSPELLINGS_DIR, "misspellings.json")
    if os.path.exists(out_file):
        logger.info(f"Misspellings at {out_file} already exist")
        with open(out_file, "r", encoding="utf8") as inf:
            all_misspellings = json.load(inf)
    else:
        tweet_misspellings = process_tweet(os.path.join(DATA_DIR, "raw", "misspellings", "tweet", "tweet.txt"))

        homophones_misspellings = process_homophones(
            os.path.join(DATA_DIR, "raw", "misspellings", "homophones", "homofonz", "homophones-1.01.txt"))

        aspell_suggestions = get_suggestions_for_file(
            os.path.join(DICTIONARIES_DIR, "merged_train_100k.txt"),
            suggestion_fn=suggest_top_k(aspell_suggest, 5),
            is_dictionary=True
        )

        hunspell_suggestions = get_suggestions_for_file(
            os.path.join(DICTIONARIES_DIR, "merged_train_100k.txt"),
            suggestion_fn=suggest_top_k(hunspell_suggest, 5),
            is_dictionary=True
        )

        birkbeck_misspellings = process_birkbeck(os.path.join(DATA_DIR, "raw", "misspellings", "birkbeck"))

        wikipedia_misspellings = process_wikipedia(
            os.path.join(DATA_DIR, "raw", "misspellings", "wikipedia", "misspellings.txt"))

        toefl_spell_misspellings = process_toefl_spell(
            os.path.join(DATA_DIR, "raw", "misspellings", "toefl_spell", "Annotations.tsv"))

        neuspell_misspellings = process_neuspell(os.path.join(DATA_DIR, "raw", "neuspell", "traintest", "wo_context"))

        fce_m2_files = io.glob_safe(os.path.join(DATA_DIR, "raw", "fce_v2.1.bea19", "fce", "m2", "*.m2"))
        lang8_m2_files = io.glob_safe(os.path.join(DATA_DIR, "raw", "lang8.bea19", "*.m2"))
        nucle_m2_files = io.glob_safe(os.path.join(DATA_DIR, "raw", "release3.3", "bea2019", "*.m2"))
        wi_locness_m2_files = io.glob_safe(
            os.path.join(DATA_DIR, "raw", "wi+locness_v2.1.bea19", "wi+locness", "m2", "*.m2"))
        bea_misspellings = process_bea(wi_locness_m2_files + fce_m2_files + lang8_m2_files + nucle_m2_files)

        moe_misspellings = process_moe(
            os.path.join(DATA_DIR, "raw", "misspellings", "moe", "moe_misspellings_train.tsv"))

        all_misspellings = merge_dicts(
            birkbeck_misspellings,
            wikipedia_misspellings,
            toefl_spell_misspellings,
            neuspell_misspellings,
            bea_misspellings,
            moe_misspellings,
            aspell_suggestions,
            hunspell_suggestions,
            tweet_misspellings,
            homophones_misspellings
        )

        # filter out invalid pairs
        all_misspellings = filter_misspellings(all_misspellings, dictionary, min_freq=5)

        with open(out_file, "w", encoding="utf8") as f:
            json.dump(all_misspellings, f)

    logger.info(f"Got {len(all_misspellings.keys())} misspelled words with "
                f"{sum(len(v) for k, v in all_misspellings.items())} misspellings in total")

    train_out_file = os.path.join(MISSPELLINGS_DIR, "train_misspellings.json")
    dev_out_file = os.path.join(MISSPELLINGS_DIR, "dev_misspellings.json")
    test_out_file = os.path.join(MISSPELLINGS_DIR, "test_misspellings.json")

    if os.path.exists(train_out_file) and os.path.exists(test_out_file):
        logger.info(f"Train, dev and test misspellings at {train_out_file}, {dev_out_file} and "
                    f"{test_out_file} already exist")
        with open(train_out_file, "r", encoding="utf8") as inf:
            train_misspellings = json.load(inf)
        with open(dev_out_file, "r", encoding="utf8") as inf:
            dev_misspellings = json.load(inf)
        with open(test_out_file, "r", encoding="utf8") as inf:
            test_misspellings = json.load(inf)
    else:
        train_misspellings, dev_and_test_misspellings = train_test_split_misspellings(
            all_misspellings,
            split=(0.95, 0.05),
            seed=22
        )
        dev_misspellings, test_misspellings = train_test_split_misspellings(
            dev_and_test_misspellings,
            split=(0.5, 0.5),
            seed=22
        )
        with open(train_out_file, "w", encoding="utf8") as f:
            json.dump(train_misspellings, f)
        with open(dev_out_file, "w", encoding="utf8") as f:
            json.dump(dev_misspellings, f)
        with open(test_out_file, "w", encoding="utf8") as f:
            json.dump(test_misspellings, f)

    logger.info(f"Split misspellings into {len(train_misspellings)}, {len(dev_misspellings)} and "
                f"{len(test_misspellings)} correct words for train, dev and test with "
                f"{sum(len(v) for v in train_misspellings.values())}, {sum(len(v) for v in dev_misspellings.values())} "
                f"and {sum(len(v) for v in test_misspellings.values())} misspellings in total respectively")
