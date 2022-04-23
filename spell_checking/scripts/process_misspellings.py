import argparse
import collections
import glob
import json
import os
from functools import partial
from typing import Dict, List, Set, Callable, Tuple, Iterable, Optional

import aspell
import hunspell
import numpy as np
from tqdm import tqdm

from nsc.data import utils
from nsc.utils import common, io

from spell_checking.utils.edit_distance import edit_distance

HUNSPELL = hunspell.HunSpell("/usr/share/hunspell/en_US.dic", "/usr/share/hunspell/en_US.aff")

ASPELL = aspell.Speller("lang", "en")
ASPELL.setConfigKey("sug-mode", "normal")


def invalid_word(word: str) -> bool:
    return not utils.is_valid_word(word)


def invalid_pair(correct: str, misspelled: str, max_ed: int = 5) -> bool:
    return (
            misspelled == correct or
            edit_distance(correct, misspelled) > max_ed
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
                line = utils.clean_sequence(line)
                if line == "":
                    continue
                if line.startswith("$"):
                    right_spelling = utils.clean_sequence(line[1:])
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
                correct_spelling = utils.clean_sequence(correct_spelling)
                misspelling = utils.clean_sequence(misspelling)
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
            correction = utils.clean_sequence(correction)
            misspelling = utils.clean_sequence(misspelling)
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
                    sequence = utils.clean_sequence(line)
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


def filter_misspellings(
        misspellings: Dict[str, Iterable[str]],
        dictionary: Optional[Dict[str, int]] = None
) -> Dict[str, List[str]]:
    def in_dict(w: str) -> bool:
        if dictionary is None:
            return True
        else:
            return w in dictionary

    if len(misspellings) == 0:
        return {}
    filtered = collections.defaultdict(list)
    invalid_correct = 0
    invalid_missp = 0
    for correct, missp in misspellings.items():
        if invalid_word(correct) or not in_dict(correct):
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


def save_misspellings(out_file: str, misspellings: Dict) -> None:
    out_dir = os.path.dirname(out_file)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(out_file, "w", encoding="utf8") as of:
        json.dump(misspellings, of)


def load_misspellings(in_file: str) -> Dict:
    with open(in_file, "r", encoding="utf8") as inf:
        return json.load(inf)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--misspellings-dir", type=str, required=True)
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--dictionary", type=str, required=True)
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--no-bea", action="store_true")
    parser.add_argument("--no-neuspell", action="store_true")
    parser.add_argument("--no-moe", action="store_true")
    parser.add_argument("--only-bea", action="store_true")
    parser.add_argument("--only-neuspell", action="store_true")
    parser.add_argument("--only-moe", action="store_true")
    parser.add_argument("--dev-split", type=float, default=0.025)
    parser.add_argument("--test-split", type=float, default=0.025)
    parser.add_argument("--filter-without-dict", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    logger = common.get_logger("PROCESS_MISSPELLINGS")
    os.makedirs(args.out_dir, exist_ok=True)

    dictionary = io.dictionary_from_file(args.dictionary)

    out_file = os.path.join(args.out_dir, "misspellings.json")
    if os.path.exists(out_file):
        logger.info(f"Misspellings at {out_file} already exist")
        all_misspellings = load_misspellings(out_file)
    else:
        misspellings_list = []
        if not args.only_bea and not args.only_neuspell:
            tweet_misspellings = process_tweet(os.path.join(args.misspellings_dir, "tweet", "tweet.txt"))

            homophones_misspellings = process_homophones(
                os.path.join(args.misspellings_dir, "homophones", "homofonz", "homophones-1.01.txt")
            )

            aspell_suggestions = get_suggestions_for_file(
                args.dictionary,
                suggestion_fn=suggest_top_k(aspell_suggest, 5),
                is_dictionary=True
            )

            hunspell_suggestions = get_suggestions_for_file(
                args.dictionary,
                suggestion_fn=suggest_top_k(hunspell_suggest, 5),
                is_dictionary=True
            )

            birkbeck_misspellings = process_birkbeck(os.path.join(args.misspellings_dir, "birkbeck"))

            wikipedia_misspellings = process_wikipedia(
                os.path.join(args.misspellings_dir, "wikipedia", "misspellings.txt")
            )

            toefl_spell_misspellings = process_toefl_spell(
                os.path.join(args.misspellings_dir, "toefl_spell", "Annotations.tsv")
            )

            save_misspellings(os.path.join(args.out_dir, "birkbeck_misspellings.json"), birkbeck_misspellings)
            save_misspellings(os.path.join(args.out_dir, "wikipedia_misspellings.json"), wikipedia_misspellings)
            save_misspellings(os.path.join(args.out_dir, "toefl_spell_misspellings.json"), toefl_spell_misspellings)
            save_misspellings(os.path.join(args.out_dir, "aspell_misspellings.json"), aspell_suggestions)
            save_misspellings(os.path.join(args.out_dir, "hunspell_misspellings.json"), hunspell_suggestions)
            save_misspellings(os.path.join(args.out_dir, "tweet_misspellings.json"), tweet_misspellings)
            save_misspellings(os.path.join(args.out_dir, "homophones_misspellings.json"), homophones_misspellings)

            misspellings_list.extend([
                birkbeck_misspellings,
                wikipedia_misspellings,
                toefl_spell_misspellings,
                aspell_suggestions,
                hunspell_suggestions,
                tweet_misspellings,
                homophones_misspellings
            ])

        if not args.only_moe and not args.no_moe:
            moe_misspellings = process_moe(
                os.path.join(args.misspellings_dir, "moe", "moe_misspellings_train.tsv")
            )
            save_misspellings(os.path.join(args.out_dir, "moe_misspellings.json"), moe_misspellings)
            misspellings_list.append(moe_misspellings)

        if not args.only_bea and not args.no_neuspell:
            neuspell_misspellings = process_neuspell(
                os.path.join(args.data_dir, "raw", "neuspell", "traintest", "wo_context")
            )
            save_misspellings(os.path.join(args.out_dir, "neuspell_misspellings.json"), neuspell_misspellings)
            misspellings_list.append(neuspell_misspellings)

        if not args.only_neuspell and not args.no_bea:
            fce_m2_files = io.glob_safe(os.path.join(args.data_dir, "raw", "fce_v2.1.bea19", "fce", "m2", "*.m2"))
            lang8_m2_files = io.glob_safe(os.path.join(args.data_dir, "raw", "lang8.bea19", "*.m2"))
            nucle_m2_files = io.glob_safe(os.path.join(args.data_dir, "raw", "release3.3", "bea2019", "*.m2"))
            wi_locness_m2_files = io.glob_safe(
                os.path.join(args.data_dir, "raw", "wi+locness_v2.1.bea19", "wi+locness", "m2", "*.m2")
            )
            bea_misspellings = process_bea(wi_locness_m2_files + fce_m2_files + lang8_m2_files + nucle_m2_files)
            save_misspellings(os.path.join(args.out_dir, "bea_misspellings.json"), bea_misspellings)

            misspellings_list.append(bea_misspellings)

        all_misspellings = merge_dicts(*misspellings_list)

        # filter out invalid pairs
        all_misspellings = filter_misspellings(
            all_misspellings,
            None if args.filter_without_dict else dictionary
        )

        save_misspellings(out_file, all_misspellings)

    logger.info(f"Got {len(all_misspellings.keys())} misspelled words with "
                f"{sum(len(v) for k, v in all_misspellings.items())} misspellings in total")

    train_out_file = os.path.join(args.out_dir, "train_misspellings.json")
    dev_out_file = os.path.join(args.out_dir, "dev_misspellings.json")
    test_out_file = os.path.join(args.out_dir, "test_misspellings.json")

    if os.path.exists(train_out_file) and os.path.exists(test_out_file):
        logger.info(f"Train, dev and test misspellings at {train_out_file}, {dev_out_file} and "
                    f"{test_out_file} already exist")
        train_misspellings = load_misspellings(train_out_file)
        dev_misspellings = load_misspellings(dev_out_file)
        test_misspellings = load_misspellings(test_out_file)
    else:
        dev_and_test = args.dev_split + args.test_split
        # split all into [train, dev_and_test]
        train_misspellings, dev_and_test_misspellings = train_test_split_misspellings(
            all_misspellings,
            split=(1 - dev_and_test, dev_and_test),
            seed=22
        )
        # split dev_and_test into [dev, test]
        dev_misspellings, test_misspellings = train_test_split_misspellings(
            dev_and_test_misspellings,
            split=(args.dev_split / dev_and_test, args.test_split / dev_and_test),
            seed=22
        )
        save_misspellings(train_out_file, train_misspellings)
        save_misspellings(dev_out_file, dev_misspellings)
        save_misspellings(test_out_file, test_misspellings)

    logger.info(f"Split misspellings into {len(train_misspellings)}, {len(dev_misspellings)} and "
                f"{len(test_misspellings)} correct words for train, dev and test with "
                f"{sum(len(v) for v in train_misspellings.values())}, {sum(len(v) for v in dev_misspellings.values())} "
                f"and {sum(len(v) for v in test_misspellings.values())} misspellings in total respectively")
