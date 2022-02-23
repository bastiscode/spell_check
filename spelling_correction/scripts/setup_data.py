import argparse
import copy
import json
import multiprocessing as mp
import os
import re
import time
from collections import defaultdict
from typing import List, Set, Dict

from tqdm import tqdm
import numpy as np
from spacy.lang.en import English

from gnn_lib.utils import common, io
from gnn_lib.data.utils import clean_sequence, is_valid_sequence, tokens_to_text

from spelling_correction.utils import neuspell


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input-dir", type=str, required=True,
                        help="Path to input directory")
    parser.add_argument("-o", "--output-dir", type=str, required=True,
                        help="Path to output directory")
    parser.add_argument("-d", "--dataset", type=str, choices=[
        "bea2019",
        "github_typo",
        "bookcorpus",
        "wikidump",
        "multi30k",
        "neuspell"
    ], required=True, help="Dataset to setup")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing dataset setup")
    return parser.parse_args()


SAMPLE = Dict[str, str]


def _save_samples_to_jsonl(samples: List[SAMPLE],
                           save_to: str,
                           mode: str = "w",
                           verbose: bool = True) -> None:
    if verbose:
        logger.info(f"Saving {len(samples)} samples to {save_to}")
    d, _ = os.path.split(save_to)
    os.makedirs(d, exist_ok=True)
    with open(save_to, mode, encoding="utf8") as f:
        for sample in samples:
            json.dump(sample, f, ensure_ascii=False)
            f.write("\n")


def _exists(path: str) -> bool:
    return os.path.exists(path)


def _check_paths(filepath: str, save_to: str, overwrite: bool) -> bool:
    if not _exists(filepath):
        logger.info(f"Skipping processing of {filepath} since it does not exist.")
        return False
    if _exists(save_to) and not overwrite:
        logger.info(f"Skipping processing of {filepath} since {save_to} already exists. "
                    f"Set overwrite flag to force processing.")
        return False
    logger.info(f"Processing {filepath}...")
    return True


def process_m2_file(filepath: str, save_to: str, overwrite: bool = False) -> None:
    if not _check_paths(filepath, save_to, overwrite):
        return
    samples = _process_m2_file(filepath)
    _save_samples_to_jsonl(samples, save_to=save_to)


def process_m2_dir(directory: str, save_to: str, overwrite: bool = False) -> None:
    os.makedirs(save_to, exist_ok=True)
    files = io.glob_safe(os.path.join(directory, "*.m2"))
    for file in files:
        process_m2_file(file, save_to=os.path.join(save_to, os.path.split(file)[1] + ".jsonl"), overwrite=overwrite)


def _process_m2_file(filepath: str) -> List[SAMPLE]:
    with open(filepath, "r", encoding="utf8") as f:
        raw = f.read()

    m2_blocks_re = re.compile(r"(?:\r?\n){2,}")

    m2_blocks = re.split(m2_blocks_re, raw)
    samples = [sample for block in m2_blocks for sample in _parse_m2_block(block)]

    return samples


def _parse_m2_block(s: str) -> List[SAMPLE]:
    samples: List[SAMPLE] = []
    lines = s.splitlines()
    if len(lines) == 0:
        return samples
    assert lines[0].startswith("S")
    orig_tokens = lines[0].split()[1:]
    annotations = defaultdict(list)
    for i in range(1, len(lines)):
        assert lines[i].startswith("A")
        corr = lines[i].split("|||")
        assert len(corr) == 6, f"expected corr to be of length 6, but got {corr}"
        prefix, error_type, corrections_str, req, _, ann_id = corr
        _, from_idx, to_idx = prefix.split()
        corrections = corrections_str.split()
        annotations[ann_id].append((int(from_idx), int(to_idx), corrections, error_type))

    for ann_id, corrections in annotations.items():
        corrected_tokens = copy.deepcopy(orig_tokens)
        # account for changes of the indices due to insertions or deletions
        length_change = 0
        for from_idx, to_idx, correction, error_type in corrections:
            from_idx += length_change
            to_idx += length_change
            if error_type.strip() == "noop":
                continue
            del corrected_tokens[from_idx: to_idx]
            for corr in correction:
                corrected_tokens.insert(from_idx, corr)
                from_idx += 1
            length_change = len(corrected_tokens) - len(orig_tokens)
        item = {"sequence": clean_sequence(tokens_to_text(orig_tokens)),
                "target_sequence": clean_sequence(tokens_to_text(corrected_tokens))}
        samples.append(item)
    return samples


def process_github_typo(filepath: str, save_to: str, overwrite: bool = False):
    if not _check_paths(filepath, save_to, overwrite):
        return

    def _process_line(line: str) -> List[SAMPLE]:
        try:
            json_obj = json.loads(line, strict=False)
        except json.decoder.JSONDecodeError as e:
            logger.error(f"JSON decode error: {e}")
            return []
        samples = []
        for edit in json_obj["edits"]:
            if "is_typo" not in edit:
                continue
            if not edit["is_typo"]:
                continue
            src = edit["src"]
            tgt = edit["tgt"]
            if src["lang"] != "eng" or tgt["lang"] != "eng":
                continue
            sample: SAMPLE = {"sequence": clean_sequence(src["text"]),
                              "target_sequence": clean_sequence(tgt["text"])}
            samples.append(sample)
        return samples

    with open(filepath, "r", encoding="utf8") as f:
        data: List[SAMPLE] = [sample
                              for line in f
                              for sample in _process_line(line)]
    _save_samples_to_jsonl(samples=data,
                           save_to=save_to)


def _process_wikidump_file(fp: str, save_to: str, overwrite: bool, invalid_doc_ids: Set[int]):
    if not _check_paths(fp, save_to, overwrite):
        return
    doc_regex = re.compile(r"<doc id=\"(\d+)\".*?>(.*?)</doc>", re.DOTALL)
    nlp = English()
    nlp.add_pipe("sentencizer")
    nlp.max_length = 1e8

    samples: List[SAMPLE] = []
    num_invalid = 0

    with open(fp, "r", encoding="utf8") as f:
        raw = f.read()

    for match in re.finditer(doc_regex, raw):
        doc_id = int(match.group(1))
        if doc_id in invalid_doc_ids:
            logger.info(f"Skipping document with doc id {doc_id}")
            continue

        g = match.group(2)
        docs = nlp.pipe([g])
        for doc in docs:
            for s in doc.sents:
                sample = clean_sequence(str(s))
                if not is_valid_sequence(sample, min_length=1):
                    num_invalid += 1
                    continue
                samples.append({"sequence": sample})
    logger.info(
        f"Percentage of invalid sequences in file {fp} is "
        f"{(num_invalid * 100) / max(1, num_invalid + len(samples)):.2f} %")
    _save_samples_to_jsonl(samples, save_to=save_to, verbose=False)


def process_wikidump(directory: str, save_to: str, overwrite: bool = False):
    files = sorted(io.glob_safe(os.path.join(directory, "*", "wiki_*")))
    save_tos = [os.path.join(save_to, file.split("/")[-2], file.split("/")[-1] + ".jsonl") for file in files]
    overwrites = [overwrite] * len(files)

    invalid_doc_ids = set()
    dev_article_ids_path = os.path.join(directory, "wikipedia_development_article_ids.txt")
    test_article_ids_path = os.path.join(directory, "wikipedia_test_article_ids.txt")
    if os.path.exists(dev_article_ids_path):
        with open(dev_article_ids_path, "r", encoding="utf8") as dev_ids:
            for line in dev_ids:
                invalid_doc_ids.add(int(line.strip()))
    if os.path.exists(test_article_ids_path):
        with open(test_article_ids_path, "r", encoding="utf8") as test_ids:
            for line in test_ids:
                invalid_doc_ids.add(int(line.strip()))

    logger.info(f"Will ignore {len(invalid_doc_ids)} documents because they are used for dev and test")

    invalid_doc_ids = [invalid_doc_ids] * len(files)

    start = time.monotonic()
    pool = mp.Pool(int(os.environ.get("NUM_PROCESSES", min(mp.cpu_count(), 8))))
    result = pool.starmap_async(_process_wikidump_file, list(zip(files, save_tos, overwrites, invalid_doc_ids)))
    tasks = result._number_left
    left = result._number_left
    while not result.ready():
        if result._number_left < left:
            end = time.monotonic()
            min_since_start = (end - start) / 60
            finished = tasks - result._number_left
            logger.info(f"Processed {finished}/{tasks} chunks\n{common.eta_minutes(min_since_start, finished, tasks)}")
            left = result._number_left
        time.sleep(2)
    result.get()
    pool.close()
    pool.join()

    logger.info(f"Generating Wikidump train, dev and test splits from files")

    _generate_train_dev_test_split(save_tos,
                                   save_to,
                                   dev_percentage=0.01,
                                   test_percentage=0.01,
                                   seed=22)


def _process_bookcorpus_file(fp: str, save_to: str, overwrite: bool):
    if not _check_paths(fp, save_to, overwrite):
        return

    nlp = English()
    nlp.add_pipe("sentencizer")
    nlp.max_length = 1e8

    num_invalid = 0
    samples: List[SAMPLE] = []

    with open(fp, "r", encoding="utf8") as f:
        raw = f.read()

    docs = nlp.pipe([raw])
    for doc in docs:
        for s in doc.sents:
            sample = clean_sequence(str(s))
            if not is_valid_sequence(sample, min_length=1):
                num_invalid += 1
                continue
            samples.append({"sequence": sample})

    logger.info(
        f"Percentage of invalid sequences in file {fp} is "
        f"{(num_invalid * 100) / max(1, num_invalid + len(samples)):.2f} %")

    _save_samples_to_jsonl(samples, save_to=save_to, verbose=False)


def process_bookcorpus(directory: str, save_to: str, overwrite: bool = False):
    files = sorted(io.glob_safe(os.path.join(directory, "*.epub.txt")))
    save_tos = [os.path.join(save_to, os.path.split(file)[-1] + ".jsonl") for file in files]
    overwrites = [overwrite] * len(files)

    start = time.monotonic()
    pool = mp.Pool(int(os.environ.get("NUM_PROCESSES", min(mp.cpu_count(), 8))))
    result = pool.starmap_async(_process_bookcorpus_file, list(zip(files, save_tos, overwrites)))
    tasks = result._number_left
    left = result._number_left
    while not result.ready():
        if result._number_left < left:
            end = time.monotonic()
            min_since_start = (end - start) / 60
            finished = tasks - result._number_left
            logger.info(f"Processed {finished}/{tasks} chunks\n{common.eta_minutes(min_since_start, finished, tasks)}")
            left = result._number_left
        time.sleep(2)
    result.get()
    pool.close()
    pool.join()

    logger.info(f"Generating Bookcorpus train, dev and test splits from files")

    _generate_train_dev_test_split(save_tos,
                                   save_to,
                                   dev_percentage=0.01,
                                   test_percentage=0.01,
                                   seed=22)


def process_multi30k(directory: str, save_to: str, overwrite: bool = False):
    for split in {"test", "val", "train"}:
        load_file = os.path.join(directory, f"{'test2016' if split == 'test' else split}.en")
        save_file = os.path.join(save_to, f"{split}.jsonl")

        if not _check_paths(load_file, save_file, overwrite):
            continue

        samples = []
        with open(load_file, "r", encoding="utf8") as f:
            for s in f:
                sample = clean_sequence(s)
                if not is_valid_sequence(sample, min_length=1):
                    continue
                samples.append({"sequence": sample})

        _save_samples_to_jsonl(samples, save_file)


def process_neuspell(directory: str, save_to: str, overwrite: bool = False) -> None:
    out_file = os.path.join(save_to, "train_1blm.jsonl")

    gt_file = os.path.join(directory, "train.1blm")
    if not _check_paths(gt_file, out_file, overwrite):
        return

    noise_files = io.glob_safe(os.path.join(directory, "train.1blm.noise*"))
    # exclude file with random noise
    noise_files = [f for f in noise_files if not f.endswith("noise.random")]

    samples = []
    for noise_file in noise_files:
        with open(gt_file, "r", encoding="utf8") as gtf, \
                open(noise_file, "r", encoding="utf8") as nf:
            for correct_line, corrupt_line in tqdm(
                    zip(gtf, nf), f"Processing {noise_file}", total=io.line_count(gt_file)):
                correct_line = correct_line.strip()
                corrupt_line = corrupt_line.strip()
                if corrupt_line == "" or corrupt_line == "":
                    continue
                correct_line, corrupt_line = neuspell.clean_sequences(correct_line, corrupt_line)
                if not is_valid_sequence(correct_line) or not is_valid_sequence(corrupt_line):
                    continue
                sample = {"sequence": correct_line, "corrupt_sequence": corrupt_line}
                samples.append(sample)

    _save_samples_to_jsonl(samples, out_file)


def _generate_train_dev_test_split(files: List[str],
                                   save_to: str,
                                   dev_percentage: float,
                                   test_percentage: float,
                                   seed: int) -> None:
    train_path = os.path.join(save_to, "train_files.txt")
    dev_path = os.path.join(save_to, "dev_files.txt")
    test_path = os.path.join(save_to, "test_files.txt")
    if (os.path.exists(train_path)
            and os.path.exists(dev_path)
            and os.path.exists(test_path)):
        return

    rand = np.random.default_rng(seed)
    perm = rand.permutation(len(files))

    num_dev_files = int(np.ceil(len(files) * dev_percentage))
    num_test_files = int(np.ceil(len(files) * test_percentage))

    cum_files = np.cumsum([num_dev_files, num_test_files])

    dev_files = perm[:cum_files[0]]
    test_files = perm[cum_files[0]:cum_files[1]]
    train_files = perm[cum_files[1]:]

    with open(train_path,
              "w",
              encoding="utf8") as f:
        for i in train_files:
            f.write(os.path.relpath(files[i], save_to))
            f.write("\n")

    with open(dev_path,
              "w",
              encoding="utf8") as f:
        for i in dev_files:
            f.write(os.path.relpath(files[i], save_to))
            f.write("\n")

    with open(test_path,
              "w",
              encoding="utf8") as f:
        for i in test_files:
            f.write(os.path.relpath(files[i], save_to))
            f.write("\n")


if __name__ == "__main__":
    args = parse_args()
    logger = common.get_logger("SETUP_DATA")

    INPUT_DIR = args.input_dir
    OUTPUT_DIR = args.output_dir

    if args.dataset == "bookcorpus":
        BOOKCORPUS = os.path.join(INPUT_DIR, "bookcorpus/books1/epubtxt")
        process_bookcorpus(BOOKCORPUS, os.path.join(OUTPUT_DIR, "bookcorpus"), overwrite=args.overwrite)

    elif args.dataset == "bea2019":
        FCE_GLOB = os.path.join(INPUT_DIR, "fce_v2.1.bea19/fce/m2")
        process_m2_dir(FCE_GLOB, os.path.join(OUTPUT_DIR, "fce"), overwrite=args.overwrite)

        LANG8 = os.path.join(INPUT_DIR, "lang8.bea19/lang8.train.auto.bea19.m2")
        process_m2_file(LANG8, os.path.join(OUTPUT_DIR, "lang8.m2.jsonl"), overwrite=args.overwrite)

        WILOCNESS_GLOB = os.path.join(INPUT_DIR, "wi+locness_v2.1.bea19/wi+locness/m2")
        process_m2_dir(WILOCNESS_GLOB, os.path.join(OUTPUT_DIR, "wilocness"), overwrite=args.overwrite)

        NUCLE = os.path.join(INPUT_DIR, "release3.3/bea2019/nucle.train.gold.bea19.m2")
        process_m2_file(NUCLE, os.path.join(OUTPUT_DIR, "nucle.m2.jsonl"), overwrite=args.overwrite)

        GEC = os.path.join(INPUT_DIR, "10gec_annotations")
        process_m2_dir(GEC, os.path.join(OUTPUT_DIR, "gec"), overwrite=args.overwrite)

    elif args.dataset == "github_typo":
        GITHUB_TYPO = os.path.join(INPUT_DIR, "github_typo/github-typo-corpus.v1.0.0.jsonl")
        process_github_typo(GITHUB_TYPO, os.path.join(OUTPUT_DIR, "github_typo.jsonl"),
                            overwrite=args.overwrite)

    elif args.dataset == "wikidump":
        WIKIDUMP = os.path.join(INPUT_DIR, "wikidump")
        process_wikidump(WIKIDUMP, os.path.join(OUTPUT_DIR, "wikidump"), overwrite=args.overwrite)

    elif args.dataset == "multi30k":
        MULTI30K = os.path.join(INPUT_DIR, "multi30k")
        process_multi30k(MULTI30K, os.path.join(OUTPUT_DIR, "multi30k"), overwrite=args.overwrite)

    elif args.dataset == "neuspell":
        NEUSPELL = os.path.join(INPUT_DIR, "neuspell", "traintest")
        process_neuspell(NEUSPELL, os.path.join(OUTPUT_DIR, "neuspell"), overwrite=args.overwrite)

    else:
        raise ValueError(f"Unknown dataset {args.dataset}")
