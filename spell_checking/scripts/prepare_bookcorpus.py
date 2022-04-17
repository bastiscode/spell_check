import argparse
import json
import multiprocessing as mp
import os
from typing import Tuple

from spacy.lang.en import English
from tqdm import tqdm

from nsc.data.utils import clean_sequence, is_valid_sequence
from nsc.utils import io, common

nlp = English()
nlp.add_pipe("sentencizer")
nlp.max_length = 1e8

logger = common.get_logger("PREPARE_BOOKCORPUS")


def txt_file_to_jsonl(args: Tuple[str, str, bool]) -> None:
    in_file, out_file, use_paragraphs = args
    with open(in_file, "r", encoding="utf8") as txt_file:
        raw = txt_file.read()

    num_invalid = 0
    samples = []
    if use_paragraphs:
        for line in raw.splitlines():
            if not is_valid_sequence(line, min_length=1):
                num_invalid += 1
                continue
            line = clean_sequence(line)
            samples.append({"sequence": line})
    else:
        g = clean_sequence(raw)
        docs = nlp.pipe([g])
        for doc in docs:
            for s in doc.sents:
                sequence = str(s)
                if not is_valid_sequence(sequence, min_length=1):
                    num_invalid += 1
                    continue
                samples.append({"sequence": sequence})

    with open(out_file, "w", encoding="utf8") as json_file:
        for sample in samples:
            json.dump(sample, json_file, ensure_ascii=False)
            json_file.write("\n")

    # logger.debug(
    #     f"Percentage of invalid sequences in file {in_file} is "
    #     f"{(num_invalid * 100) / max(1, num_invalid + len(samples)):.2f} %")


def txt_to_jsonl(args: argparse.Namespace) -> None:
    os.makedirs(args.out_dir, exist_ok=True)
    in_files = io.glob_safe(os.path.join(args.in_dir, "books1", "epubtxt", "*.txt"))

    logger.info(f"Preparing bookcorpus from {args.in_dir}, saving to {args.out_dir} "
                f"(use_paragraphs={args.use_paragraphs})")

    tasks = []
    for in_file in in_files:
        in_file_name = os.path.basename(in_file)
        out_file = os.path.join(args.out_dir, f"{in_file_name}.jsonl")
        tasks.append((in_file, out_file, args.use_paragraphs))

    num_processes = int(os.environ.get("NUM_PROCESSES", min(len(os.sched_getaffinity(0)), 8)))

    with mp.Pool(num_processes) as pool:
        for _ in tqdm(pool.imap_unordered(txt_file_to_jsonl, tasks), total=len(tasks)):
            pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-dir", type=str, required=True)
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--use-paragraphs", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    txt_to_jsonl(parse_args())
