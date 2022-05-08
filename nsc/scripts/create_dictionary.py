import argparse
import multiprocessing as mp
import os
import random
from collections import Counter

from tqdm import tqdm

from nsc.data import utils
from nsc.utils import common, io


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    in_file_group = parser.add_mutually_exclusive_group()
    in_file_group.add_argument("--in-files", type=str, nargs="+")
    in_file_group.add_argument("--in-file", type=str)
    parser.add_argument("--max-sequences", type=int, default=None)
    parser.add_argument("--out-file", type=str, required=True)
    parser.add_argument("--top-k", type=int, required=True)
    parser.add_argument("--seed", type=int, default=22)
    return parser.parse_args()


def create(args: argparse.Namespace) -> None:
    logger = common.get_logger("CREATE_DICTIONARY")

    if os.path.exists(args.out_file):
        logger.info(f"Dictionary {args.out_file} already exists")
        return

    if args.in_file:
        files = []
        with open(args.in_file, "r", encoding="utf8") as f:
            for line in f:
                line = line.strip()
                if line == "":
                    continue
                files.append(os.path.join(os.path.dirname(args.in_file), line))
    else:
        files = args.in_files

    if len(files) == 0:
        logger.error("Got no input files")
        return

    files = sorted(files)
    rand = random.Random(args.seed)
    rand.shuffle(files)
    if args.max_sequences is not None and args.max_sequences > 0:
        running_sum = 0
        num_files = 0
        for file in files:
            running_sum += io.line_count(file)
            num_files += 1
            if running_sum >= args.max_sequences:
                break
        files = files[:num_files]

    ctx = mp.get_context("spawn")

    dictionary = Counter()
    with ctx.Pool(processes=min(int(os.getenv("NSC_NUM_PROCESSES", len(os.sched_getaffinity(0)))), len(files))) as pool:
        for d in tqdm(
                pool.imap_unordered(utils.get_word_frequencies_from_file, files),
                total=len(files),
                desc="calculating word frequencies from files",
                leave=False,
                disable=common.disable_tqdm()
        ):
            dictionary += d

    logger.info(f"Created dictionary with {len(dictionary)} entries")
    logger.info(f"Saving {args.top_k} most frequent entries from dictionary to {args.out_file}")

    with open(args.out_file, "w", encoding="utf8") as f:
        for w, freq in dictionary.most_common(n=args.top_k):
            f.write(f"{w}\t{freq}\n")


if __name__ == "__main__":
    create(parse_args())
