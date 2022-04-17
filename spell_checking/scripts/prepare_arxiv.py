import argparse
import json
import os
import multiprocessing as mp
from typing import Tuple

from tqdm import tqdm

from nsc.data.utils import clean_sequence, is_valid_sequence
from nsc.utils import io, common

logger = common.get_logger("PREPARE_ARXIV")


def arxiv_file_to_jsonl(inputs: Tuple[str, str, str]) -> None:
    no_error_file, with_error_file, out_file = inputs
    with open(no_error_file, "r", encoding="utf8") as nef, \
            open(with_error_file, "r", encoding="utf8") as wef, \
            open(out_file, "w", encoding="utf8") as of:
        for sequence, target_sequence in zip(wef, nef):
            sequence = clean_sequence(sequence)
            target_sequence = clean_sequence(target_sequence)
            if (
                    not is_valid_sequence(sequence, min_length=1)
                    or not is_valid_sequence(target_sequence, min_length=1)
                    or (len(sequence.split()) != len(target_sequence.split()))
            ):
                continue
            of.write(json.dumps({"sequence": sequence, "target_sequence": target_sequence}) + "\n")


def arxiv_to_jsonl(args: argparse.Namespace) -> None:
    os.makedirs(args.out_dir, exist_ok=True)

    logger.info(f"Preparing arxiv from {args.arxiv_no_errors_dir} and {args.arxiv_with_errors_dir}, "
                f"saving to {args.out_dir}")

    no_errors_files = io.glob_safe(os.path.join(args.arxiv_no_errors_dir, "*.txt"))
    with_errors_files = io.glob_safe(os.path.join(args.arxiv_with_errors_dir, "*.txt"))
    no_errors_files = common.natural_sort(no_errors_files)
    with_errors_files = common.natural_sort(with_errors_files)

    assert len(no_errors_files) == len(with_errors_files), f"expected same number of files with and without errors"
    assert all(
        int(nef.split("_")[-2]) == int(wef.split("_")[-2])
        for nef, wef in zip(no_errors_files, with_errors_files)
    ), f"expected the files to be aligned properly after sorting"

    tasks = []
    for no_error_file, with_error_file in zip(no_errors_files, with_errors_files):
        file_idx = no_error_file.split("_")[-2]
        out_file = os.path.join(args.out_dir, f"arxiv_{file_idx}.jsonl")
        tasks.append((no_error_file, with_error_file, out_file))

    num_processes = int(os.environ.get("NUM_PROCESSES", min(len(os.sched_getaffinity(0)), 8)))

    with mp.Pool(num_processes) as pool:
        for _ in tqdm(pool.imap_unordered(arxiv_file_to_jsonl, tasks), total=len(tasks)):
            pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arxiv-no-errors-dir", type=str, required=True)
    parser.add_argument("--arxiv-with-errors-dir", type=str, required=True)
    parser.add_argument("--out-dir", type=str, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    arxiv_to_jsonl(parse_args())
