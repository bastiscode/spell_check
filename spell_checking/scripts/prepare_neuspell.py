import argparse
import json
import os
import multiprocessing as mp
from typing import Dict, Optional, Tuple

from tqdm import tqdm

from nsc.data.utils import is_valid_sequence
from nsc.utils import io, common

from spell_checking.utils import spacy_utils

logger = common.get_logger("PREPARE_NEUSPELL")


def get_sample(inputs: Tuple[str, str]) -> Optional[Dict[str, str]]:
    correct_line, corrupt_line = inputs
    correct_line = correct_line.strip()
    corrupt_line = corrupt_line.strip()
    correct_line, corrupt_line = spacy_utils.fix_sequences(correct_line, corrupt_line)
    if (
            not is_valid_sequence(correct_line, min_length=1)
            or not is_valid_sequence(corrupt_line, min_length=1)
    ):
        return None
    else:
        return {"sequence": corrupt_line, "target_sequence": correct_line}


def prepare_neuspell(args: argparse.Namespace) -> None:
    logger.info(f"Preparing neuspell from {args.in_dir}, saving to {args.out_dir}")

    if args.out_dir:
        os.makedirs(args.out_dir, exist_ok=True)

    gt_file = os.path.join(args.in_dir, "train.1blm")
    noise_files = sorted(io.glob_safe(os.path.join(args.in_dir, "train.1blm.noise*")))

    for noise_file in noise_files:
        samples = []
        out_file = os.path.join(args.out_dir, f"{os.path.basename(noise_file)}.jsonl")
        if os.path.exists(out_file):
            logger.info(f"File at {out_file} already exists, skipping.")
            continue
        with open(gt_file, "r", encoding="utf8") as gtf, \
                open(noise_file, "r", encoding="utf8") as nf, \
                mp.Pool(processes=int(os.getenv("NSC_NUM_PROCESSES", len(os.sched_getaffinity(0))))) as pool:
            for sample in tqdm(
                    pool.imap_unordered(get_sample, list(zip(gtf, nf))),
                    f"Processing {noise_file}",
                    total=io.line_count(gt_file)
            ):
                if sample is not None:
                    samples.append(sample)

        with open(out_file, "w", encoding="utf8") as of:
            for sample in samples:
                of.write(json.dumps(sample) + "\n")

        logger.info(f"Got {len(samples)} samples from neuspell file {noise_file}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-dir", type=str, required=True)
    parser.add_argument("--out-dir", type=str, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    prepare_neuspell(parse_args())
