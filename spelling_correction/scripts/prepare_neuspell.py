import argparse
import json
import os

from tqdm import tqdm

from gnn_lib.data.utils import is_valid_sequence
from gnn_lib.utils import io, common

from spelling_correction.utils import spacy_utils

logger = common.get_logger("PREPARE_NEUSPELL")


def prepare_neuspell(args: argparse.Namespace) -> None:
    out_file = os.path.join(args.out_dir, "train_1blm.jsonl")

    logger.info(f"Preparing neuspell from {args.in_dir}, saving to {args.out_dir}")

    gt_file = os.path.join(args.in_dir, "train.1blm")

    noise_files = io.glob_safe(os.path.join(args.in_dir, "train.1blm.noise*"))
    # exclude file with random noise
    noise_files = [f for f in noise_files if not f.endswith("noise.random")]

    samples = []
    for noise_file in noise_files:
        with open(gt_file, "r", encoding="utf8") as gtf, \
                open(noise_file, "r", encoding="utf8") as nf:
            for correct_line, corrupt_line in tqdm(
                    zip(gtf, nf), f"Processing {noise_file}", total=io.line_count(gt_file)
            ):
                correct_line = correct_line.strip()
                corrupt_line = corrupt_line.strip()
                correct_line, corrupt_line = neuspell.fix_sequences(correct_line, corrupt_line)
                if (
                        not is_valid_sequence(correct_line, min_length=1)
                        or not is_valid_sequence(corrupt_line, min_length=1)
                ):
                    continue
                sample = {"sequence": corrupt_line, "target_sequence": correct_line}
                samples.append(sample)

    with open(out_file, "w", encoding="utf8") as of:
        for sample in samples:
            of.write(json.dumps(sample) + "\n")

    logger.info(f"Got {len(samples)} samples from neuspell data files {noise_files}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-dir", type=str, required=True)
    parser.add_argument("--out-dir", type=str, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    prepare_neuspell(parse_args())
