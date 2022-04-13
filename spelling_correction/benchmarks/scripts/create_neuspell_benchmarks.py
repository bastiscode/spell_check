import argparse
import os

import numpy as np
from tqdm import tqdm

from gnn_lib.utils import io

from spelling_correction.utils import spacy_utils


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--neuspell-dir", type=str, required=True)
    parser.add_argument("--benchmark-dir", type=str, required=True)
    parser.add_argument("--benchmarks", type=str, nargs="+", required=True)
    parser.add_argument("--seed", type=int, default=22)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    rand = np.random.default_rng(args.seed)

    benchmarks = {
        "bea4k.noise": ("bea4k", "bea4k"),
        "bea322.noise": ("bea322", "bea322"),
        "bea4660.noise": ("bea4660", "bea4660"),
        "bea60k.noise": ("bea60k", "bea60k"),
        "jfleg.noise": ("jfleg", "jfleg"),
        "1blm.noise.prob": ("1blm", "1blm_prob"),
        "1blm.noise.word": ("1blm", "1blm_word"),
        "1blm.noise.random": ("1blm", "1blm_random")
    }

    for output_type in ["sed_sequence", "sed_words", "sec"]:
        for noise, b in tqdm(benchmarks.items(), desc=f"Creating {output_type} benchmarks"):
            if b[1] not in args.benchmarks:
                continue

            benchmark_dir = os.path.join(args.benchmark_dir, output_type, "neuspell", b[1])

            os.makedirs(benchmark_dir, exist_ok=True)

            correct_src = os.path.join(args.neuspell_dir, f"test.{b[0]}")
            corrupt_src = os.path.join(args.neuspell_dir, f"test.{noise}")

            correct_b = os.path.join(benchmark_dir, "correct.txt")
            corrupt_b = os.path.join(benchmark_dir, "corrupt.txt")
            correct_seq_b = os.path.join(benchmark_dir, "correct.sequences.txt")
            if output_type != "sec":
                correct_seq_b_file = open(correct_seq_b, "w", encoding="utf8")
            else:
                correct_seq_b_file = None

            if os.path.exists(corrupt_b) and os.path.exists(correct_b):
                continue

            with open(correct_src, "r", encoding="utf8") as correct_inf, \
                    open(correct_b, "w", encoding="utf8") as correct_of, \
                    open(corrupt_src, "r", encoding="utf8") as corrupt_inf, \
                    open(corrupt_b, "w", encoding="utf8") as corrupt_of:
                for correct_line, corrupt_line in tqdm(
                        zip(correct_inf, corrupt_inf),
                        leave=False,
                        total=io.line_count(correct_src),
                        desc=f"Creating {output_type} {b[1]} benchmark"
                ):
                    correct_line = correct_line.strip()
                    corrupt_line = corrupt_line.strip()
                    if corrupt_line == "" or correct_line == "":
                        continue

                    correct_line, corrupt_line = spacy_utils.fix_sequences(correct_line, corrupt_line)

                    if output_type == "sed_sequence":
                        correct_seq_b_file.write(correct_line + "\n")

                        if rand.random() < 0.5:
                            corrupt_line = correct_line
                            correct_of.write("0")
                        else:
                            correct_of.write(str(int(correct_line != corrupt_line)))

                    elif output_type == "sed_words":
                        correct_seq_b_file.write(correct_line + "\n")

                        correct_words = correct_line.split()
                        corrupt_words = corrupt_line.split()
                        correct_of.write(
                            " ".join(str(int(correct != corrupt))
                                     for correct, corrupt in zip(correct_words, corrupt_words))
                        )

                    elif output_type == "sec":
                        correct_of.write(correct_line)

                    else:
                        raise ValueError(f"Unknown output type {output_type}")

                    correct_of.write("\n")

                    corrupt_of.write(corrupt_line)
                    corrupt_of.write("\n")

            if correct_seq_b_file:
                correct_seq_b_file.close()
