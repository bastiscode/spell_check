import os

import numpy as np
from tqdm import tqdm

from gnn_lib.utils import io
from spelling_correction import DATA_DIR, BENCHMARK_DIR
from spelling_correction.utils import neuspell


if __name__ == "__main__":
    NEUSPELL_SRC_DIR = os.path.join(DATA_DIR, "raw", "neuspell", "traintest")

    rand = np.random.default_rng(22)

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
            benchmark_dir = os.path.join(BENCHMARK_DIR, output_type, "neuspell", b[1])

            os.makedirs(benchmark_dir, exist_ok=True)

            correct_src = os.path.join(NEUSPELL_SRC_DIR, f"test.{b[0]}")
            corrupt_src = os.path.join(NEUSPELL_SRC_DIR, f"test.{noise}")

            correct_b = os.path.join(benchmark_dir, "correct.txt")
            corrupt_b = os.path.join(benchmark_dir, "corrupt.txt")

            if os.path.exists(corrupt_b) and os.path.exists(correct_b):
                continue

            with open(correct_src, "r", encoding="utf8") as correct_inf, \
                    open(correct_b, "w", encoding="utf8") as correct_of, \
                    open(corrupt_src, "r", encoding="utf8") as corrupt_inf, \
                    open(corrupt_b, "w", encoding="utf8") as corrupt_of:
                for correct_line, corrupt_line in tqdm(zip(correct_inf, corrupt_inf),
                                                       leave=False,
                                                       total=io.line_count(correct_src),
                                                       desc=f"Creating {output_type} {b[1]} benchmark"):
                    correct_line = correct_line.strip()
                    corrupt_line = corrupt_line.strip()
                    if corrupt_line == "" or corrupt_line == "" or len(corrupt_line) > 512 or len(correct_line) > 512:
                        continue
                    correct_line, corrupt_line = neuspell.clean_sequences(correct_line, corrupt_line)

                    if output_type == "sed_sequence":
                        if rand.random() < 0.5:
                            corrupt_line = correct_line
                            correct_of.write("0")
                        else:
                            correct_of.write(str(int(correct_line != corrupt_line)))
                    elif output_type == "sed_words":
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
