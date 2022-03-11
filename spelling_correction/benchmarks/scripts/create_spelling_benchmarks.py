import argparse
import json
import os

import numpy as np
import omegaconf
from tqdm import tqdm

from gnn_lib.data import preprocessing

from spelling_correction import MISSPELLINGS_DIR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    in_file_group = parser.add_mutually_exclusive_group()
    in_file_group.add_argument("--in-files", type=str, nargs="+")
    in_file_group.add_argument("--in-file", type=str)
    parser.add_argument("--max-sequences", type=int, required=True)
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--benchmark-name", type=str, required=True)
    parser.add_argument("--misspelling-type", choices=["artificial", "realistic"], required=True)
    parser.add_argument("--misspelling-split", choices=["train", "dev", "test"], required=True)
    parser.add_argument("--output-type", choices=["sed_sequence", "sed_words", "sec"], required=True)
    parser.add_argument("--seed", type=int, default=22)
    return parser.parse_args()


def create(args: argparse.Namespace) -> None:
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
        print("Got no input files")
        return

    all_sequences = []
    for file in files:
        with open(file, "r", encoding="utf8") as inf:
            for line in inf:
                line = line.strip()
                if file.endswith(".jsonl"):
                    sequence = json.loads(line)["sequence"]
                elif file.endswith(".txt"):
                    sequence = line
                else:
                    raise ValueError(f"Unknown file ending of in file, must be either .jsonl or .txt")
                all_sequences.append(sequence)

    print(f"Extracted {len(all_sequences)} sequences from {len(files)} files")

    rand = np.random.default_rng(args.seed)
    indices = rand.permutation(len(all_sequences))

    out_dir = os.path.join(args.out_dir, args.output_type, args.benchmark_name, args.misspelling_type)
    correct_out_path = os.path.join(out_dir, "correct.txt")
    corrupt_out_path = os.path.join(out_dir, "corrupt.txt")
    if os.path.exists(correct_out_path) and os.path.exists(corrupt_out_path):
        print(f"Benchmark {args.output_type} {args.benchmark_name} {args.misspelling_type} already exists")
        return

    os.makedirs(out_dir, exist_ok=True)
    correct_out_file = open(correct_out_path, "w", encoding="utf8")
    corrupt_out_file = open(corrupt_out_path, "w", encoding="utf8")

    if args.misspelling_type == "artificial":
        noise_config = omegaconf.DictConfig({
            "type": "ARTIFICIAL",
            "edit_token_p": 0.2,
            "num_edits_p": 0.8
        })
    else:
        noise_config = omegaconf.DictConfig({
            "type": "REALISTIC",
            "edit_token_p": 0.2,
            "word_misspellings_file": os.path.join(MISSPELLINGS_DIR, f"{args.misspelling_split}_misspellings.json")
        })

    spelling_noise = noise.get_preprocessing_from_config(noise_config, args.seed)
    print(f"Using noise config: {noise_config}")

    for idx in tqdm(indices[:args.max_sequences],
                    desc=f"Creating benchmark {args.benchmark_name} ({args.output_type}, {args.misspelling_type}, "
                         f"{args.misspelling_split})"):
        sequence = all_sequences[idx]
        sequence, corrupted = spelling_noise.apply(sequence)

        if args.output_type == "sed_sequence":
            # make even 50/50 split here
            if rand.random() < 0.5:
                correct_out_file.write("0")
                corrupted = sequence
            else:
                correct_out_file.write(str(int(corrupted != sequence)))
        elif args.output_type == "sed_words":
            corrupted_words = corrupted.split()
            correct_words = sequence.split()
            assert len(corrupted_words) == len(correct_words)
            correct_out_file.write(" ".join(str(int(c != s)) for c, s in zip(corrupted_words, correct_words)))
        elif args.output_type == "sec":
            correct_out_file.write(sequence)
        else:
            raise ValueError(f"Unknown output type {args.output_type}")

        correct_out_file.write("\n")

        corrupt_out_file.write(corrupted)
        corrupt_out_file.write("\n")

    corrupt_out_file.close()
    correct_out_file.close()


if __name__ == "__main__":
    create(parse_args())
