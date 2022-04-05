import argparse
import os
import random
import time

import numpy as np

from gnn_lib.data.index import NNIndex
from gnn_lib.utils import common
from gnn_lib.data import utils


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    in_file_group = parser.add_mutually_exclusive_group()
    in_file_group.add_argument("--in-files", type=str, nargs="+")
    in_file_group.add_argument("--in-file", type=str, nargs="+")
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--context-length", type=int, required=True)
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--dictionary-file", type=str, default=None)
    parser.add_argument("--dist", choices=["euclidean", "cosine", "edit_distance", "norm_edit_distance"], required=True)
    parser.add_argument("--vectorizer", choices=["ft", "bert", "string", "custom"], required=True)
    parser.add_argument("--vectorizer-path", type=str, default=None)
    parser.add_argument("--m", type=int, default=16)
    parser.add_argument("--ef-construction", type=int, default=200)
    parser.add_argument("--post", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--seed", type=int, default=22)
    parser.add_argument("--interactive", action="store_true")
    return parser.parse_args()


def create(args: argparse.Namespace) -> None:
    logger = common.get_logger("CREATE_SPELL_CHECK_INDEX")

    if not os.path.exists(os.path.abspath(args.out_dir)) or args.overwrite:
        if args.in_file:
            files = []
            for file in args.in_file:
                with open(file, "r", encoding="utf8") as f:
                    for line in f:
                        line = line.strip()
                        if line == "":
                            continue
                        files.append(os.path.join(os.path.dirname(file), line))
        else:
            files = args.in_files

        rand = random.Random(args.seed)
        rand.shuffle(files)
        if args.max_files is not None:
            files = files[:args.max_files]

        NNIndex.create(
            files,
            args.out_dir,
            args.context_length,
            args.vectorizer,
            args.dist,
            dictionary_file=args.dictionary_file,
            ef_construction=args.ef_construction,
            m=args.m,
            post=args.post,
            vectorizer_path=args.vectorizer_path
        )

    index = NNIndex(args.out_dir)
    index_context_length = 2 * index.params["context_length"] + 1

    num_iterations = 1000
    neighbours = [3, 5, 10, 25, 50, 100]
    context_str = " ".join("this" for _ in range(index.params["context_length"]))
    for n_neighbors in neighbours:
        times = []
        for i in range(num_iterations):
            start = time.perf_counter()
            _ = index.retrieve((context_str, "this", context_str), n_neighbors)
            end = time.perf_counter()

            times.append((end - start) * 1000)

        logger.info(f"Average query time over {num_iterations} iterations for {n_neighbors} neighbours "
                    f"was {np.mean(times):.2f}ms (+-{np.std(times):.2f})")

    if args.interactive:
        while True:
            sequence = input("Input a sequence you want to retrieve from the index: ")
            if sequence == "quit":
                break

            sequence = utils.clean_sequence(sequence)
            contexts = index.prepare_sequence(sequence, index.params["context_length"])
            if len(contexts) != index_context_length:
                logger.warning(f"Number of context from input {sequence} "
                               f"does not match the index context length of {index_context_length}")
                continue

            for context in contexts:
                neighbors = index.retrieve(context, 5)
                logger.info(f"Neighbors for context: {context}\n{neighbors}")


if __name__ == "__main__":
    create(parse_args())
