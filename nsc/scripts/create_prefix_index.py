import argparse
import os
import time

from nsc.data import index
from nsc.utils import common, io


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dictionary", type=str, required=True)
    parser.add_argument("--out-file", type=str, required=True)
    return parser.parse_args()


def create(args: argparse.Namespace) -> None:
    logger = common.get_logger("CREATE_PREFIX_INDEX")

    if os.path.exists(args.out_file):
        logger.info(f"Prefix index {args.out_file} already exists")
    else:
        dictionary = io.dictionary_from_file(args.dictionary)
        index.PrefixIndex.create(dictionary, args.out_file)

    prefix_index = index.PrefixIndex(args.out_file)
    iterations = 1000
    for prefix in ["th", "Hou", "play"]:
        start = time.perf_counter()
        for i in range(iterations):
            results = prefix_index.retrieve(prefix)
            if i == 0:
                logger.info(f"Retrieving from prefix index: {prefix} -> {results}")
        end = time.perf_counter()
        logger.info(f"Retrieving with prefix {prefix} took on avg {(end - start) * 1000 / 1000:.2f}ms")


if __name__ == "__main__":
    create(parse_args())
