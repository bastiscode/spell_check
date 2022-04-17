import argparse
import os
from collections import Counter

from nsc.utils import common, io


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dictionaries", type=str, required=True, nargs="+")
    parser.add_argument("--out-file", type=str, required=True)
    parser.add_argument("--min-freq", type=int, default=0)
    return parser.parse_args()


def merge(args: argparse.Namespace) -> None:
    logger = common.get_logger("MERGE_DICTIONARIES")

    if os.path.exists(args.out_file):
        logger.info(f"Merged dictionary at {args.out_file} already exists")
        return

    if len(args.dictionaries) < 2:
        logger.info(f"Need at least two dictionaries to merge them, but got {len(args.dictionaries)}")
        return

    dicts = []
    for dict_file in args.dictionaries:
        d = io.dictionary_from_file(dict_file)
        dicts.append(d)

    merged = Counter(dicts[0])
    for d in dicts[1:]:
        merged += Counter(d)

    merged = Counter({w: freq for w, freq in merged.items() if freq >= args.min_freq})

    with open(os.path.join(args.out_file), "w", encoding="utf8") as of:
        for w, freq in merged.most_common():
            of.write(f"{w}\t{freq}\n")

    logger.info(
        f"Merged dictionaries {args.dictionaries} into new dictionary {args.out_file} "
        f"with {len(merged)} items with min_freq={args.min_freq}")


if __name__ == "__main__":
    merge(parse_args())
