import argparse
import os

from nsc.data.datasets import _decompress
from nsc.data.utils import open_lmdb
from nsc.utils import common


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lmdbs", type=str, nargs="+", required=True)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    logger = common.get_logger("LMDB_INSPECTION")

    sizes = []
    lengths = []
    for path in args.lmdbs:
        lmdb_name = os.path.basename(path)
        logger.info(f"Opening LMDB {lmdb_name}")
        env = open_lmdb(path)
        with env.begin(write=False) as txn:
            sizes.append(_decompress(txn.get(b"dataset_length")))
            lengths.extend(_decompress(txn.get(b"lengths")))
        logger.info(f"LMDB {lmdb_name} contains {sizes[-1]:,} samples")

    lengths = sorted(lengths)
    num_tokens = sum(lengths)
    logger.info(f"Number of samples in all LMDBs is {sum(sizes):,}")
    logger.info(f"Number of tokens in all LMDBs is {num_tokens:,}")
    logger.info(f"Average sample length is {num_tokens / len(lengths): .6f}")
    logger.info(f"Median sample length is {lengths[len(lengths) // 2]}")
    logger.info(f"Min and max sample lengths are {lengths[0]} and {lengths[-1]}")
