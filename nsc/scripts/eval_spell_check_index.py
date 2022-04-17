import argparse
import json
import time

from tqdm import tqdm
import numpy as np

from nsc.data.index import NNIndex
from nsc.utils import common


def evaluate(args: argparse.Namespace) -> None:
    logger = common.get_logger("EVALUATE_VECTORIZER")

    with open(args.misspellings_file, "r", encoding="utf8") as f:
        misspellings = json.load(f)

    index = NNIndex(args.index_dir, ef_search=args.ef_search)

    num_neighbors = 100

    num_elements = len(misspellings)
    if args.limit:
        num_elements = min(num_elements, args.limit)

    total = 0
    p_at_k = np.zeros(num_neighbors)
    total_time = 0
    for i, (correct, misspelled) in tqdm(enumerate(misspellings.items()), total=num_elements):
        start = time.perf_counter()
        results = index.batch_retrieve([("", m, "") for m in misspelled], num_neighbors)
        end = time.perf_counter()
        total_time += end - start
        total += len(misspelled)
        neighbor_sets = [neighbors.words for neighbors in results]
        for neighbor_set in neighbor_sets:
            for j, neighbor in enumerate(neighbor_set):
                if correct == neighbor:
                    p_at_k[j:] += 1
                    break
        if (i + 1) >= num_elements:
            break

    for i, p in enumerate(p_at_k):
        logger.info(f"P@{i + 1}: {p * 100 / total:.2f}")

    logger.info(f"Retrieving {num_neighbors} neighbors took on average {1000 * total_time / num_elements:.2f}ms")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--misspellings-file", type=str, required=True)
    parser.add_argument("--index-dir", type=str, required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--ef-search", type=int, default=200)
    return parser.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
