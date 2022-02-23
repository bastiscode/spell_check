import argparse
import os
from typing import TextIO, List

import torch
from torch.backends import cudnn
from torch.utils.data import DataLoader
from tqdm import tqdm

import gnn_lib.data.utils
from gnn_lib.data.utils import StringDataset
from gnn_lib.modules import inference
from gnn_lib.utils import common
from spelling_correction import baselines, BENCHMARK_DIR
from spelling_correction.baselines import Baselines


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-file", type=str, required=True)
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=22)
    parser.add_argument("--baseline", choices=[baseline.name for baseline in Baselines], required=True)
    parser.add_argument("--sec-baseline",
                        choices=[baseline.name for baseline in Baselines
                                 if baseline.name.startswith("SEC")] + [""],
                        default="")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--sort-by-length", action="store_false")
    return parser.parse_args()


def predict_and_write(sequences: List[str], baseline: baselines.Baseline, out_file: TextIO) -> None:
    predictions = baseline.inference(sequences)
    for prediction in predictions:
        out_file.write(str(prediction))
        out_file.write("\n")
    sequences.clear()


def run(args: argparse.Namespace) -> None:
    logger = common.get_logger("BASELINE")

    torch.manual_seed(args.seed)
    torch.set_num_threads(8)
    torch.backends.cudnn.benchmark = True
    torch.use_deterministic_algorithms(False)

    kwargs = {}
    if args.sec_baseline:
        kwargs["sec_baseline"] = Baselines[args.sec_baseline]
    baseline = baselines.get_baseline(Baselines[args.baseline], args.seed, **kwargs)

    out_path = os.path.join(args.out_dir, f"baseline_{baseline.name}.txt")
    if os.path.exists(out_path) and not args.overwrite:
        logger.info(f"Out file {out_path} already exists, skipping baseline {baseline.name} "
                    f"({args.baseline})")
        return

    text_data: List[str] = []
    with open(args.in_file, "r", encoding="utf8") as in_file:  # type: TextIO
        for line in in_file:
            line = gnn_lib.data.utils.clean_sequence(line)
            if line == "":
                continue
            text_data.append(line)

    dataset = StringDataset(text_data, sort_by_length=args.sort_by_length)
    loader = DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        collate_fn=dataset.collate_fn,
        pin_memory=True
    )

    os.makedirs(args.out_dir, exist_ok=True)
    all_outputs = []
    for batch, _ in tqdm(loader,
                         total=len(loader),
                         desc=f"Running baseline {baseline.name} ({args.baseline}) on "
                              f"{os.path.relpath(args.in_file, BENCHMARK_DIR)}"):
        outputs = baseline.inference(batch)
        all_outputs.extend(outputs)

    unordered_outputs: List[str] = [""] * len(all_outputs)
    for output, idx in zip(all_outputs, dataset.indices):
        unordered_outputs[idx] = output

    with open(out_path, "w", encoding="utf8") as of:
        for output in unordered_outputs:
            of.write(inference.inference_output_to_str(output) + "\n")


if __name__ == "__main__":
    run(parse_args())
