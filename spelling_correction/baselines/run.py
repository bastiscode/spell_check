import argparse
import os
from typing import TextIO, List

import torch
from torch.backends import cudnn
from tqdm import tqdm

from gnn_lib.api.utils import get_string_dataset_and_loader, reorder_data
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
    parser.add_argument("--suffix", type=str, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--sort-by-length", action="store_true")
    parser.add_argument("--detections-file", type=str, default=None)
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
    torch.backends.cudnn.benchmark = True
    torch.use_deterministic_algorithms(False)

    kwargs = {}
    if args.sec_baseline:
        kwargs["sec_baseline"] = Baselines[args.sec_baseline]
    baseline = baselines.get_baseline(Baselines[args.baseline], args.seed, **kwargs)

    suffix = "" if args.suffix is None else f"_{args.suffix}"
    out_path = os.path.join(args.out_dir, f"baseline_{baseline.name}{suffix}.txt")
    if os.path.exists(out_path) and not args.overwrite:
        logger.info(f"Out file {out_path} already exists, skipping baseline {baseline.name} "
                    f"({args.baseline})")
        return

    dataset, loader = get_string_dataset_and_loader(args.in_file, args.sort_by_length, args.batch_size)

    detections = []
    if args.detections_file is not None:
        with open(args.detections_file, "r") as det_f:
            for line in det_f:
                detections.append([int(d) for d in line.strip().split()])

    os.makedirs(args.out_dir, exist_ok=True)
    all_outputs = []
    for batch, indices in tqdm(
            loader,
            total=len(loader),
            desc=f"Running baseline {baseline.name} ({args.baseline}) on "
                 f"{os.path.relpath(args.in_file, BENCHMARK_DIR)}"
    ):
        inference_kwargs = {}
        if args.detections_file is not None:
            inference_kwargs.update({
                "detections": [detections[idx] for idx in indices]
            })

        outputs = baseline.inference(batch, **inference_kwargs)
        all_outputs.extend(outputs)

    reordered_outputs = reorder_data(all_outputs, dataset.indices)

    with open(out_path, "w", encoding="utf8") as of:
        for output in reordered_outputs:
            of.write(inference.inference_output_to_str(output) + "\n")


if __name__ == "__main__":
    run(parse_args())
