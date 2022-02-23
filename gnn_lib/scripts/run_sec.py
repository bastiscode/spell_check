import argparse
import os
import time

import torch
from tabulate import tabulate
from torch.backends import cudnn
from tqdm import tqdm

import gnn_lib
import gnn_lib.data.utils
from gnn_lib.data import utils, index
from gnn_lib.modules.inference import inference_output_to_str
from gnn_lib.utils import common, io
from gnn_lib.tasks import sec_nmt, sec_words_nmt


def run(args: argparse.Namespace) -> None:
    logger = common.get_logger("SEC")

    device = torch.device(args.device)

    override_env_vars = {
        "GNN_LIB_DATA_DIR": args.data_dir,
        "GNN_LIB_CONFIG_DIR": args.config_dir
    }

    cfg, task, model = gnn_lib.load_experiment(args.experiment, device, override_env_vars)
    assert isinstance(task, sec_nmt.SECNMT) or isinstance(task, sec_words_nmt.SECWordsNMT)

    suffix = "" if args.suffix is None else f"_{args.suffix}"
    experiment_name = cfg.experiment_name + suffix

    file_name = experiment_name + ".txt"

    torch.manual_seed(args.seed)
    torch.set_num_threads(len(os.sched_getaffinity(0)))
    torch.backends.cudnn.benchmark = True
    torch.use_deterministic_algorithms(False)

    inference_kwargs = {
        "inference_mode": args.inference_mode,
        "beam_width": args.beam_width,
        "sample_top_k": args.sample_top_k,
        "best_first_top_k": args.best_first_top_k
    }
    if args.prefix_index is not None:
        prefix_index = index.PrefixIndex(args.prefix_index)
        inference_kwargs["prefix_index"] = prefix_index

    out_dirs = []
    if args.out_dirs is None:
        for in_file in args.in_files:
            path, _ = os.path.split(in_file)
            path_split = path.split("/")
            benchmark_dir = path_split[:-2]
            benchmark = path_split[-2:]
            out_dirs.append(os.path.join(*benchmark_dir, "results", *benchmark))
    else:
        out_dirs = args.out_dirs

    assert len(args.in_files) == len(out_dirs), "specify a output directory for each input file"

    if args.detection_files is not None:
        assert len(args.detection_files) == len(args.in_files), "specify a detection file for each input file"
        detection_files = args.detection_files
    else:
        detection_files = [None] * len(args.in_files)

    runtimes = []
    for in_file, out_dir, det_file in zip(args.in_files, out_dirs, detection_files):
        out_file = os.path.join(out_dir, file_name)
        if os.path.exists(out_file):
            raise RuntimeError(f"Out file {out_file} already exists, skipping")

        os.makedirs(out_dir, exist_ok=True)

        test_dataset, test_loader = utils.get_string_dataset_and_loader(
            in_file, args.sort_by_length, args.batch_size
        )
        logger.info(f"Dataset/input contains {len(test_dataset)} samples")

        detections = []
        if det_file is not None:
            with open(det_file, "r", encoding="utf8") as detf:
                for line in detf:
                    line = line.strip()
                    detections.append([int(d) for d in line.split()])
            assert len(detections) == len(test_dataset), \
                f"got different number of detections {len(detections)} than inputs {len(test_dataset)}"

        start = time.perf_counter()
        all_outputs = []
        for i, (batch, info) in tqdm(
                enumerate(test_loader),
                f"Running experiment {experiment_name} on {os.path.basename(in_file)}",
                total=len(test_loader),
                disable=not args.show_progress
        ):
            # inference_kwargs.update({
            #     "input_strings": batch
            # })
            if det_file is not None:
                inference_kwargs.update({
                    "detections": [detections[idx] for idx in info["indices"]]
                })
            outputs = task.inference(
                model,
                batch,
                **inference_kwargs
            )

            all_outputs.extend(outputs)

            if args.verbose:
                logger.info(f"Batch {i + 1}: \n{batch} ---> {outputs}\n")

        reordered_outputs = utils.reorder_data(all_outputs, test_dataset.indices)

        with open(out_file, "w", encoding="utf8") as of:
            for output in reordered_outputs:
                of.write(inference_output_to_str(output) + "\n")

        end = time.monotonic()
        runtime = end - start

        file_size = os.path.getsize(in_file) / 1024
        file_length = io.line_count(in_file)

        logger.info(f"Running {experiment_name} on {in_file} took {runtime:.2f}s")
        runtimes.append([in_file, runtime, file_length, file_size])

    runtimes_table = tabulate(runtimes,
                              headers=["Directory", "Runtime in seconds", "Number of samples", "File size in KB"],
                              tablefmt="pipe")

    total_samples = sum([r[2] for r in runtimes])
    total_file_size = sum([r[3] for r in runtimes])
    total_time = sum([r[1] for r in runtimes])

    aggregated_runtimes = [
        [experiment_name,
         total_time,
         total_samples / total_time,
         total_time / total_file_size]
    ]

    runtimes_aggregated_table = tabulate(aggregated_runtimes,
                                         headers=["Model", "Total runtime in seconds", "samples/s", "s/KB"],
                                         tablefmt="pipe")

    logger.info(f"\nModel: {experiment_name}, Batch size: {args.batch_size}, Sorted: {args.sort_by_length}\n"
                f"{runtimes_table}\n\n"
                f"{runtimes_aggregated_table}\n")

    if args.save_markdown_dir is not None:
        os.makedirs(args.save_markdown_dir, exist_ok=True)

        with open(os.path.join(args.save_markdown_dir,
                               f"{experiment_name}_{args.batch_size}"
                               f"{'_sorted_by_length' if args.sorted_by_length else ''}.md"),
                  "w",
                  encoding="utf8") as f:
            f.write(runtimes_table)
            f.write("\n\n")
            f.write(runtimes_aggregated_table)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment", type=str)
    parser.add_argument("--data-dir",
                        required=True,
                        help="Path to data dir")
    parser.add_argument("--config-dir",
                        required=True,
                        help="Path to config dir")
    parser.add_argument("--in-files", nargs="+", type=str, required=True)
    parser.add_argument("--detection-files", nargs="+", type=str, default=None)
    parser.add_argument("--out-dirs", nargs="+", type=str, default=None)
    parser.add_argument("--sort-by-length", action="store_true")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=22)
    parser.add_argument("--save-markdown-dir", type=str, default=None)
    parser.add_argument("--suffix", type=str, default=None)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--show-progress", action="store_true")

    # sec specific args
    parser.add_argument("--inference-mode", choices=["greedy", "sample", "best_first", "beam"], default="greedy")
    parser.add_argument("--beam-width", type=int, default=5)
    parser.add_argument("--best-first-top-k", type=int, default=5)
    parser.add_argument("--sample-top-k", type=int, default=5)
    parser.add_argument("--prefix-index", type=str, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
