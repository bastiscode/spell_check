import argparse
import os
import pickle
import time
from typing import BinaryIO

import torch
from omegaconf import OmegaConf
from torch.backends import cudnn
from tqdm import tqdm

import gnn_lib
import gnn_lib.data.utils
from gnn_lib.data import utils
from gnn_lib.modules.inference import inference_output_to_str
from gnn_lib.utils import common, config


def test(args: argparse.Namespace) -> None:
    logger = common.get_logger("TEST")

    with open(os.path.join(args.experiment, "cfg.pkl"), "rb") as cfg_file:  # type: BinaryIO
        train_cfg, env_vars = pickle.load(cfg_file)

    suffix = os.getenv("GNN_LIB_EXP_SUFFIX", "")
    if suffix != "":
        suffix = "_" + suffix

    env_vars.update({
        "GNN_LIB_DATA_DIR": args.data_dir,
        "GNN_LIB_CONFIG_DIR": args.config_dir
    })
    config.set_gnn_lib_env_vars(env_vars)

    out_file = os.path.join(
        args.out_path, f"{train_cfg.experiment_name}{suffix}.txt"
    ) if not args.out_path.endswith(".txt") else args.out_path

    if os.path.exists(out_file) and not args.overwrite:
        logger.info(f"Out file {out_file} already exists, skipping this test run")
        return

    os.makedirs(os.path.dirname(out_file), exist_ok=True)

    logger.info(
        f"Successfully loaded train config:\n"
        f"{OmegaConf.to_yaml(train_cfg, resolve=True)}"
    )

    torch.manual_seed(train_cfg.seed)
    torch.set_num_threads(8)
    torch.backends.cudnn.benchmark = True
    torch.use_deterministic_algorithms(False)

    test_dataset, test_loader = utils.get_string_dataset_and_loader(args.in_file, args.sort_by_length, args.batch_size)
    logger.info(f"Test dataset/input contains {len(test_dataset)} samples")

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    task = gnn_lib.get_task(
        variant_cfg=train_cfg.variant,
        checkpoint_dir=os.path.join(args.experiment, "checkpoints"),
        seed=train_cfg.seed
    )
    # IMPORTANT: the sample graph needs to contain
    # all possible node and edge types
    sample_g, _ = task.generate_sample_inputs(2)
    model = task.get_model(
        sample_g=sample_g,
        cfg=train_cfg.model,
        device=device
    )
    task.load_best(model=model)

    logger.info(f"Model has "
                f"{common.get_num_parameters(model)['total']:,} "
                f"parameters")

    if args.runtime_file is not None:
        os.makedirs(os.path.dirname(args.runtime_file), exist_ok=True)
        runtime_file = open(args.runtime_file, "a", encoding="utf8")
    else:
        runtime_file = None

    inference_kwargs = {}

    start = time.monotonic()

    all_outputs = []
    for i, (batch, _) in tqdm(
            enumerate(test_loader),
            f"Running experiment {train_cfg.experiment_name} ({os.path.basename(args.experiment)})",
            total=len(test_loader)
    ):
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
    logger.info(f"Running {train_cfg.experiment_name} on {args.in_file} took {runtime:.2f}s")

    if runtime_file is not None:
        file_size_kb = os.path.getsize(args.in_file) / 1024
        seconds_per_kb = runtime / file_size_kb
        sequences_per_second = len(test_dataset) / runtime
        runtime_file.write(
            f"{train_cfg.experiment_name}\t"
            f"{args.in_file}\t"
            f"{args.batch_size}\t"
            f"{args.sort_by_length}\t"
            f"{file_size_kb}\t"
            f"{len(test_dataset)}\t"
            f"{runtime}\t"
            f"{seconds_per_kb}\t"
            f"{sequences_per_second}\t"
            f"{common.get_device_info(device)}\t"
            f"{inference_kwargs}"
            f"\n"
        )
        runtime_file.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment", type=str)
    parser.add_argument("--data-dir",
                        required=True,
                        help="Path to data dir")
    parser.add_argument("--config-dir",
                        required=True,
                        help="Path to config dir")
    parser.add_argument("--in-file",
                        type=str, required=True)
    parser.add_argument("--out-path",
                        type=str, required=True)
    parser.add_argument("--sort-by-length", type=bool, default=1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--runtime-file", type=str, default=None)
    parser.add_argument("--overwrite", type=bool, default=0)
    parser.add_argument("--cpu", type=bool, default=0)
    parser.add_argument("--verbose", type=bool, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    test(parse_args())
