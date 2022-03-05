import argparse
import glob
import os
import zipfile

import torch

from gnn_lib.utils import io


def zip_experiment(args: argparse.Namespace) -> None:
    if not args.out_file.endswith(".zip"):
        args.out_file += ".zip"

    with zipfile.ZipFile(args.out_file, "w") as zip_file:
        checkpoint_best = os.path.join(args.experiment, "checkpoints", "checkpoint_best.pt")
        checkpoint = io.load_checkpoint(checkpoint_best)
        only_model_checkpoint = {"model_state_dict": checkpoint["model_state_dict"]}
        only_model_checkpoint_path = os.path.join(args.experiment, "checkpoints", "model_only_checkpoint_best.pt")
        torch.save(only_model_checkpoint, only_model_checkpoint_path)

        experiment_dir = os.path.join(args.experiment, "..")

        config_path = os.path.join(args.experiment, "cfg.pkl")

        # best checkpoint
        zip_file.write(
            os.path.join(only_model_checkpoint_path),
            os.path.relpath(only_model_checkpoint_path, experiment_dir)
        )

        # config
        zip_file.write(
            os.path.join(config_path),
            os.path.relpath(config_path, experiment_dir)
        )

        # delete only model checkpoint again
        os.remove(only_model_checkpoint_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=str, required=True)
    parser.add_argument("--out-file", type=str, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    zip_experiment(parse_args())
