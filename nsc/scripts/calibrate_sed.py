import argparse
import os
import pickle

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import optim, nn
from torch.nn import functional as F
from tqdm import tqdm

import nsc
from nsc.api.utils import load_experiment, get_string_dataset_and_loader
from nsc.data import utils
from nsc.utils import common


def calc_bins(preds, labels):
    # Assign each prediction to a bin
    num_bins = 10
    bins = np.linspace(0.1, 1, num_bins)
    binned = np.digitize(preds, bins)

    # Save the accuracy, confidence and size of each bin
    bin_accs = np.zeros(num_bins)
    bin_confs = np.zeros(num_bins)
    bin_sizes = np.zeros(num_bins)

    for bin in range(num_bins):
        bin_sizes[bin] = len(preds[binned == bin])
        if bin_sizes[bin] > 0:
            bin_accs[bin] = (labels[binned == bin]).sum() / bin_sizes[bin]
            bin_confs[bin] = (preds[binned == bin]).sum() / bin_sizes[bin]

    return bins, binned, bin_accs, bin_confs, bin_sizes


def get_metrics(preds, labels):
    ECE = 0
    MCE = 0
    bins, _, bin_accs, bin_confs, bin_sizes = calc_bins(preds, labels)

    for i in range(len(bins)):
        abs_conf_dif = abs(bin_accs[i] - bin_confs[i])
        ECE += (bin_sizes[i] / sum(bin_sizes)) * abs_conf_dif
        MCE = max(MCE, abs_conf_dif)

    return ECE, MCE


def draw_reliability_graph(preds, labels, file_path: str):
    ECE, MCE = get_metrics(preds, labels)
    bins, _, bin_accs, _, _ = calc_bins(preds, labels)

    fig = plt.figure(figsize=(8, 8))
    ax = fig.gca()

    # x/y limits
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1)

    # x/y labels
    plt.xlabel('Confidence')
    plt.ylabel('Accuracy')

    # Create grid
    ax.set_axisbelow(True)
    ax.grid(color='gray', linestyle='dashed')

    # Error bars
    plt.bar(bins, bins, width=0.1, alpha=0.3, edgecolor='black', color='r', hatch='\\')

    # Draw bars and identity line
    plt.bar(bins, bin_accs, width=0.1, alpha=1, edgecolor='black', color='b')
    plt.plot([0, 1], [0, 1], '--', color='gray', linewidth=2)

    # Equally spaced axes
    plt.gca().set_aspect('equal', adjustable='box')

    # ECE and MCE legend
    ECE_patch = mpatches.Patch(color='green', label='ECE = {:.2f}%'.format(ECE * 100))
    MCE_patch = mpatches.Patch(color='red', label='MCE = {:.2f}%'.format(MCE * 100))
    plt.legend(handles=[ECE_patch, MCE_patch])

    # plt.show()

    plt.savefig(file_path, bbox_inches='tight')


def temp_scaling(logits: torch.Tensor, temperature: torch.Tensor) -> torch.Tensor:
    return torch.div(logits, temperature)


def calibrate(args: argparse.Namespace) -> None:
    logger = common.get_logger("SED_CALIBRATION")

    device = torch.device(args.device)

    cfg, task, model = load_experiment(
        args.experiment, args.device, {"NSC_DATA_DIR": args.data_dir,
                                       "NSC_CONFIG_DIR": args.config_dir}
    )

    dataset, loader = get_string_dataset_and_loader(args.in_file, False, args.batch_size)
    label_dataset, label_loader = get_string_dataset_and_loader(args.label_file, False, 1)

    logger.info(f"Dataset/input contains {len(dataset)} samples")

    all_outputs = []
    all_preds = []
    for i, (batch, _) in tqdm(
            enumerate(loader),
            f"Running experiment {cfg.experiment_name} ({os.path.dirname(args.experiment)})",
            total=len(loader),
            disable=not args.show_progress
    ):
        outputs = task.inference(
            model,
            batch,
            **{
                "return_logits": True
            }
        )
        if isinstance(outputs[0][0], list):
            outputs = [o for output in outputs for o in output]

        outputs = torch.tensor(outputs, dtype=torch.float).reshape(-1, 2)
        all_preds.append(torch.softmax(outputs, -1))
        all_outputs.append(outputs)

    all_outputs = torch.flatten(torch.cat(all_outputs))
    all_preds = torch.flatten(torch.cat(all_preds))

    all_labels = []
    for line, _ in label_loader:
        line = line[0]
        for label in line.split():
            all_labels.append(int(label))

    all_labels = torch.tensor(all_labels, dtype=torch.long, device=device)
    all_one_hot_labels = torch.flatten(F.one_hot(all_labels, 2))

    assert len(all_outputs) == len(all_one_hot_labels) and len(all_preds) == len(all_one_hot_labels)

    draw_reliability_graph(all_preds, all_one_hot_labels, f"{cfg.experiment_name}_uncalibrated.png")

    temperature = nn.Parameter(torch.ones(1, device=device))
    optimizer = optim.LBFGS([temperature], lr=0.001, max_iter=10_000, line_search_fn="strong_wolfe")

    all_outputs = all_outputs.to(device)
    all_labels = all_labels.to(device)

    def _closure():
        loss = F.cross_entropy(temp_scaling(all_outputs.view(-1, 2), temperature), all_labels)
        loss.backward()
        return loss

    optimizer.step(_closure)

    logger.info(f"Optimal temperature: {temperature.item():.4f}")
    with open(args.out_file, "wb") as of:
        pickle.dump(temperature.item(), of)

    calibrated_preds = torch.flatten(torch.cat([
        torch.softmax(temp_scaling(o, temperature.item()), -1) for o in all_outputs.cpu().view(-1, 2)
    ]))
    draw_reliability_graph(calibrated_preds, all_one_hot_labels, f"{cfg.experiment_name}_calibrated.png")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment", type=str)
    parser.add_argument("--data-dir",
                        required=True,
                        help="Path to data dir")
    parser.add_argument("--config-dir",
                        required=True,
                        help="Path to config dir")
    parser.add_argument("--in-file", type=str, required=True)
    parser.add_argument("--label-file", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--show-progress", action="store_true")

    # calibration specific args
    parser.add_argument("--out-file", type=str, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    calibrate(parse_args())
