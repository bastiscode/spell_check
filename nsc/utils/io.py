import collections
import glob
import os
from typing import Any, Dict, List, Optional, OrderedDict

import torch
from torch import nn
from torch import optim

from nsc.utils import common


def glob_safe(pattern: str, error_on_empty: bool = True) -> List[str]:
    """
    Safe version of glob.glob in the sense that it errors when
    no files are found with the glob pattern.

    :param pattern: glob pattern
    :param error_on_empty: whether to throw an error when no files are found
    :return: files matched by the pattern
    """
    files = glob.glob(pattern.strip())
    if len(files) == 0 and error_on_empty:
        raise RuntimeError(f"Found no files using glob pattern {pattern}")
    return files


def line_count(filepath: str) -> int:
    """

    Count the number of lines in a file

    :param filepath: path to the file
    :return: number of lines
    """
    i = 0
    with open(filepath, "r") as f:
        for _ in f:
            i += 1
    return i


def save_checkpoint(
        checkpoint_path: str,
        model: nn.Module,
        step: int,
        val_loss: float,
        optimizer: Optional[optim.Optimizer] = None,
        lr_scheduler: Optional[Any] = None
) -> None:
    """
    Saves a checkpoint to a directory.
    :param checkpoint_path: Filepath to save the checkpoint
    :param model: Pytorch module
    :param step: Global step
    :param val_loss: Validation loss achieved by this checkpoint
    :param optimizer: Pytorch optimizer
    :param lr_scheduler: Pytorch learning rate scheduler
    """
    state = {
        "model_state_dict": model.state_dict(),
        "step": step,
        "val_loss": val_loss,
        "optimizer_state_dict": None if optimizer is None
        else optimizer.state_dict(),
        "lr_scheduler_state_dict": None if lr_scheduler is None
        else lr_scheduler.state_dict()
    }
    torch.save(state, f=checkpoint_path)


def load_checkpoint(
        path: str,
        device: torch.device = torch.device("cpu")
) -> Dict[str, Any]:
    """
    Loads a checkpoint from disk. Maps checkpoint values to cpu by default.
    :param path: Path to the checkpoint file
    :param device: Optional pytorch device
    :return: Dictionary mapping from string keys to the checkpointed values
    """
    checkpoint = torch.load(path, map_location=device)
    return checkpoint


def filter_state_dict(
        state_dict: OrderedDict[str, torch.Tensor],
        prefix: str
) -> OrderedDict[str, torch.Tensor]:
    return collections.OrderedDict({k[len(prefix):]: v for k, v in state_dict.items() if k.startswith(prefix)})


def last_n_checkpoints(checkpoint_dir: str, n: int) -> List[str]:
    """
    Returns the paths to the newest n checkpoints in a checkpoint directory.
    :param checkpoint_dir: path to directory
    :param n: number of checkpoints
    :return: list of newest n checkpoints
    """
    assert os.path.isdir(checkpoint_dir), f"checkpoint_dir '{checkpoint_dir}' has to be a directory"
    checkpoints = glob_safe(os.path.join(checkpoint_dir, "checkpoint_*.pt"))
    # filter out last and best checkpoints
    checkpoints = [cp for cp in checkpoints if os.path.basename(cp) != "checkpoint_best.pt"]
    checkpoints = common.natural_sort(checkpoints)
    if n <= 0:
        return checkpoints
    else:
        return checkpoints[-n:]


def dictionary_from_file(file_path: str) -> Dict[str, int]:
    dictionary = {}
    with open(file_path, "r", encoding="utf8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if line == "":
                continue
            try:
                word, freq = line.split("\t")
            except Exception as e:
                s = line.split("\t")
                print(f"file: {file_path}, error on line: {i}: {s}, {line}: {e}")
                continue
            dictionary[word] = int(freq)
    return dictionary
