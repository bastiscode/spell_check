import logging
import os
import re
from typing import Dict, List, Union, Optional, Set

from torch import nn

_LOG_FORMATTER = logging.Formatter("%(asctime)s [%(name)s] "
                                   "[%(levelname)s] [%(process)s] %(message)s")


def add_file_log(logger: logging.Logger, log_file: str) -> None:
    """

    Add file logging to an existing logger

    :param logger: logger
    :param log_file: path to logfile
    :return: logger with file logging handler
    """
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(_LOG_FORMATTER)
    logger.addHandler(file_handler)


def get_logger(name: str) -> logging.Logger:
    """

    Get a logger that writes to stderr and the specified log file.

    :param name: name of the logger (usually __name__)
    :return: logger
    """

    logger = logging.getLogger(name)
    logger.propagate = False
    logger.setLevel(logging.DEBUG)
    if not logger.handlers:
        stderr_handler = logging.StreamHandler()
        stderr_handler.setFormatter(_LOG_FORMATTER)
        logger.addHandler(stderr_handler)
    return logger


def _eta(dur: float, num_iter: int, total_iter: int) -> float:
    return (dur / max(num_iter, 1)) * total_iter - dur


def eta_minutes(num_minutes: float, num_iter: int, total_iter: int) -> str:
    _eta_minutes = _eta(num_minutes, num_iter, total_iter)
    return f"{num_minutes:.2f} minutes since start, " \
           f"eta: {_eta_minutes:.2f} minutes"


def eta_seconds(num_sec: float, num_iter: int, total_iter: int) -> str:
    _eta_seconds = _eta(num_sec, num_iter, total_iter)
    return f"{num_sec:.2f} seconds since start, " \
           f"eta: {_eta_seconds:.2f} seconds"


def last_n_k_path(path: str, n: int = 3, k: int = None) -> str:
    if k is None:
        return "/".join(path.split("/")[-n:])
    else:
        return "/".join(path.split("/")[-n:-k])


def natural_sort(unsorted: List[str], reverse: bool = False) -> List[str]:
    """

    Sort a list of strings naturally (like humans would sort them)
    Example:
        Natural  With normal sorting
        -------  --------
        1.txt    1.txt
        2.txt    10.txt
        10.txt   2.txt
        20.txt   20.txt

    :param unsorted: unsorted list of strings
    :param reverse: reverse order of list
    :return: naturally sorted list
    """

    def _convert(s: str) -> Union[str, int]:
        return int(s) if s.isdigit() else s.lower()

    def _alphanum_key(key: str) -> List[Union[int, str]]:
        return [_convert(c) for c in re.split(r"([0-9]+)", key)]

    return sorted(unsorted, key=_alphanum_key, reverse=reverse)


def get_num_parameters(module: nn.Module, unused_parameters: Optional[Set[str]] = None) -> Dict[str, int]:
    """

    Get the number of trainable, fixed and
    total parameters of a pytorch module.

    :param module: pytorch module
    :param optional unused_parameters: set of unused parameters to ignore
    :return: dict containing number of parameters
    """
    trainable = 0
    fixed = 0
    unused = 0
    for name, p in module.named_parameters():
        if unused_parameters is not None and name in unused_parameters:
            assert not p.requires_grad, f"expected unused parameter {name} to be fixed (requires_grad=False)"
            unused += p.numel()
        if p.requires_grad:
            trainable += p.numel()
        else:
            fixed += p.numel()
    return {"trainable": trainable,
            "fixed": fixed,
            "unused": unused,
            "total": trainable + fixed}


def disable_tqdm() -> bool:
    return os.getenv("NSC_DISABLE_TQDM", "false") == "true"


_MIXED_PRECISION = False


def set_mixed_precision(v: bool) -> None:
    global _MIXED_PRECISION
    _MIXED_PRECISION = v


def mixed_precision() -> bool:
    return _MIXED_PRECISION
