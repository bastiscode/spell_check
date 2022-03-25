import collections
import io
import logging
import os
import pickle
import platform
import re
import shutil
import zipfile
from typing import Optional, Union, List, Tuple, Dict, Callable

import requests
import torch
from omegaconf import OmegaConf
from tabulate import tabulate
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from gnn_lib import tasks, models
from gnn_lib.data.utils import clean_sequence
from gnn_lib.utils import config

_BASE_URL = "https://tokenization.cs.uni-freiburg.de/transformer"
_CONFIGS_URL = f"{_BASE_URL}/configs.zip"
_DATA_URL = f"{_BASE_URL}/data.zip"
_TASK_AND_NAME_TO_URL = {
    "sed_sequence": {
        "gnn_default": f"{_BASE_URL}/sed_sequence_gnn_default.zip",
        "gnn_cliques_wfc": f"{_BASE_URL}/sed_sequence_gnn_cliques_wfc.zip",
    },
    "sed_words": {
        "gnn_default": f"{_BASE_URL}/sed_words_gnn_default.zip",
        "gnn_cliques_wfc": f"{_BASE_URL}/sed_words_gnn_cliques_wfc.zip",
    },
    "sec": {
        "transformer_nmt": f"{_BASE_URL}/sec_transformer_nmt.zip",
        "transformer_words_nmt": f"{_BASE_URL}/sec_transformer_words_nmt.zip",
    }
}

ModelInfo = collections.namedtuple("ModelInfo", ["task", "name", "description"])
StringInputOutput = Union[str, List[str]]


def _download_and_unpack_zip(
        url: str,
        directory: str,
        remove_on_error: bool,
        description: Optional[str] = None
) -> None:
    response = requests.get(url, stream=True)
    if not response.ok:
        raise RuntimeError(f"error downloading file from from {url}: "
                           f"status {response.status_code}, {response.reason}")

    try:
        file_size = int(response.headers.get("content-length", 0))
        pbar = tqdm(
            desc=f"Downloading {url.split('/')[-1]}" if description is None else description,
            total=file_size,
            ascii=True,
            leave=False,
            unit="iB",
            unit_scale=True,
            unit_divisor=1024
        )

        buf = io.BytesIO()
        for data in response.iter_content():
            buf.write(data)
            pbar.update(len(data))

        with zipfile.ZipFile(buf, "r") as zip_file:
            shutil.rmtree(directory, ignore_errors=True)
            os.makedirs(directory)
            zip_file.extractall(directory)

        pbar.close()

    except Exception as e:
        # only remove the model dir on error when it did not exist before
        if remove_on_error:
            shutil.rmtree(directory)
        raise e


def download_data(force_download: bool, logger: logging.Logger, cache_dir: Optional[str] = None) -> str:
    data_dir = os.path.join(cache_dir or get_cache_dir(), "data")
    data_dir_does_not_exist = not os.path.exists(data_dir)
    if data_dir_does_not_exist or force_download:
        logger.info(f"downloading data files from {_DATA_URL} to data cache directory {data_dir}")
        _download_and_unpack_zip(
            _DATA_URL,
            data_dir,
            remove_on_error=data_dir_does_not_exist,
            description="Downloading data files"
        )
    else:
        logger.info(f"data files were already downloaded to data cache directory {data_dir}")
    return data_dir


def download_configs(force_download: bool, logger: logging.Logger, cache_dir: Optional[str] = None) -> str:
    config_dir = os.path.join(cache_dir or get_cache_dir(), "configs")
    config_dir_does_not_exist = not os.path.exists(config_dir)
    if config_dir_does_not_exist or force_download:
        logger.info(f"downloading config files from {_CONFIGS_URL} to config cache directory {config_dir}")
        _download_and_unpack_zip(
            _CONFIGS_URL,
            config_dir,
            remove_on_error=config_dir_does_not_exist,
            description="Downloading config files"
        )
    else:
        logger.info(f"config files were already downloaded to config cache directory {config_dir}")
    return config_dir


def download_model(
        task: str, name: str, force_download: bool, logger: logging.Logger, cache_dir: Optional[str] = None
) -> str:
    """

    Downloads and extracts a model into cache dir and returns the path to the model directory

    :param task: task name
    :param name: unique name of the model
    :param cache_dir: directory to store the model
    :param force_download: download model even if it is already in the cache dir
    :param logger: instance of a logger to log some useful information
    :return: path of the model directory
    """
    cache_dir = cache_dir or get_cache_dir()
    model_dir = os.path.join(cache_dir, task, name)
    model_does_not_exist = not os.path.exists(model_dir)
    if model_does_not_exist or force_download:
        if task not in _TASK_AND_NAME_TO_URL or name not in _TASK_AND_NAME_TO_URL[task]:
            raise RuntimeError(f"no URL for task {task} and model {name}, should not happen")

        url = _TASK_AND_NAME_TO_URL[task][name]
        logger.info(f"downloading model {name} for task {task} from {url} to cache directory {cache_dir}")
        _download_and_unpack_zip(
            url,
            model_dir,
            remove_on_error=model_does_not_exist,
            description=f"Downloading model {name} for task {task}"
        )
    else:
        logger.info(f"model {name} for task {task} was already downloaded to cache directory {cache_dir}")

    experiment_dir = os.listdir(model_dir)
    assert len(experiment_dir) == 1, f"zip file for model {name} for task {task} should contain" \
                                     f" exactly one subdirectory, but found {len(experiment_dir)}"
    return os.path.join(model_dir, experiment_dir[0])


def get_cpu_info() -> str:
    if platform.system() == "Linux":
        try:
            cpu_regex = re.compile(r"model name\t: (.*)", re.DOTALL)
            with open("/proc/cpuinfo", "r", encoding="utf8") as inf:
                cpu_info = inf.readlines()

            for line in cpu_info:
                line = line.strip()
                match = cpu_regex.match(line)
                if match is not None:
                    return match.group(1)
        except Exception:
            return platform.processor()
    return platform.processor()


def get_gpu_info(device: Union[torch.device, str, int]) -> str:
    device_props = torch.cuda.get_device_properties(device)
    return f"{device_props.name} ({device_props.total_memory // 1024 // 1024:,}MiB memory, " \
           f"{device_props.major}.{device_props.minor} compute capability, " \
           f"{device_props.multi_processor_count} multiprocessors)"


def get_device_info(device: torch.device) -> str:
    return get_gpu_info(device) if device.type == "cuda" else get_cpu_info()


def get_cache_dir() -> str:
    return os.path.join(os.path.dirname(__file__), ".cache")


def load_experiment_config(
        experiment: str,
        override_env_vars: Optional[Dict] = None,
        keep_existing_env_vars: bool = False
) -> config.TrainConfig:
    with open(os.path.join(experiment, "cfg.pkl"), "rb") as inf:
        cfg, env_vars = pickle.load(inf)

    if override_env_vars is not None:
        env_vars.update(override_env_vars)

    config.set_gnn_lib_env_vars(env_vars, keep_existing_env_vars)
    OmegaConf.resolve(cfg)
    schema = OmegaConf.structured(config.TrainConfig)
    cfg = OmegaConf.merge(schema, cfg)
    return cfg


def load_experiment(
        experiment: str,
        device: torch.device,
        override_env_vars: Optional[Dict] = None,
        keep_existing_env_vars: bool = False
) -> Tuple[config.TrainConfig, tasks.Task, models.Model]:
    cfg = load_experiment_config(experiment, override_env_vars, keep_existing_env_vars)

    task = tasks.get_task(
        checkpoint_dir=os.path.join(experiment, "checkpoints"),
        variant_cfg=cfg.variant,
        seed=cfg.seed
    )
    sample_inputs = task.generate_sample_inputs(2)
    model = task.get_model(
        sample_inputs=sample_inputs,
        cfg=cfg.model,
        device=device
    )
    task.load_best(model)
    return cfg, task, model


class StringDataset(Dataset):
    def __init__(self, sequences: List[str], sort_by_length: bool = False):
        self.sequences = sequences
        self.sort_by_length = sort_by_length
        if not self.sort_by_length:
            self.indices = list(range(len(self.sequences)))
        else:
            indices_lengths = sorted(
                [(i, len(s)) for i, s in enumerate(self.sequences)],
                key=lambda item: -item[1]
            )
            self.indices = [idx for idx, _ in indices_lengths]

    def __getitem__(self, idx: int) -> Tuple[str, int]:
        return self.sequences[self.indices[idx]], self.indices[idx]

    def __len__(self) -> int:
        return len(self.sequences)

    def word_length(self) -> int:
        return sum(len(s.split()) for s in self.sequences)

    def char_length(self) -> int:
        return sum(len(s) for s in self.sequences)

    @staticmethod
    def collate_fn(batch: List[Tuple[str, Dict]]) -> Tuple[List[str], Dict]:
        return [b[0] for b in batch], {"indices": [b[1] for b in batch], "lengths": [len(b[0]) for b in batch]}


def load_text_file(file_path: str) -> List[str]:
    text = []
    with open(file_path, "r", encoding="utf8") as inf:
        for line in inf:
            text.append(line.strip())
    return text


def get_string_dataset_and_loader(
        file_or_list_of_strings: Union[str, List[str]],
        sort_by_length: bool,
        batch_size: int,
        clean_fn: Callable[[str], str] = clean_sequence
) -> Tuple[StringDataset, DataLoader]:
    if isinstance(file_or_list_of_strings, list):
        text_data = file_or_list_of_strings
    else:
        text_data = load_text_file(file_or_list_of_strings)

    text_data = [clean_fn(line) for line in text_data]

    dataset = StringDataset(text_data, sort_by_length=sort_by_length)
    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        collate_fn=dataset.collate_fn
    )

    return dataset, loader


def reorder_data(items: List, original_indices: List[int]) -> List:
    assert len(items) == len(original_indices)
    reordered_items = [None] * len(items)
    for item, idx in zip(items, original_indices):
        reordered_items[idx] = item
    return reordered_items


def generate_report(
        task: str,
        model: str,
        inputs: List[str],
        runtime: float,
        batch_size: int,
        sort_by_length: bool,
        device: torch.device,
        file_path: Optional[str] = None
) -> Optional[str]:
    input_size = len(inputs)
    input_size_chars = sum(len(ipt) for ipt in inputs)
    report = tabulate(
        [
            [
                task,
                model,
                f"{input_size:,} sequences, {input_size_chars:,} chars",
                runtime,
                input_size / runtime,
                input_size_chars / runtime,
                batch_size,
                "yes" if sort_by_length else "no",
                f"{torch.cuda.get_device_name(device)}, {get_cpu_info()}" if device.type == "cuda" else get_cpu_info()
            ]
        ],
        headers=[
            "Task", "Model", "Input size", "Runtime in seconds", "Seq/s", "Char/s", "Batch size", "Sorted", "Device"
        ],
        floatfmt=[None, None, None, ".3f", ".2f", ".2f", None, None, None, None],
        tablefmt="pipe"
    )
    if file_path is not None:
        if os.path.dirname(file_path):
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
        exists = os.path.exists(file_path)
        with open(file_path, "a" if exists else "w", encoding="utf8") as of:
            if exists:
                of.write(report.splitlines()[-1] + "\n")
            else:
                of.write(report + "\n")
        return None
    else:
        return report
