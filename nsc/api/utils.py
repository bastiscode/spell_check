import collections
import io
import logging
import os
import pickle
import platform
import re
import shutil
import zipfile
from typing import Optional, Union, List, Tuple, Dict, Any, Iterator

import requests
import torch
from omegaconf import OmegaConf
from torch import multiprocessing as mp, autocast
from torch.utils.data import DataLoader, Dataset, BatchSampler, SequentialSampler
from tqdm import tqdm

from nsc import tasks, models
from nsc.api import tables
from nsc.data import utils
from nsc.data.utils import BucketSampler, clean_sequence
from nsc.tasks import Task
from nsc.utils import config, common, DataInput, InfoInput, Batch

_BASE_URL = "https://tokenization.cs.uni-freiburg.de/transformer"
_CONFIGS_URL = f"{_BASE_URL}/configs.zip"
_DATA_URL = f"{_BASE_URL}/data.zip"
_TASK_AND_NAME_TO_URL = {
    "sed_sequence": {
        "gnn_no_feat": f"{_BASE_URL}/sed_sequence_gnn_no_feat.zip",
        "gnn_cliques_wfc": f"{_BASE_URL}/sed_sequence_gnn_cliques_wfc.zip",
        "transformer_no_feat": f"{_BASE_URL}/sed_sequence_transformer_no_feat.zip",
        "transformer": f"{_BASE_URL}/sed_sequence_transformer.zip",
    },
    "sed_words": {
        "gnn_no_feat": f"{_BASE_URL}/sed_words_gnn_no_feat.zip",
        "gnn_cliques_wfc": f"{_BASE_URL}/sed_words_gnn_cliques_wfc.zip",
        "transformer_no_feat": f"{_BASE_URL}/sed_words_transformer_no_feat.zip",
        "transformer": f"{_BASE_URL}/sed_words_transformer.zip",
    },
    "sec": {
        "transformer_nmt": f"{_BASE_URL}/sec_transformer_nmt.zip",
        "transformer_words_nmt": f"{_BASE_URL}/sec_transformer_words_nmt.zip",
    }
}

ModelInfo = collections.namedtuple("ModelInfo", ["task", "name", "description"])
StringInputOutput = Union[str, List[str]]


class _APIBase:
    def __init__(
            self,
            model: models.Model,
            cfg: config.TrainConfig,
            task: tasks.Task,
            device: torch.device,
            logger: logging.Logger
    ) -> None:
        self.model = model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

        self.cfg = cfg
        self.task = task
        self.device = device
        self.logger = logger

        self._mixed_precision_dtype = torch.float32

    @staticmethod
    def from_experiment(
            experiment_dir: str,
            device: Union[str, int]
    ) -> "_APIBase":
        raise NotImplementedError

    @staticmethod
    def from_pretrained(
            model: str,
            device: Union[str, int],
            cache_dir: Optional[str],
            force_download: bool
    ) -> "_APIBase":
        raise NotImplementedError

    @torch.inference_mode()
    def _run_raw(
            self,
            inputs: Union[str, List[str]],
            batch_size: int,
            max_length: int,
            batch_max_length_factor: Optional[float] = None,
            sort_by_length: bool = True,
            show_progress: bool = False,
            **inference_kwargs: Any
    ) -> List[Any]:
        if isinstance(inputs, str):
            inputs = load_text_file(inputs)

        inputs = [clean_sequence(ipt) for ipt in inputs]

        num_workers = 0 if len(inputs) <= 16 else min(4, len(os.sched_getaffinity(0)))
        dataset, loader = get_inference_dataset_and_loader(
            sequences=inputs,
            task=self.task,
            max_length=max_length,
            sort_by_length=sort_by_length,
            batch_size=batch_size,
            batch_max_length_factor=batch_max_length_factor,
            num_workers=num_workers,
            **inference_kwargs
        )

        pbar = tqdm(
            loader,
            total=dataset.byte_length(),
            ascii=True,
            leave=False,
            disable=not show_progress,
            unit="B",
            unit_scale=True,
            unit_divisor=1000
        )

        all_outputs = []
        for i, (batch, infos, indices) in enumerate(pbar):
            batch_strings = [str(dataset.samples[idx]) for idx in indices]
            batch_bytes = sum(len(s.encode("utf8")) for s in batch_strings)

            inference_kwargs.update({
                "input_strings": batch_strings
            })

            pbar.set_description(
                f"[Batch {i + 1}] Running {self.task_name} on {len(indices):,} sequences ({batch_bytes / 1000:,.1f}kB)"
            )

            # this is a slight hack for now, because fp32 on cpu throws an error even when enabled=False
            if self.mixed_precision_enabled:
                with autocast(
                        device_type=self.device.type,
                        dtype=self._mixed_precision_dtype,
                        enabled=self.mixed_precision_enabled
                ):
                    outputs = self.task.inference(self.model, batch, **inference_kwargs)
            else:
                outputs = self.task.inference(self.model, batch, **inference_kwargs)

            all_outputs.extend(outputs)
            pbar.update(batch_bytes)

        pbar.close()
        all_outputs = reorder_data(all_outputs, dataset.indices)
        return self.task.postprocess_inference_outputs(
            inputs, dataset.sample_infos, all_outputs, **inference_kwargs
        )

    def set_precision(self, precision: str) -> None:
        assert precision in {"fp32", "fp16", "bfp16"}

        if precision == "fp32":
            mixed_precision_dtype = torch.float32
        elif precision == "fp16":
            mixed_precision_dtype = torch.float16
        else:
            mixed_precision_dtype = torch.bfloat16

        if self.device.type == "cpu" and precision == "fp16":
            self.logger.info("Setting precision to bfp16 instead of fp16, because fp16 is not supported on CPU yet")
            mixed_precision_dtype = torch.bfloat16

        self._mixed_precision_dtype = mixed_precision_dtype

    @property
    def mixed_precision_enabled(self) -> bool:
        return self._mixed_precision_dtype != torch.float32

    def to(self, device: Union[str, int]) -> "_APIBase":
        self.device = torch.device(device)
        self.model.to(self.device)
        return self

    @property
    def task_name(self) -> str:
        raise NotImplementedError

    @property
    def model_name(self) -> str:
        return self.cfg.experiment_name

    @staticmethod
    def _download(task: str, model: str, cache_dir: Optional[str], force_download: bool) -> Tuple[str, str, str]:
        logger = common.get_logger("DOWNLOAD")

        data_dir = download_data(force_download, logger, cache_dir)

        model_dir = download_model(
            task=task,
            name=model,
            cache_dir=cache_dir,
            force_download=force_download,
            logger=logger
        )
        # check if model comes with a configs.zip, if not download global configs
        if os.path.exists(os.path.join(model_dir, "configs.zip")):
            shutil.unpack_archive(os.path.join(model_dir, "configs.zip"), extract_dir=model_dir)
            config_dir = os.path.join(model_dir, "configs")
        else:
            config_dir = download_configs(force_download, logger, cache_dir)

        return model_dir, data_dir, config_dir


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

    print(cfg, env_vars, override_env_vars)
    if override_env_vars is not None:
        env_vars.update(override_env_vars)

    config.set_nsc_env_vars(env_vars, keep_existing_env_vars)

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
    def __init__(
            self,
            strings: List[str],
            sort_by_length: bool = False
    ) -> None:
        self.strings = strings
        self.sort_by_length = sort_by_length
        if not self.sort_by_length:
            self.indices = list(range(len(self.strings)))
        else:
            indices_lengths = sorted(
                [(i, len(s)) for i, s in enumerate(self.strings)],
                key=lambda item: -item[1]
            )
            self.indices = [idx for idx, _ in indices_lengths]

    def __getitem__(self, idx: int) -> Tuple[str, int]:
        return self.strings[self.indices[idx]], self.indices[idx]

    def __len__(self) -> int:
        return len(self.strings)

    def word_ws_length(self) -> int:
        return sum(len(s.split()) for s in self.strings)

    def char_length(self) -> int:
        return sum(len(s) for s in self.strings)

    @staticmethod
    def collate_fn(items: List[Tuple[str, int]]) -> Tuple[List[str], List[int]]:
        return zip(*items)  # type: ignore


def load_text_file(file_path: str) -> List[str]:
    text = []
    with open(file_path, "r", encoding="utf8") as inf:
        for line in inf:
            text.append(line.strip())
    return text


def save_text_file(file_path: str, content: Iterator[str]) -> None:
    if os.path.dirname(file_path):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w", encoding="utf8") as of:
        for line in content:
            of.write(line + "\n")


def get_string_dataset_and_loader(
        file_or_list_of_strings: Union[str, List[str]],
        sort_by_length: bool,
        batch_size: int
) -> Tuple[StringDataset, DataLoader]:
    if isinstance(file_or_list_of_strings, list):
        text_data = file_or_list_of_strings
    else:
        text_data = load_text_file(file_or_list_of_strings)

    dataset = StringDataset(
        text_data,
        sort_by_length=sort_by_length
    )

    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        collate_fn=dataset.collate_fn
    )

    return dataset, loader


class InferenceDataset(Dataset):
    def __init__(
            self,
            strings: List[str],
            task: Task,
            max_length: int,
            context_length: Optional[int] = None,
            sort_by_length: bool = False,
            **kwargs: Any
    ) -> None:
        self.strings = strings
        self.task = task
        self.max_length = max_length
        self.context_length = context_length
        if self.context_length is None:
            self.context_length = self.max_length // 8
        self.sort_by_length = sort_by_length

        self.samples, self.sample_infos = self.task.prepare_sequences_for_inference(
            self.strings, self.max_length, self.context_length, **kwargs
        )

        if not self.sort_by_length:
            self.indices = list(range(len(self.samples)))
        else:
            indices_lengths = sorted(
                [(i, info.length) for i, info in enumerate(self.sample_infos)],
                key=lambda item: -item[1]
            )
            self.indices = [idx for idx, _ in indices_lengths]

    def __getitem__(self, idx: int) -> Tuple[Tuple[DataInput, InfoInput], utils.InferenceInfo, int]:
        data = self.task.variant.get_inputs(self.samples[self.indices[idx]], is_inference=True)
        return data, self.sample_infos[self.indices[idx]], self.indices[idx]

    def __len__(self) -> int:
        return len(self.samples)

    @property
    def lengths(self) -> List[int]:
        return [info.length for info in self.sample_infos]

    def byte_length(self) -> int:
        return sum([len(str(sample).encode("utf8"))
                    for sample, info in zip(self.samples, self.sample_infos)])

    def char_length(self) -> int:
        return sum(info.ctx_end - info.ctx_start for info in self.sample_infos)

    def token_length(self) -> int:
        return sum(info.length for info in self.sample_infos)

    @staticmethod
    def collate_fn(
            items: List[Tuple[Tuple[DataInput, InfoInput], utils.InferenceInfo, int]]
    ) -> Tuple[Batch, List[utils.InferenceInfo], List[int]]:
        data, infos, indices = list(zip(*items))
        return utils.collate(data), infos, indices


def get_inference_dataset_and_loader(
        sequences: List[str],
        task: Task,
        max_length: int,
        sort_by_length: bool,
        batch_size: int = 16,
        batch_max_length_factor: Optional[float] = None,
        context_length: Optional[int] = None,
        num_workers: int = 0,
        **kwargs: Any
) -> Tuple[InferenceDataset, DataLoader]:
    dataset = InferenceDataset(
        strings=sequences,
        task=task,
        max_length=max_length,
        context_length=context_length,
        sort_by_length=sort_by_length,
        **kwargs
    )

    if batch_max_length_factor is not None:
        assert batch_max_length_factor >= 1.0, \
            f"batch max length factor {batch_max_length_factor} must be larger or equal to 1"
        batch_sampler = BucketSampler(
            dataset=dataset,
            values=dataset.lengths,
            batch_max_value=int(batch_max_length_factor) * max_length,
            seed=0,
            shuffle=False,
            max_value=max_length,
            verbose=False
        )
    else:
        batch_sampler = BatchSampler(
            sampler=SequentialSampler(dataset),
            batch_size=batch_size,
            drop_last=False
        )

    mp.set_sharing_strategy("file_system")
    loader = DataLoader(
        dataset=dataset,
        batch_sampler=batch_sampler,
        collate_fn=dataset.collate_fn,
        num_workers=num_workers,
        worker_init_fn=lambda worker_id: mp.set_sharing_strategy("file_system")
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
        model_name: str,
        num_parameters: int,
        num_sequences: int,
        num_characters: int,
        runtime: float,
        precision: torch.dtype,
        batch_size: int,
        sort_by_length: bool,
        device: torch.device,
        file_path: Optional[str] = None
) -> Optional[str]:
    if precision == torch.float16:
        precision_str = "fp16"
    elif precision == torch.bfloat16:
        precision_str = "bfp16"
    elif precision == torch.float32:
        precision_str = "fp32"
    else:
        raise ValueError(f"expected precision to be one of torch.float16, torch.bfloat16 or torch.float32")

    report = tables.generate_table(
        headers=[[
            "Task",
            "Model",
            "Input size",
            "Runtime in seconds",
            "Seq/s",
            "kChar/s",
            "MiB GPU memory",
            "Mio. parameters",
            "Precision",
            "Batch size",
            "Sorted",
            "Device"
        ]],
        data=[
            [
                task,
                model_name,
                f"{num_sequences:,} sequences, {num_characters:,} chars",
                f"{runtime:.1f}",
                f"{num_sequences / runtime:.1f}",
                f"{num_characters / (runtime * 1000):.1f}",
                f"{torch.cuda.max_memory_reserved(device) // (1024 ** 2):,}" if device.type == "cuda" else "-",
                f"{num_parameters / 1e6:,.1f}",
                precision_str,
                str(batch_size),
                "yes" if sort_by_length else "no",
                f"{torch.cuda.get_device_name(device)}, {get_cpu_info()}" if device.type == "cuda" else get_cpu_info()
            ]
        ],
        fmt="markdown"
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
