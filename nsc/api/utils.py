import collections
import logging
import os
import pickle
import platform
import re
import shutil
import zipfile
from typing import Optional, Union, List, Tuple, Dict, Any, Iterator, Set

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

_BASE_URL = "https://ad-publications.informatik.uni-freiburg.de/" \
            "EMNLP_tokenization_repair_transformer_BHW_2022.materials/nsc"
_CONFIGS_URL = f"{_BASE_URL}/configs.zip"
_DATA_URL = f"{_BASE_URL}/data.zip"
_TASK_AND_NAME_TO_URL = {
    "tokenization repair": {
        "eo small arxiv with errors": f"{_BASE_URL}/tokenization_repair_eo_small_arxiv_with_errors_ported.zip",
        "eo medium arxiv with errors": f"{_BASE_URL}/tokenization_repair_eo_medium_arxiv_with_errors_ported.zip",
        "eo large arxiv with errors": f"{_BASE_URL}/tokenization_repair_eo_large_arxiv_with_errors_ported.zip",
        "tokenization repair+": f"{_BASE_URL}/tokenization_repair_plus_sed.zip",
        "tokenization repair++": f"{_BASE_URL}/tokenization_repair_plus_sed_plus_sec.zip",
    },
    "sed words": {
        "gnn": f"{_BASE_URL}/sed_words_gnn_no_feat.zip",
        "gnn neuspell": f"{_BASE_URL}/sed_words_gnn_no_feat_neuspell.zip",
        "gnn+": f"{_BASE_URL}/sed_words_gnn_cliques_wfc.zip",
        "gnn+ neuspell": f"{_BASE_URL}/sed_words_gnn_cliques_wfc_neuspell.zip",
        "transformer": f"{_BASE_URL}/sed_words_transformer_no_feat.zip",
        "transformer neuspell": f"{_BASE_URL}/sed_words_transformer_no_feat_neuspell.zip",
        "transformer+": f"{_BASE_URL}/sed_words_transformer.zip",
        "transformer+ neuspell": f"{_BASE_URL}/sed_words_transformer_neuspell.zip",
        "tokenization repair+": f"{_BASE_URL}/tokenization_repair_plus_sed.zip",
        "tokenization repair++": f"{_BASE_URL}/tokenization_repair_plus_sed_plus_sec.zip",
    },
    "sec": {
        "transformer nmt": f"{_BASE_URL}/sec_transformer_nmt.zip",
        "transformer nmt neuspell": f"{_BASE_URL}/sec_transformer_nmt_neuspell.zip",
        "transformer with tokenization repair nmt": f"{_BASE_URL}/sec_transformer_with_tokenization_repair_nmt.zip",
        "transformer words nmt": f"{_BASE_URL}/sec_transformer_words_nmt.zip",
        "transformer words nmt neuspell": f"{_BASE_URL}/sec_transformer_words_nmt_neuspell.zip",
        "tokenization repair++": f"{_BASE_URL}/tokenization_repair_plus_sed_plus_sec.zip"
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
        self.model = self.model.requires_grad_(False)

        self.cfg = cfg
        self.task = task
        self.device = device
        self.logger = logger
        self.max_length = task.get_max_input_length(self.model)

        self._mixed_precision_dtype = torch.float32

    @staticmethod
    def from_experiment(
            experiment_dir: str,
            device: Union[str, int]
    ) -> "_APIBase":
        """

        Create a new NSC instance using your own experiment.

        Args:
            experiment_dir: path to the experiment directory
            device: device to load the model to (e.g. "cuda", "cpu" or integer denoting GPU device index)

        Returns: NSC instance

        """
        raise NotImplementedError

    @staticmethod
    def from_pretrained(
            task: str,
            model: str,
            device: Union[str, int],
            download_dir: Optional[str],
            cache_dir: Optional[str],
            force_download: bool
    ) -> "_APIBase":
        """

        Create a new NSC instance using a pretrained model.

        Args:
            task: name of the task
            model: name of the pretrained model
            device: device to load the model to (e.g. "cuda", "cpu" or integer denoting GPU device index)
            download_dir: directory where the compressed model will be downloaded to
            cache_dir: directory where the downloaded, compressed model will be extracted to
            force_download: force download the pretrained model again even if it already exists in the cache_dir

        Returns: NSC instance

        """
        raise NotImplementedError

    @torch.inference_mode()
    def _run_raw(
            self,
            inputs: Union[str, List[str]],
            batch_size: int,
            batch_max_length_factor: Optional[float] = None,
            sort_by_length: bool = True,
            show_progress: bool = False,
            fix_unicode_errors: bool = False,
            fix_all_uppercase: bool = False,
            **inference_kwargs: Any
    ) -> List[Optional[Any]]:
        if isinstance(inputs, str):
            inputs = load_text_file(inputs)

        inputs = [clean_sequence(ipt, fix_unicode_errors, fix_all_uppercase) for ipt in inputs]
        num_inputs = len(inputs)
        invalid_inputs = set(i for i in range(num_inputs) if inputs[i] == "")
        inputs = [ipt for i, ipt in enumerate(inputs) if i not in invalid_inputs]

        num_workers = 0 if len(inputs) <= 16 else len(os.sched_getaffinity(0)) - 1
        dataset, loader = get_inference_dataset_and_loader(
            sequences=inputs,
            task=self.task,
            max_length=self.max_length,
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
        all_outputs = self.task.postprocess_inference_outputs(
            inputs, dataset.sample_infos, all_outputs, **inference_kwargs
        )
        # recombine with invalid/empty inputs
        all_outputs_with_invalid = []
        output_idx = 0
        for i in range(num_inputs):
            if i in invalid_inputs:
                all_outputs_with_invalid.append(None)
            else:
                all_outputs_with_invalid.append(all_outputs[output_idx])
                output_idx += 1
        assert output_idx == len(all_outputs)
        return all_outputs_with_invalid

    def set_precision(self, precision: str) -> None:
        """

        Set the inference precision to use. Default is standard 32bit full precision.
        Using 16bit floats or 16bit brain floats should only be used when you have a supported NVIDIA GPU.
        When running on CPU fp16 will be overwritten with bfp16 since only bfp16 is supported on CPU for now.

        Args:
            precision: precision identifier (one of {fp32, fp16, bfp16})

        Returns: None

        """
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
        """

        Check if mixed precision is enabled (precision is set to something else than fp32).

        Returns: bool whether mixed precision is enabled

        """
        return self._mixed_precision_dtype != torch.float32

    def to(self, device: Union[str, int]) -> "_APIBase":
        """

        Move the model to a different device.

        Args:
            device: device specifier (e.g. "cuda", "cpu" or integer denoting GPU device index)

        Returns: self

        """
        self.device = torch.device(device)
        self.model.to(self.device)
        return self

    @property
    def task_name(self) -> str:
        """
        Check which NSC task is run.

        Returns: name of the task

        """
        raise NotImplementedError

    @property
    def model_name(self) -> str:
        """
        Gives the name of the NSC model in use.

        Returns: name of the model

        """
        return self.cfg.experiment_name

    @staticmethod
    def _download(
            task: str,
            model: str,
            download_dir: Optional[str],
            cache_dir: Optional[str],
            force_download: bool
    ) -> Tuple[str, str, str]:
        logger = common.get_logger("DOWNLOAD")

        data_dir = download_data(force_download, logger, download_dir, cache_dir)

        model_dir = download_model(
            task=task,
            name=model,
            download_dir=download_dir,
            cache_dir=cache_dir,
            force_download=force_download,
            logger=logger
        )

        config_dir = download_configs(force_download, logger, download_dir, cache_dir)

        return model_dir, data_dir, config_dir


def _download_zip(
        url: str,
        zip_file_path: str,
        description: Optional[str] = None
) -> None:
    response = requests.get(url, stream=True)
    if not response.ok:
        raise RuntimeError(f"error downloading file from from {url}: "
                           f"status {response.status_code}, {response.reason}")

    directory = os.path.dirname(zip_file_path)
    if directory:
        os.makedirs(directory, exist_ok=True)
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

        with open(zip_file_path, "wb") as of:
            for data in response.iter_content():
                of.write(data)
                pbar.update(len(data))

        pbar.close()

    except BaseException as e:
        if os.path.exists(zip_file_path):
            os.remove(zip_file_path)
        raise e


def _unpack_zip(
        zip_file_path: str,
        directory: str
) -> None:
    with zipfile.ZipFile(zip_file_path, "r") as zip_file:
        zip_file.extractall(directory)


def download_data(
        force_download: bool,
        logger: logging.Logger,
        download_dir: Optional[str] = None,
        cache_dir: Optional[str] = None
) -> str:
    download_dir = download_dir or get_download_dir()
    zip_file_path = os.path.join(download_dir, "data.zip")
    data_zip_not_downloaded = not os.path.exists(zip_file_path)
    if data_zip_not_downloaded or force_download:
        logger.info(f"downloading data files from {_DATA_URL} to data directory {download_dir}")
        _download_zip(
            _DATA_URL,
            zip_file_path,
            description="Downloading data files"
        )

    cache_dir = cache_dir or get_cache_dir()
    data_dir = os.path.join(cache_dir, "data")
    data_zip_not_extracted = not os.path.exists(data_dir)
    if data_zip_not_extracted or force_download:
        shutil.rmtree(data_dir, ignore_errors=True)
        _unpack_zip(zip_file_path, data_dir)

    return data_dir


def download_configs(
        force_download: bool,
        logger: logging.Logger,
        download_dir: Optional[str] = None,
        cache_dir: Optional[str] = None
) -> str:
    download_dir = download_dir or get_download_dir()
    zip_file_path = os.path.join(download_dir, "configs.zip")
    config_zip_not_downloaded = not os.path.exists(zip_file_path)
    if config_zip_not_downloaded or force_download:
        logger.info(f"downloading data files from {_CONFIGS_URL} to directory {download_dir}")
        _download_zip(
            _CONFIGS_URL,
            zip_file_path,
            description="Downloading config files"
        )

    cache_dir = cache_dir or get_cache_dir()
    config_dir = os.path.join(cache_dir, "configs")
    config_zip_not_extracted = not os.path.exists(config_dir)
    if config_zip_not_extracted or force_download:
        shutil.rmtree(config_dir, ignore_errors=True)
        _unpack_zip(zip_file_path, config_dir)

    return config_dir


def download_model(
        task: str,
        name: str,
        force_download: bool,
        logger: logging.Logger,
        download_dir: Optional[str] = None,
        cache_dir: Optional[str] = None
) -> str:
    """

    Downloads and extracts a model into cache dir and returns the path to the model directory

    :param task: task name
    :param name: unique name of the model
    :param download_dir: directory where compressed models will be saved to
    :param cache_dir: directory where downloaded models are extracted to
    :param force_download: download and extract model even if it is already in the download or cache dir
    :param logger: instance of a logger to log some useful information
    :return: path of the model directory
    """
    if task not in _TASK_AND_NAME_TO_URL or name not in _TASK_AND_NAME_TO_URL[task]:
        raise RuntimeError(f"no URL for task {task} and model {name}, should not happen")
    url = _TASK_AND_NAME_TO_URL[task][name]

    download_dir = download_dir or get_download_dir()
    zip_file_path = os.path.join(download_dir, url.split("/")[-1])
    model_zip_not_downloaded = not os.path.exists(zip_file_path)
    if model_zip_not_downloaded or force_download:
        logger.info(f"downloading model {name} for task {task} from {url} to directory {download_dir}")
        _download_zip(
            url,
            zip_file_path,
            description=f"Downloading model {name} for task {task}"
        )

    cache_dir = cache_dir or get_cache_dir()
    # replace whitespaces in task and name with underscores to avoid any file system issues
    model_dir = os.path.join(cache_dir, task.replace(" ", "_"), name.replace(" ", "_"))
    model_zip_not_extracted = not os.path.exists(model_dir)
    if model_zip_not_extracted or force_download:
        shutil.rmtree(model_dir, ignore_errors=True)
        _unpack_zip(zip_file_path, model_dir)

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
        except BaseException:
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
    return os.getenv("NSC_CACHE_DIR", os.path.join(os.path.dirname(__file__), ".cache"))


def get_download_dir() -> str:
    return os.getenv("NSC_DOWNLOAD_DIR", os.path.join(os.path.dirname(__file__), ".download"))


def load_experiment_config(
        experiment: str,
        override_env_vars: Optional[Dict[str, str]] = None,
        keep_existing_env_vars: Optional[Set[str]] = None
) -> config.TrainConfig:
    with open(os.path.join(experiment, "cfg.pkl"), "rb") as inf:
        cfg, env_vars = pickle.load(inf)

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
        override_env_vars: Optional[Dict[str, str]] = None,
        keep_existing_env_vars: Optional[Set[str]] = None
) -> Tuple[config.TrainConfig, tasks.Task, models.Model]:
    cfg = load_experiment_config(experiment, override_env_vars, keep_existing_env_vars)

    # disable some config options here that are only used for training
    cfg.start_from_checkpoint = None

    task = tasks.get_task(
        checkpoint_dir=os.path.join(experiment, "checkpoints"),
        variant_cfg=cfg.variant,
        seed=cfg.seed
    )
    sample_inputs = task.generate_sample_inputs(2)
    model = task.get_model(
        sample_inputs=sample_inputs,
        cfg=cfg.model,
        device=device,
        inference=True
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
        parameters: Dict[str, int],
        num_sequences: int,
        num_bytes: int,
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

    parameter_str = f"{parameters['used'] / 1e6:,.1f}"
    if parameters["fixed"] > 0:
        parameter_str += f" ({parameters['fixed'] / 1e6:,.1f} fixed)"

    report = tables.generate_table(
        headers=[[
            "Task",
            "Model",
            "Input size",
            "Runtime in seconds",
            "Seq/s",
            "kB/s",
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
                f"{num_sequences:,} sequences, {num_bytes / 1000:,.1f} kB",
                f"{runtime:.1f}",
                f"{num_sequences / runtime:.1f}",
                f"{num_bytes / (runtime * 1000):.1f}",
                f"{torch.cuda.max_memory_reserved(device) // (1024 ** 2):,}" if device.type == "cuda" else "-",
                parameter_str,
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
