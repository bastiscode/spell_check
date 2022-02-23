import os.path
from dataclasses import dataclass
from typing import Optional, Any, Dict, List

import omegaconf
from omegaconf import MISSING

from gnn_lib.data.tokenization import TokenizerConfig


def _load_from_file(file_path: str) -> omegaconf.DictConfig:
    assert "GNN_LIB_CONFIG_DIR" in os.environ, \
        ("to use the from_file resolver, you must specify the root directory of all "
         "configs using the GNN_LIB_CONFIG_DIR env variable")
    cfg_directory = os.environ["GNN_LIB_CONFIG_DIR"]
    cfg = omegaconf.OmegaConf.load(os.path.join(cfg_directory, file_path))
    return cfg


omegaconf.OmegaConf.register_new_resolver("from_file", _load_from_file)


@dataclass
class PreprocessConfig:
    # datasets: List[Datasets] = MISSING
    data: List[str] = MISSING
    # data_dir: str = MISSING
    output_dir: str = MISSING
    tokenizer: TokenizerConfig = MISSING
    respect_leading_whitespaces: bool = False
    spell_check_index_dir: Optional[str] = None
    spell_check_index_num_neighbors: int = 20
    noise: Any = MISSING

    max_length: int = 512
    seed: int = 22
    limit: Optional[int] = None


@dataclass
class TrainConfig:
    # training info to be specified by user
    model: Any = MISSING
    variant: Any = MISSING
    experiment_dir: str = MISSING
    experiment_name: str = MISSING

    # data
    data_dir: str = MISSING
    datasets: List[str] = MISSING
    dataset_limits: List[int] = MISSING
    val_splits: List[float] = MISSING

    # training hyperparameters
    epochs: int = 20
    batch_size: int = 64
    batch_max_length: Optional[int] = None
    bucket_span: Optional[int] = None
    optimizer: Any = MISSING
    lr_scheduler: Optional[Any] = None

    # additional options
    log_per_epoch: int = 100
    eval_per_epoch: int = 4
    keep_last_n_checkpoints: int = 0  # 0 means we only keep the best checkpoint
    seed: int = 22
    num_workers: Optional[int] = None
    mixed_precision: bool = True

    start_from_checkpoint: Optional[str] = None
    exponential_moving_average: Optional[List[float]] = None


@dataclass
class TestConfig:
    experiment: str = MISSING

    in_file: Optional[str] = None
    out_path: str = MISSING
    runtime_file: Optional[str] = None

    batch_size: int = 64
    inference_kwargs: Optional[Dict[str, Any]] = None

    verbose: bool = False
    overwrite: bool = False
    sort_by_length: bool = False
    cpu: bool = False


def set_gnn_lib_env_vars(env_vars: Dict[str, Any], delete_old: bool = True) -> None:
    # delete all previous gnn lib env vars
    if delete_old:
        for k in os.environ:
            if k.startswith("GNN_LIB_"):
                del os.environ[k]
    # set all new gnn lib env vars
    for k, v in env_vars.items():
        os.environ[k] = v

