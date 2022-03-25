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


omegaconf.OmegaConf.register_new_resolver("from_file", _load_from_file, replace=True)


@dataclass
class PreprocessConfig:
    data: List[str] = MISSING
    output_dir: str = MISSING
    preprocessing: Any = MISSING

    # options for tokenization and neighbor index
    tokenizer: TokenizerConfig = MISSING
    respect_leading_whitespaces: bool = True
    index: Optional[str] = None
    index_num_neighbors: int = 20

    # options for creating space docs
    with_pos_tags: bool = False
    with_ner: bool = False
    with_dep_parser: bool = False
    batch_size: Optional[int] = None

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


def set_gnn_lib_env_vars(env_vars: Dict[str, Any], keep_existing_env_vars: bool = False) -> None:
    existing_env_vars = set()
    if not keep_existing_env_vars:
        # delete all previous gnn lib env vars
        for k in os.environ:
            if k.startswith("GNN_LIB_"):
                del os.environ[k]
    else:
        for k in os.environ:
            if k.startswith("GNN_LIB"):
                existing_env_vars.add(k)
    # set all new gnn lib env vars
    for k, v in env_vars.items():
        if k not in existing_env_vars:
            os.environ[k] = v

