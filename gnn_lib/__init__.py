import os
import pickle
from typing import Tuple, Optional, Dict

import torch
from omegaconf import OmegaConf

from gnn_lib import models
from gnn_lib.data import Datasets, DatasetVariants
from gnn_lib.tasks import Tasks, get_task
from gnn_lib.utils import config

GNN_LIB_DIR = os.path.dirname(__file__)


def load_experiment_config(
        experiment: str,
        override_env_vars: Optional[Dict] = None
) -> config.TrainConfig:
    with open(os.path.join(experiment, "cfg.pkl"), "rb") as inf:
        cfg, env_vars = pickle.load(inf)

    if override_env_vars is not None:
        env_vars.update(override_env_vars)

    config.set_gnn_lib_env_vars(env_vars)
    OmegaConf.resolve(cfg)
    schema = OmegaConf.structured(config.TrainConfig)
    cfg = OmegaConf.merge(schema, cfg)
    return cfg


def load_experiment(
        experiment: str,
        device: torch.device,
        override_env_vars: Optional[Dict] = None
) -> Tuple[config.TrainConfig, tasks.Task, models.Model]:
    cfg = load_experiment_config(experiment, override_env_vars)

    task = tasks.get_task(
        checkpoint_dir=os.path.join(experiment, "checkpoints"),
        variant_cfg=cfg.variant,
        seed=cfg.seed
    )
    sample_g, _ = task.generate_sample_inputs(2)
    model = task.get_model(
        sample_g=sample_g,
        cfg=cfg.model,
        device=device
    )
    task.load_best(model)
    return cfg, task, model
