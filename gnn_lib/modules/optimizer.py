from dataclasses import dataclass

import omegaconf
from torch import optim, nn

from gnn_lib.modules import Optimizers


@dataclass
class OptimizerConfig:
    type: Optimizers  # = MISSING
    lr: float = 0.0001


@dataclass
class AdamWConfig(OptimizerConfig):
    type: Optimizers = Optimizers.ADAMW
    weight_decay: float = 0.001


@dataclass
class AdamConfig(OptimizerConfig):
    type: Optimizers = Optimizers.ADAM
    weight_decay: float = 0.001


@dataclass
class SGDConfig(OptimizerConfig):
    type: Optimizers = Optimizers.SGD
    weight_decay: float = 0.001
    momentum: float = 0.9


def get_optimizer_from_config(
        cfg: omegaconf.DictConfig,
        model: nn.Module
) -> optim.Optimizer:
    optim_type = Optimizers[cfg.type]
    if optim_type == Optimizers.ADAMW:
        cfg = omegaconf.OmegaConf.structured(AdamWConfig(**cfg))
        return optim.AdamW(
            model.parameters(),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay
        )
    elif optim_type == Optimizers.ADAM:
        cfg = omegaconf.OmegaConf.structured(AdamConfig(**cfg))
        return optim.Adam(
            model.parameters(),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay
        )
    elif optim_type == Optimizers.SGD:
        cfg = omegaconf.OmegaConf.structured(SGDConfig(**cfg))
        return optim.SGD(
            model.parameters(),
            lr=cfg.lr,
            momentum=cfg.momentum,
            weight_decay=cfg.weight_decay,
            nesterov=True
        )
    else:
        raise ValueError(f"Unknown optimizer type {optim_type}")
