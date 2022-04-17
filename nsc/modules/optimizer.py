from dataclasses import dataclass

import omegaconf
from torch import optim, nn
from torch.distributed import optim as dist_optim

from nsc.modules import Optimizers


@dataclass
class OptimizerConfig:
    type: Optimizers  # = MISSING
    zero_redundancy: bool = False
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
        cfg: AdamWConfig = omegaconf.OmegaConf.structured(AdamWConfig(**cfg))
        optim_cls = optim.AdamW
        kwargs = {
            "lr": cfg.lr,
            "weight_decay": cfg.weight_decay
        }
    elif optim_type == Optimizers.ADAM:
        cfg = omegaconf.OmegaConf.structured(AdamConfig(**cfg))
        optim_cls = optim.Adam
        kwargs = {
            "lr": cfg.lr,
            "weight_decay": cfg.weight_decay
        }
    elif optim_type == Optimizers.SGD:
        cfg: SGDConfig = omegaconf.OmegaConf.structured(SGDConfig(**cfg))
        optim_cls = optim.SGD
        kwargs = {
            "lr": cfg.lr,
            "momentum": cfg.momentum,
            "weight_decay": cfg.weight_decay,
            "nesterov": True
        }
    else:
        raise ValueError(f"Unknown optimizer type {optim_type}")

    if cfg.zero_redundancy:
        return dist_optim.ZeroRedundancyOptimizer(
            params=model.parameters(),
            optimizer_class=optim_cls,
            **kwargs
        )
    else:
        return optim_cls(
            params=model.parameters(),
            **kwargs
        )
