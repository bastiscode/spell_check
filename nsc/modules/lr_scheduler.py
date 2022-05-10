import math
from dataclasses import dataclass
from typing import List, Union

import omegaconf
from omegaconf import MISSING
from torch import optim

from nsc.modules import LRSchedulers


@dataclass
class LRSchedulerConfig:
    type: LRSchedulers  # = MISSING


@dataclass
class CosineWithWarmupConfig(LRSchedulerConfig):
    type: LRSchedulers = LRSchedulers.COSINE_WITH_WARMUP
    warmup_steps: float = 0.05
    max_reduce_factor: float = 0.01


@dataclass
class LinearWithWarmupConfig(LRSchedulerConfig):
    type: LRSchedulers = LRSchedulers.LINEAR_WITH_WARMUP
    warmup_steps: float = 0.05
    max_reduce_factor: float = 0.01


@dataclass
class StepLRWithWarmupConfig(LRSchedulerConfig):
    type: LRSchedulers = LRSchedulers.STEP_WITH_WARMUP
    warmup_steps: float = 0.05
    reduce_at: List[float] = MISSING
    reduce_factor: float = 0.1


@dataclass
class ConstantWithWarmupConfig(LRSchedulerConfig):
    type: LRSchedulers = LRSchedulers.CONSTANT_WITH_WARMUP
    warmup_steps: float = 0.05


def get_lr_scheduler_from_config(
        cfg: Union[LRSchedulerConfig, omegaconf.DictConfig],
        optimizer: optim.Optimizer,
        num_training_steps: int
) -> optim.lr_scheduler.LambdaLR:
    # explicitly convert ot dict config first, this way we support both dictconfigs
    # and structured configs as input
    cfg: omegaconf.DictConfig = omegaconf.DictConfig(cfg)
    lr_type = LRSchedulers[cfg.type] if isinstance(cfg.type, str) else cfg.type
    if lr_type == LRSchedulers.LINEAR_WITH_WARMUP:
        cfg = omegaconf.OmegaConf.structured(LinearWithWarmupConfig(**cfg))
        if cfg.warmup_steps <= 1.0:
            warmup_steps = num_training_steps * cfg.warmup_steps
        else:
            warmup_steps = int(cfg.warmup_steps)
        assert 0 <= cfg.max_reduce_factor < 1

        decay_steps = max(1, num_training_steps - warmup_steps)
        slope = -(1 - cfg.max_reduce_factor) / decay_steps

        def _linear(step: int) -> float:
            if step < warmup_steps:
                return step / max(1.0, warmup_steps)
            return 1 + slope * (step - warmup_steps)

        return optim.lr_scheduler.LambdaLR(optimizer=optimizer,
                                           lr_lambda=_linear)

    elif lr_type == LRSchedulers.COSINE_WITH_WARMUP:
        cfg = omegaconf.OmegaConf.structured(CosineWithWarmupConfig(**cfg))
        if cfg.warmup_steps <= 1.0:
            warmup_steps = num_training_steps * cfg.warmup_steps
        else:
            warmup_steps = int(cfg.warmup_steps)
        assert 0 <= cfg.max_reduce_factor < 1

        factor = 0.5 * (1 - cfg.max_reduce_factor)
        offset = (1 / factor) - 1
        decay_steps = max(1, num_training_steps - warmup_steps)

        def _cosine(step: int) -> float:
            if step < warmup_steps:
                return step / max(1.0, warmup_steps)
            frac = (step - warmup_steps) / decay_steps
            return factor * (offset + math.cos(math.pi * frac))

        return optim.lr_scheduler.LambdaLR(optimizer=optimizer,
                                           lr_lambda=_cosine)

    elif lr_type == LRSchedulers.STEP_WITH_WARMUP:
        cfg = omegaconf.OmegaConf.structured(StepLRWithWarmupConfig(**cfg))
        if cfg.warmup_steps <= 1.0:
            warmup_steps = num_training_steps * cfg.warmup_steps
        else:
            warmup_steps = int(cfg.warmup_steps)

        assert all(r < 1.0 for r in cfg.reduce_at)
        assert all(cfg.reduce_at[i] > cfg.reduce_at[i - 1] for i in range(1, len(cfg.reduce_at)))
        reduce_at_steps = [r * num_training_steps for r in cfg.reduce_at]
        reduce_factor = cfg.reduce_factor

        def _reduce(step: int) -> float:
            if step < warmup_steps:
                return step / max(1.0, warmup_steps)
            factor = 1.0
            for reduce_at in reduce_at_steps:
                if step < reduce_at:
                    break
                factor *= reduce_factor
            return factor

        return optim.lr_scheduler.LambdaLR(optimizer=optimizer,
                                           lr_lambda=_reduce)

    elif lr_type == LRSchedulers.CONSTANT_WITH_WARMUP:

        def _constant(step: int) -> float:
            if step < warmup_steps:
                return step / max(1.0, warmup_steps)
            return 1.0

        return optim.lr_scheduler.LambdaLR(optimizer=optimizer,
                                           lr_lambda=_constant)

    else:
        raise ValueError(f"Unknown learning rate scheduler {lr_type}")
