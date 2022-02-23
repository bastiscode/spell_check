import enum


class Optimizers(enum.IntEnum):
    ADAMW = 1
    ADAM = 2
    SGD = 3


class LRSchedulers(enum.IntEnum):
    COSINE_WITH_WARMUP = 1
    LINEAR_WITH_WARMUP = 2
    STEP_WITH_WARMUP = 3
    CONSTANT_WITH_WARMUP = 4
