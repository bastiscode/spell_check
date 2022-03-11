from typing import Tuple, List, Optional

import numpy as np
import omegaconf

from gnn_lib.data.datasets import (
    Datasets, PreprocessedDataset, PreprocessedSubsetDataset, PreprocessedConcatDataset
)
from gnn_lib.data.tokenization import TokenizerConfig
from gnn_lib.data.variants import DatasetVariants, DatasetVariantConfig


def get_train_val_datasets(
        datasets: List[str],
        variant_cfg: omegaconf.DictConfig,
        seed: int,
        val_splits: List[float],
        dataset_limits: List[Optional[int]]
) -> Tuple[PreprocessedConcatDataset, PreprocessedConcatDataset]:
    train_datasets = []
    val_datasets = []

    assert len(val_splits) == len(datasets)

    for dataset_dir, dataset_limit, val_split in zip(datasets, dataset_limits, val_splits):
        dataset = PreprocessedDataset(
            directory=dataset_dir,
            variant_cfg=variant_cfg,
            seed=seed,
            limit=dataset_limit
        )

        assert val_split > 0, "val split must be greater 0"
        if 1 > val_split > 0:
            val_split = int(val_split) * len(dataset)
        else:
            val_split = int(val_split)

        assert val_split < len(dataset), \
            f"dataset {dataset_dir}: val samples {val_split} must be smaller than number of all " \
            f"dataset samples {len(dataset)}"

        rand = np.random.default_rng(seed)
        indices = rand.permutation(len(dataset))
        val_indices = indices[:val_split]
        train_indices = indices[val_split:]

        train_dataset = PreprocessedSubsetDataset(dataset, train_indices)
        train_datasets.append(train_dataset)
        val_dataset = PreprocessedSubsetDataset(dataset, val_indices)
        val_datasets.append(val_dataset)

    return PreprocessedConcatDataset(train_datasets), PreprocessedConcatDataset(val_datasets)
