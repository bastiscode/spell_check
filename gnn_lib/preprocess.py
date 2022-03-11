import argparse

from omegaconf import OmegaConf

from gnn_lib.data import datasets
from gnn_lib.utils import common, config


def preprocess(args: argparse.Namespace) -> None:
    logger = common.get_logger("PREPROCESS")

    cfg = OmegaConf.load(args.config)
    OmegaConf.resolve(cfg)
    schema = OmegaConf.structured(config.PreprocessConfig)
    cfg = OmegaConf.merge(schema, cfg)

    logger.info(f"Using config:\n{OmegaConf.to_yaml(cfg)}\n")

    datasets.preprocess_dataset(cfg)  # type: ignore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str)
    return parser.parse_args()


if __name__ == "__main__":
    preprocess(parse_args())
