import argparse
import os
import random

from omegaconf import OmegaConf

from gnn_lib.data import datasets
from gnn_lib.utils import common, config, io


def preprocess(args: argparse.Namespace) -> None:
    logger = common.get_logger("PREPROCESS")

    cfg = OmegaConf.load(args.config)
    OmegaConf.resolve(cfg)
    schema = OmegaConf.structured(config.PreprocessConfig)
    cfg = OmegaConf.merge(schema, cfg)

    logger.info(f"Using config:\n{OmegaConf.to_yaml(cfg)}\n")

    files = []
    for pattern in cfg.data:
        glob_files = sorted(io.glob_safe(pattern))
        for file in glob_files:
            if file.endswith(".txt"):
                file_dir = os.path.dirname(file)
                with open(file, "r", encoding="utf8") as inf:
                    for line in inf:
                        line = line.strip()
                        if line == "":
                            continue
                        files.append(os.path.join(file_dir, line))
            elif file.endswith(".jsonl"):
                files.append(file)
            else:
                raise ValueError(f"Expected either .jsonl files that contain cleaned data or .txt files that contain "
                                 f"relative paths to .jsonl files, but got {os.path.basename(file)}")

    rand = random.Random(cfg.seed)
    rand.shuffle(files)

    os.makedirs(cfg.output_dir)
    with open(os.path.join(cfg.output_dir, "cfg.yaml"), "w") as of:
        of.write(OmegaConf.to_yaml(cfg, resolve=True, sort_keys=True) + "\n")

    datasets.preprocess_dataset(
        lmdb_dir=cfg.output_dir,
        lmdb_name="lmdb",
        files=files,
        max_length=cfg.max_length,
        sample_limit=cfg.limit if cfg.limit is not None else int(1e15),
        preprocess_kwargs={"spell_check_index_dir": cfg.spell_check_index_dir,
                           "spell_check_index_num_neighbors": cfg.spell_check_index_num_neighbors,
                           "tokenizer_cfg": cfg.tokenizer,
                           "tokenizer_respect_leading_whitespaces": cfg.respect_leading_whitespaces,
                           "noise_cfg": cfg.noise,
                           "seed": cfg.seed}
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str)
    return parser.parse_args()


if __name__ == "__main__":
    preprocess(parse_args())
