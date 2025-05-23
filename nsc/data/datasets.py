import enum
import json
import os
import pickle
import random
import shutil
import tempfile
import time
from typing import List, Tuple, Any, Dict, Optional

import dgl
import lz4.frame
import numpy as np
import omegaconf
from omegaconf import OmegaConf
from torch import multiprocessing as mp
from torch.utils import data
from torch.utils.data import ConcatDataset
from tqdm import tqdm

from nsc.data import utils, tokenization, index
from nsc.data.preprocessing import get_preprocessing_from_config, get_preprocessing_fn
from nsc.data.variants import get_variant_from_config
from nsc.utils import common, io
from nsc.utils.config import PreprocessConfig


class Datasets(enum.IntEnum):
    MULTI30K = 1
    WIKIDUMP = 2
    BOOKCORPUS = 3
    NEUSPELL = 4


def _compress(obj: Any) -> bytes:
    return lz4.frame.compress(pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL), compression_level=16)


def _decompress(data: bytes) -> Any:
    return pickle.loads(lz4.frame.decompress(data))


def write_lmdb(
        file_queue: mp.Queue,
        lmdb_path: str,
        preprocess_cfg: PreprocessConfig,
        tqdm_disable: bool = True
) -> None:
    logger = common.get_logger("LMDB_WRITER")
    logger.info(f"Opened LMDB at {lmdb_path}")
    env = utils.open_lmdb(lmdb_path, write=True)
    db_handle = env.open_db()
    with env.begin(write=True) as txn:
        txn.drop(db_handle)

    txn = env.begin(write=True)

    tokenizer = tokenization.get_tokenizer_from_config(preprocess_cfg.tokenizer)
    tok_fn = tokenization.get_tokenization_fn(tokenizer, preprocess_cfg.respect_leading_whitespaces)
    preprocessing = get_preprocessing_from_config(preprocess_cfg.preprocessing, preprocess_cfg.seed)

    if preprocess_cfg.index is not None:
        neighbor_index = index.NNIndex(preprocess_cfg.index)
        neighbor_fn = index.get_neighbor_fn(neighbor_index, preprocess_cfg.index_num_neighbors)
    else:
        neighbor_fn = None

    preprocessing_fn = get_preprocessing_fn(
        preprocessing,
        tok_fn,
        neighbor_fn,
        with_pos_tags=preprocess_cfg.with_pos_tags,
        with_ner=preprocess_cfg.with_ner,
        with_dep_parser=preprocess_cfg.with_dep_parser,
        batch_size=preprocess_cfg.batch_size
    )

    batch_size = 256

    sequences_batch = []
    target_sequences_batch = []
    lengths = []
    num_sequences = 0

    def add_batch() -> None:
        nonlocal sequences_batch, target_sequences_batch, num_sequences, lengths

        preprocessed_batch = preprocessing_fn(sequences_batch, target_sequences_batch, False)

        preprocessed_samples_ser = utils.serialize_samples([sample for sample, _ in preprocessed_batch])
        for (sample, target_sequence), sample_ser in zip(
                preprocessed_batch,
                preprocessed_samples_ser
        ):
            length = sum(max(len(t), 1) for t in sample.tokens)
            if length > preprocess_cfg.max_length:
                continue
            assert txn.put(f"{num_sequences}".encode("utf8"), sample_ser)
            assert txn.put(f"{num_sequences}_target".encode("utf8"), target_sequence.encode("utf8"))
            lengths.append(length)
            num_sequences += 1

        sequences_batch = []
        target_sequences_batch = []

    while not file_queue.empty():
        try:
            filepath = file_queue.get(timeout=5)
        except Exception:
            continue

        with open(filepath, "r", encoding="utf8") as f:
            for line in tqdm(f, desc=f"processing {filepath}", total=io.line_count(filepath), disable=tqdm_disable):
                json_obj = json.loads(line)
                sequence = json_obj["sequence"]
                target_sequence = json_obj.get("target_sequence")

                sequences_batch.append(sequence)
                target_sequences_batch.append(target_sequence)

                if len(sequences_batch) % batch_size == 0:
                    add_batch()

    # add remaining elements
    add_batch()

    txn.put(b"dataset_length", _compress(len(lengths)))
    txn.put(b"lengths", _compress(lengths))
    txn.commit()
    env.close()

    logger.info(f"Wrote {len(lengths)} elements to LMDB {lmdb_path}")


def preprocess_dataset(preprocess_cfg: PreprocessConfig) -> None:
    logger = common.get_logger("PREPROCESS_DATASET")

    files = []
    for pattern in preprocess_cfg.data:
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

    rand = random.Random(preprocess_cfg.seed)
    rand.shuffle(files)

    os.makedirs(preprocess_cfg.output_dir)
    with open(os.path.join(preprocess_cfg.output_dir, "cfg.yaml"), "w") as of:
        of.write(OmegaConf.to_yaml(preprocess_cfg, resolve=True, sort_keys=True))

    sample_limit = preprocess_cfg.limit or float("inf")

    logger.info("Determining which files to process")
    subset_files = []
    total_sequences = 0
    for file in files:
        num_samples = io.line_count(file)
        total_sequences += num_samples
        subset_files.append(file)
        if total_sequences >= sample_limit:
            break

    files = subset_files
    assert len(files) > 0, "got no files to preprocess dataset"

    logger.info(f"Preparing to preprocess {total_sequences} sequences from {len(files)} files")

    # give each process a subset of the files
    temp_dir = tempfile.mkdtemp()
    logger.info(f"Created temporary directory {temp_dir} to store intermediate files")

    # fill queue with files
    file_queue = mp.Queue()
    for file in subset_files:
        file_queue.put(file)

    num_processes = min(int(os.environ.get("NSC_NUM_PROCESSES", min(8, len(os.sched_getaffinity(0))))), len(files))

    try:
        processes = []
        lmdb_files = []
        for i in range(num_processes):
            lmdb_path = os.path.join(temp_dir, f"lmdb_{i}")
            lmdb_files.append(lmdb_path)
            p = mp.Process(
                target=write_lmdb,
                args=(
                    file_queue,
                    lmdb_path,
                    preprocess_cfg
                )
            )
            p.start()
            processes.append(p)

            logger.info(f"Started writer process {p.pid}")

        start = time.perf_counter()
        file_queue_size = file_queue.qsize()
        while not file_queue.empty():
            current_queue_size = file_queue.qsize()
            if current_queue_size >= file_queue_size:
                time.sleep(1)
                continue

            file_queue_size = current_queue_size
            end = time.perf_counter()
            files_processed = len(files) - file_queue_size
            logger.info(
                f"{file_queue_size}/{len(files)} left in queue: "
                f"{common.eta_minutes((end - start) / 60, files_processed, len(files))}"
            )

        for p in processes:
            p.join()
            logger.info(f"Successfully stopped writer process {p.pid}")

        logger.info(f"Moving LMDBs from {temp_dir} to {preprocess_cfg.output_dir}")
        for lmdb_file in lmdb_files:
            shutil.move(lmdb_file, preprocess_cfg.output_dir)

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

    end = time.monotonic()
    logger.info(f"Finished preprocessing in {(end - start) / 60:.2f} minutes")


class PreprocessedDataset(data.Dataset):
    def __init__(self,
                 directory: str,
                 variant_cfg: omegaconf.DictConfig,
                 seed: int,
                 limit: Optional[int] = None) -> None:
        self.limit = limit or float("inf")

        self.seed = seed
        self.envs = []
        self.txns = []
        self.sizes = []
        self.lengths = []
        self.lmdb_paths = sorted(io.glob_safe(os.path.join(directory, "lmdb_*")))
        assert len(self.lmdb_paths) > 0, f"could not find any LMDB files in directory {directory}"
        for lmdb_path in self.lmdb_paths:
            env = utils.open_lmdb(lmdb_path, write=False)
            txn = env.begin(write=False)
            self.envs.append(env)
            self.txns.append(txn)
            self.sizes.append(_decompress(txn.get(b"dataset_length")))
            self.lengths.extend(_decompress(txn.get(b"lengths")))
        assert len(self.lengths) == sum(self.sizes)
        self.cum_sizes = list(np.cumsum(self.sizes))

        # we do this here because if a limit is set, we want to be sure that we get samples from all parts of
        # the underlying preprocessed lmdb dataset
        rand = np.random.default_rng(self.seed)
        self.indices = rand.permutation(len(self))

        self.variant_cfg = variant_cfg
        self.variant = get_variant_from_config(self.variant_cfg, self.seed)

    def __getstate__(self) -> Dict[str, Any]:
        state = self.__dict__
        state["envs"] = None
        state["txns"] = None
        state["lengths"] = None
        state["indices"] = None
        state["variant"] = None
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        self.__dict__ = state
        self.envs = []
        self.txns = []
        self.lengths = []
        for lmdb_path in self.lmdb_paths:
            env = utils.open_lmdb(lmdb_path, write=False)
            txn = env.begin(write=False)
            self.envs.append(env)
            self.txns.append(txn)
            self.lengths.extend(_decompress(txn.get(b"lengths")))

        rand = np.random.default_rng(self.seed)
        self.indices = rand.permutation(len(self))
        self.variant = get_variant_from_config(self.variant_cfg, self.seed)

    def __getitem__(self, idx: int) -> Tuple[dgl.DGLHeteroGraph, Dict]:
        idx = self.indices[idx]
        txn_idx = 0
        offset = 0
        for size in self.cum_sizes:
            if idx < size:
                break
            txn_idx += 1
            offset = size
        idx -= offset
        data_sequence = self.txns[txn_idx].get(f"{idx}".encode("utf8"))
        sequence = utils.deserialize_samples([data_sequence])[0]
        data_target_sequence = self.txns[txn_idx].get(f"{idx}_target".encode("utf8"))
        target_sequence = data_target_sequence.decode("utf8")
        return self.variant.get_inputs(sequence, target_sequence)

    def __len__(self) -> int:
        return min(self.limit, sum(self.sizes))

    def get_lengths(self) -> List[int]:
        return [self.lengths[idx] for idx in self.indices]


class PreprocessedSubsetDataset(data.Dataset):
    def __init__(self, dataset: PreprocessedDataset, indices: List[int]) -> None:
        self.dataset = dataset
        self.indices = indices
        lengths = self.dataset.get_lengths()
        self.lengths = [lengths[idx] for idx in self.indices]

    def __getitem__(self, idx: int) -> Tuple[dgl.DGLHeteroGraph, Dict]:
        return self.dataset[self.indices[idx]]

    def __len__(self) -> int:
        return len(self.indices)

    def get_lengths(self) -> List[int]:
        return self.lengths


class PreprocessedConcatDataset(ConcatDataset[PreprocessedSubsetDataset]):
    def get_lengths(self) -> List[int]:
        all_lengths = []
        for dataset in self.datasets:
            all_lengths.extend(dataset.get_lengths())
        return all_lengths
