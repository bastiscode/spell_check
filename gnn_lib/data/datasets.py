import cProfile
import enum
import glob
import json
import multiprocessing as mp
import os
import pickle
import random
import shutil
import tempfile
import time
from typing import List, Tuple, Any, Dict, Optional

import dgl
import lmdb
import lz4.frame
import numpy as np
import omegaconf
from torch.utils import data
from torch.utils.data import ConcatDataset
from tqdm import tqdm

from gnn_lib.data import variants, utils, tokenization, index
from gnn_lib.data.noise import get_noise_from_config
from gnn_lib.data.variants import get_variant_from_config
from gnn_lib.utils import common, io


class Datasets(enum.IntEnum):
    MULTI30K = 1
    WIKIDUMP = 2
    BOOKCORPUS = 3
    NEUSPELL = 4


def _compress(obj: Any) -> bytes:
    return lz4.frame.compress(pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL), compression_level=16)


def _compress_file(in_file: str, out_file: str) -> None:
    with open(in_file, "rb") as inf:
        data = lz4.frame.compress(inf.read(), compression_level=16)
    with open(out_file, "wb") as of:
        of.write(data)


def _decompress(data: bytes) -> Any:
    return pickle.loads(lz4.frame.decompress(data))


def _decompress_file(in_file: str, out_file: str) -> None:
    with open(in_file, "rb") as inf:
        data = lz4.frame.decompress(inf.read())
    with open(out_file, "wb") as of:
        of.write(data)


def write_lmdb_profile(files: List[str],
                       lmdb_path: str,
                       process_idx: int,
                       max_length: Optional[int] = None,
                       num_files_processed: Optional[mp.Value] = None,
                       preprocess_kwargs: Optional[Dict[str, Any]] = None,
                       tqdm_disable: bool = True
                       ) -> None:
    cProfile.runctx(
        "write_lmdb(files, lmdb_path, "
        "process_idx, max_length, num_files_processed, preprocess_kwargs, tqdm_disable)",
        globals(),
        locals(),
        f"prof_write_lmdb_{process_idx}.pstat"
    )


def write_lmdb(files: List[str],
               lmdb_path: str,
               process_idx: int,
               max_length: Optional[int] = None,
               num_files_processed: Optional[mp.Value] = None,
               preprocess_kwargs: Optional[Dict[str, Any]] = None,
               tqdm_disable: bool = True) -> None:
    if max_length is None:
        max_length = float("inf")

    env = utils.open_lmdb(lmdb_path, write=True)
    db_handle = env.open_db()
    with env.begin(write=True) as txn:
        txn.drop(db_handle)

    txn = env.begin(write=True)

    prepare_sample_kwargs = {}
    if preprocess_kwargs is not None:
        tokenizer = tokenization.get_tokenizer_from_config(preprocess_kwargs["tokenizer_cfg"])
        tok_fn = tokenization.get_tokenization_fn(tokenizer, preprocess_kwargs["tokenizer_respect_leading_whitespaces"])
        noise = get_noise_from_config(preprocess_kwargs["noise_cfg"], preprocess_kwargs["seed"])

        prepare_sample_kwargs["batch_size"] = 256
        prepare_sample_kwargs["with_pos_tags"] = True
        prepare_sample_kwargs["with_ner"] = True
        prepare_sample_kwargs["with_dep_parser"] = True

        if "spell_check_index_dir" in preprocess_kwargs:
            nn_index = index.NNIndex(preprocess_kwargs["spell_check_index_dir"])
            neighbor_fn = index.get_neighbor_fn(nn_index, preprocess_kwargs["spell_check_index_num_neighbors"])
        else:
            neighbor_fn = None

    batch_size = 256

    sequences_batch = []
    corrupt_sequences_batch = []
    lengths = []
    num_sequences = 0

    def add_batch() -> None:
        nonlocal sequences_batch, corrupt_sequences_batch, num_sequences, lengths
        if preprocess_kwargs is not None:
            new_sequences_batch = []
            new_corrupt_sequences_batch = []
            for seq, corr_seq in zip(sequences_batch, corrupt_sequences_batch):
                if corr_seq is None:
                    seq, corr_seq = noise.apply(seq)
                new_sequences_batch.append(seq)
                new_corrupt_sequences_batch.append(corr_seq)
            sequences_batch = new_sequences_batch
            corrupt_sequences_batch = new_corrupt_sequences_batch

            corrupt_samples = utils.prepare_samples(corrupt_sequences_batch,
                                                    tok_fn,
                                                    neighbor_fn,
                                                    **prepare_sample_kwargs)

            corrupt_samples_ser = utils.serialize_samples(corrupt_samples)
            for corr_sample, corr_sample_ser, seq in zip(corrupt_samples, corrupt_samples_ser, sequences_batch):
                length = sum(len(t) for t in corr_sample.tokens)
                if length > max_length:
                    continue
                assert txn.put(f"{num_sequences}".encode("utf8"), seq.encode("utf8"))
                assert txn.put(f"{num_sequences}_corrupt".encode("utf8"), corr_sample_ser)
                lengths.append(length)
                num_sequences += 1
        else:
            for seq, corr_seq in zip(sequences_batch, corrupt_sequences_batch):
                assert txn.put(f"{num_sequences}".encode("utf8"), seq.encode("utf8"))
                if corr_seq is not None:
                    assert txn.put(f"{num_sequences}_corrupt".encode("utf8"), corr_seq.encode("utf8"))
                lengths.append(len(seq) if corr_seq is None else len(corr_seq))
                num_sequences += 1

        sequences_batch = []
        corrupt_sequences_batch = []

    for filepath in files:
        with open(filepath, "r", encoding="utf8") as f:
            for line in tqdm(f, desc=f"processing {filepath}", total=io.line_count(filepath), disable=tqdm_disable):
                json_obj = json.loads(line)
                sequence = json_obj["sequence"]
                corrupt_sequence = json_obj.get("corrupt_sequence")

                sequences_batch.append(sequence)
                corrupt_sequences_batch.append(corrupt_sequence)

                if len(sequences_batch) % batch_size == 0:
                    add_batch()

        if num_files_processed is not None:
            num_files_processed.value += 1

    # add remaining elements
    add_batch()

    txn.put(b"dataset_length", _compress(len(lengths)))
    txn.put(b"lengths", _compress(lengths))
    txn.commit()
    env.close()


def preprocess_dataset(lmdb_dir: str,
                       lmdb_name: str,
                       files: List[str],
                       max_length: int,
                       sample_limit: int,
                       preprocess_kwargs: Dict[str, Any]) -> None:
    logger = common.get_logger("PREPROCESS_DATASET")
    start = time.monotonic()

    subset_files = []
    file_idx = 0
    total_sequences = 0
    while total_sequences < sample_limit and file_idx < len(files):
        num_samples = io.line_count(files[file_idx])
        subset_files.append(files[file_idx])
        total_sequences += num_samples
        file_idx += 1

    files = subset_files
    assert len(files) > 0, "got no files to preprocess dataset"

    logger.info(f"Preparing to preprocess {total_sequences} sequences from {len(files)} files")

    # give each process a subset of the files
    temp_dir = tempfile.mkdtemp()
    logger.info(f"Created temporary directory {temp_dir} to store intermediate files")

    ctx = mp.get_context("spawn")
    num_files_processed: mp.Value = ctx.Value("i", 0)

    processes = []

    num_processes = min(int(os.environ.get("GNN_LIB_NUM_PROCESSES", mp.cpu_count())), len(files))
    files_per_process = len(files) / num_processes
    files_per_process_ceil = int(np.ceil(files_per_process))
    files_per_process_floor = int(np.floor(files_per_process))
    file_chunks = []
    file_idx = 0
    for i in range(num_processes):
        if i < int(np.ceil((files_per_process - files_per_process_floor) * num_processes)):
            file_chunks.append(files[file_idx:file_idx + files_per_process_ceil])
            file_idx += files_per_process_ceil
        else:
            file_chunks.append(files[file_idx:file_idx + files_per_process_floor])
            file_idx += files_per_process_floor

    assert sum(len(chunk) for chunk in file_chunks) == len(files)

    lmdb_files = []
    try:
        for i, file_chunk in enumerate(file_chunks):
            lmdb_path = os.path.join(temp_dir, f"{lmdb_name}_{i}")
            lmdb_files.append(lmdb_path)
            p = ctx.Process(
                target=write_lmdb,
                args=(file_chunk,
                      lmdb_path,
                      i,
                      max_length,
                      num_files_processed,
                      preprocess_kwargs)
            )
            p.start()
            processes.append(p)

            logger.info(f"Started writer process {p.pid} on {len(file_chunk)} files")

        log_every = max(len(files) // 1000, 1)
        last_num_files_processed = 0
        while num_files_processed.value < len(files):
            if num_files_processed.value <= last_num_files_processed:
                time.sleep(1)
                continue

            last_num_files_processed = num_files_processed.value
            if num_files_processed.value % log_every == 0:
                end = time.monotonic()
                logger.info(f"Processed {num_files_processed.value}/{len(files)} files: "
                            f"{common.eta_minutes((end - start) / 60, num_files_processed.value, len(files))}")

        for p in processes:
            p.join()
            logger.info(f"Successfully stopped writer process {p.pid}")

        logger.info(f"Moving LMDBs from {temp_dir} to {lmdb_dir}")
        for lmdb_file in lmdb_files:
            shutil.move(lmdb_file, lmdb_dir)

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

    end = time.monotonic()
    logger.info(f"Finished preprocessing in {(end - start) / 60:.2f}min")


class PreprocessedDataset(data.Dataset):
    def __init__(self,
                 directory: str,
                 variant_cfg: omegaconf.DictConfig,
                 seed: int,
                 limit: Optional[int] = None) -> None:
        self.limit = limit if limit is not None and limit > 0 else float("inf")

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

        self.variant = get_variant_from_config(variant_cfg, seed)

    def __getstate__(self) -> Dict[str, Any]:
        state = self.__dict__
        state["envs"] = None
        state["txns"] = None
        state["lengths"] = None
        state["indices"] = None
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
        data = self.txns[txn_idx].get(f"{idx}".encode("utf8"))
        sequence = data.decode("utf8")
        data_corrupt = self.txns[txn_idx].get(f"{idx}_corrupt".encode("utf8"))
        corrupt_sequence = utils.deserialize_samples([data_corrupt])[0]
        return self.variant.prepare_sequence(sequence, corrupt_sequence)

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


class LMDBDataset(dgl.data.DGLDataset):
    def __init__(self,
                 name: str,
                 split: str,
                 raw_dir: str,
                 save_dir: str,
                 variant_cfg: variants.DatasetVariantConfig,
                 seed: int,
                 sample_limit: Optional[int] = None,
                 preprocess_name: Optional[str] = None):
        assert split in {"train", "val", "test"}
        self.split = split
        self.seed = seed
        self.variant_cfg = variant_cfg
        self.variant = get_variant_from_config(self.variant_cfg, self.seed)
        self.rand = np.random.RandomState(self.seed)
        self.envs: List[lmdb.Environment] = []
        self.txns: List[lmdb.Transaction] = []

        self.sizes: List[int] = []
        self.cum_sizes: List[int] = []
        self.lengths: List[int] = []

        self.preprocess_name = preprocess_name
        self.logger = common.get_logger(f"DATASET {name}")

        self.sample_limit = sample_limit if sample_limit is not None and sample_limit > 0 else float("inf")
        super().__init__(name, "", raw_dir, save_dir, (), False, False)

    def __getitem__(self, idx: int) -> Optional[Tuple[dgl.DGLHeteroGraph, Dict]]:
        if self.preprocess_name is not None:
            txn_idx = 0
            offset = 0
            for size in self.cum_sizes:
                if idx < size:
                    break
                txn_idx += 1
                offset = size
            idx -= offset
            data = self.txns[txn_idx].get(f"{idx}".encode("utf8"))
            sequence = data.decode("utf8")
            data_corrupt = self.txns[txn_idx].get(f"{idx}_corrupt".encode("utf8"))
            corrupt_sequence = utils.deserialize_samples([data_corrupt])[0]
            g, info = self.variant.prepare_sequence(sequence, corrupt_sequence)
        else:
            data = self.txns[0].get(f"{idx}".encode("utf8"))
            sequence = data.decode("utf8")
            data_corrupt = self.txns[0].get(f"{idx}_corrupt".encode("utf8"))
            if data_corrupt is not None:
                corrupt_sequence = data_corrupt.decode("utf8")
            else:
                corrupt_sequence = None
            g, info = self.variant.prepare_sequence(sequence, corrupt_sequence)
        return g, info

    def load(self) -> None:
        lmdbs = sorted(self._get_lmdbs())
        assert len(lmdbs) > 0, f"could not find any LMDB files"
        if self.preprocess_name is None:
            assert len(lmdbs) == 1, f"expected a single LMDB file, but got {len(lmdbs)}"
        for lmdb in lmdbs:
            env = utils.open_lmdb(lmdb, write=False)
            txn = env.begin(write=False)
            self.envs.append(env)
            self.txns.append(txn)
            self.sizes.append(_decompress(txn.get(b"dataset_length")))
            self.lengths.extend(_decompress(txn.get(b"lengths")))
        assert len(self.lengths) == sum(self.sizes)
        self.cum_sizes = list(np.cumsum(self.sizes))

    def has_cache(self) -> bool:
        lmdbs = self._get_lmdbs()
        if self.preprocess_name is not None:
            assert len(lmdbs) > 0, f"expected to find some preprocessed lmdbs for dataset {self.name} " \
                                   f"when preprocess_name is given, but got 0, check that the preprocess_name " \
                                   f"{self.preprocess_name} is correct"
        return len(lmdbs) > 0

    def download(self) -> None:
        super().download()

    def save(self) -> None:
        super().save()

    def process(self) -> None:
        if self.preprocess_name is not None:
            return

        os.makedirs(
            self.save_path,
            exist_ok=True
        )

        files = sorted(self.get_files(self.raw_path, self.split))
        rand = random.Random(self.seed)
        rand.shuffle(files)

        write_lmdb(
            files=files,
            lmdb_path=os.path.join(self.save_path, f"{self.split}_lmdb"),
            process_idx=0,
            tqdm_disable=True
        )

    def _get_lmdbs(self) -> List[str]:
        if self.preprocess_name is not None:
            preprocessed_lmdb = os.path.join(self.save_path, self._preprocessed_dir_name(), "lmdb_*")
            return glob.glob(preprocessed_lmdb)
        else:
            return [os.path.join(self.save_path, f"{self.split}_lmdb")]

    @staticmethod
    def get_name() -> str:
        raise NotImplementedError

    @staticmethod
    def get_files(raw_path: str, split: str) -> List[str]:
        raise NotImplementedError

    def _preprocessed_dir_name(self) -> str:
        return f"{self.split}_{self.preprocess_name}"

    def get_lengths(self) -> List[int]:
        return self.lengths[:len(self)]

    def __len__(self) -> int:
        return min(self.sample_limit, sum(self.sizes))


class Multi30k(LMDBDataset):
    def __init__(self,
                 split: str,
                 raw_dir: str,
                 save_dir: str,
                 variant_cfg: variants.DatasetVariantConfig,
                 seed: int,
                 sample_limit: Optional[int] = None,
                 preprocess_name: Optional[str] = None) -> None:
        super().__init__(self.get_name(), split, raw_dir, save_dir, variant_cfg, seed, sample_limit, preprocess_name)

    @staticmethod
    def get_files(raw_path: str, split: str) -> List[str]:
        return [os.path.join(raw_path, f"{split}.jsonl")]

    @staticmethod
    def get_name() -> str:
        return "multi30k"


class Wikidump(LMDBDataset):
    def __init__(self,
                 split: str,
                 raw_dir: str,
                 save_dir: str,
                 variant_cfg: variants.DatasetVariantConfig,
                 seed: int,
                 sample_limit: Optional[int] = None,
                 preprocess_name: Optional[str] = None) -> None:
        super().__init__(self.get_name(), split, raw_dir, save_dir, variant_cfg, seed, sample_limit, preprocess_name)

    @staticmethod
    def get_files(raw_path: str, split: str) -> List[str]:
        with open(os.path.join(raw_path, f"{split}_files.txt"),
                  "r",
                  encoding="utf8") as f:
            return [os.path.join(raw_path, line.strip()) for line in f]

    @staticmethod
    def get_name() -> str:
        return "wikidump"


class Bookcorpus(LMDBDataset):
    def __init__(self,
                 split: str,
                 raw_dir: str,
                 save_dir: str,
                 variant_cfg: variants.DatasetVariantConfig,
                 seed: int,
                 sample_limit: Optional[int] = None,
                 preprocess_name: Optional[str] = None) -> None:
        super().__init__(self.get_name(), split, raw_dir, save_dir, variant_cfg, seed, sample_limit, preprocess_name)

    @staticmethod
    def get_files(raw_path: str, split: str) -> List[str]:
        with open(os.path.join(raw_path, f"{split}_files.txt"),
                  "r",
                  encoding="utf8") as f:
            return [os.path.join(raw_path, line.strip()) for line in f]

    @staticmethod
    def get_name() -> str:
        return "bookcorpus"


class Neuspell(LMDBDataset):
    def __init__(self,
                 split: str,
                 raw_dir: str,
                 save_dir: str,
                 variant_cfg: variants.DatasetVariantConfig,
                 seed: int,
                 sample_limit: Optional[int] = None,
                 preprocess_name: Optional[str] = None):
        if split == "test":
            raise RuntimeError("Split test not supported for dataset Neuspell")
        super().__init__(self.get_name(), split, raw_dir, save_dir, variant_cfg, seed, sample_limit, preprocess_name)

    @staticmethod
    def get_files(raw_path: str, split: str) -> List[str]:
        return [os.path.join(raw_path, f"{split}_1blm.jsonl")]

    @staticmethod
    def get_name() -> str:
        return "neuspell"


class ConcatenatedDataset(data.Dataset):
    def __init__(self,
                 datasets: List[Datasets],
                 **kwargs: Any) -> None:
        super().__init__()
        assert len(datasets) > 0

        self.datasets: List[LMDBDataset] = []
        for dataset in datasets:
            if dataset == Datasets.WIKIDUMP:
                self.datasets.append(Wikidump(**kwargs))
            elif dataset == Datasets.MULTI30K:
                self.datasets.append(Multi30k(**kwargs))
            elif dataset == Datasets.BOOKCORPUS:
                self.datasets.append(Bookcorpus(**kwargs))
            elif dataset == Datasets.NEUSPELL:
                self.datasets.append(Neuspell(**kwargs))
            else:
                raise ValueError(f"Unknown dataset {dataset.name}")

        self.cum_lengths = [0]
        for dataset in self.datasets:
            self.cum_lengths.append(self.cum_lengths[-1] + len(dataset))

    def __getitem__(self, idx: int) -> Any:
        for dataset, cum_length, offset in zip(self.datasets, self.cum_lengths[1:], self.cum_lengths[:-1]):
            if idx < cum_length:
                return dataset[idx - offset]
        raise RuntimeError(f"Should not happen")

    def __len__(self) -> int:
        return sum(len(dataset) for dataset in self.datasets)

    def get_lengths(self) -> List[int]:
        return [length for dataset in self.datasets for length in dataset.get_lengths()]
