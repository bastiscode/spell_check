import argparse
import collections
import json
import os
from typing import List, Tuple, Callable, Dict, Optional

import numpy as np
import omegaconf
import torch
from torch.backends import cudnn
from torch.utils import data
from tqdm import tqdm

from gnn_lib.data import utils, index
from gnn_lib.modules import lr_scheduler
from gnn_lib.utils import common, io


class VectorizationDataset(data.Dataset):
    def __init__(self,
                 misspelling_files: List[str],
                 jsonl_files: List[str],
                 dictionary: Dict[str, int],
                 seed: int,
                 stem: bool = True,
                 labels_dict: Optional[Dict[str, int]] = None) -> None:
        if stem:
            try:
                from nltk.stem import SnowballStemmer
                stemmer = SnowballStemmer("english")
            except ImportError:
                raise "expect nltk with SnowballStemmer to be installed when stem=True"

        label = 0
        self.labels_dict = labels_dict or {}
        self.joined_misspellings = collections.defaultdict(set)

        for misspelling_file in misspelling_files:
            with open(misspelling_file, "r", encoding="utf8") as inf:
                m = json.load(inf)

            for k, v in m.items():
                if stem:
                    k = stemmer.stem(k)
                if k not in dictionary:
                    continue
                if k in self.labels_dict:
                    word_label = self.labels_dict[k]
                else:
                    self.labels_dict[k] = label
                    word_label = label
                    label += 1
                self.joined_misspellings[word_label] = self.joined_misspellings[k].union(v)
                self.joined_misspellings[word_label].add(k)

        self.rand = np.random.default_rng(seed)
        for file in tqdm(jsonl_files, "Preparing jsonl files", disable=common.disable_tqdm()):
            with open(file, "r", encoding="utf8") as inf:
                for line in inf:
                    json_data = json.loads(line.strip())
                    words = utils.tokenize_words_regex(json_data["sequence"])[0]
                    for org_word in words:
                        if stem:
                            org_word = stemmer.stem(org_word)
                        if org_word not in dictionary:
                            continue
                        if org_word in self.labels_dict:
                            word_label = self.labels_dict[org_word]
                        else:
                            self.labels_dict[org_word] = label
                            word_label = label
                            label += 1
                        word = utils.edit_token(org_word, self.rand)[0]
                        self.joined_misspellings[word_label].add(word)
                        self.joined_misspellings[word_label].add(org_word)

        # convert dict of sets to list
        self.items: List[Tuple[str, int]] = [(input_word, label)
                                             for label in self.joined_misspellings
                                             for input_word in self.joined_misspellings[label]
                                             if len(input_word) <= self.max_length]

    @property
    def max_length(self) -> int:
        return 64

    @property
    def num_labels(self) -> int:
        return len(self.labels_dict)

    def __getitem__(self, idx: int) -> Tuple[str, int]:
        return self.items[idx]

    def __len__(self) -> int:
        return len(self.items)

    def get_collate_fn(self) -> Callable:
        def join(items: List[Tuple[str, int]]) -> Tuple[List[str], torch.Tensor]:
            labels = []
            inputs = []
            for ipt, tgt in items:
                inputs.append(ipt)
                labels.append(tgt)
            return inputs, torch.tensor(labels, dtype=torch.long)

        return join


def train(args: argparse.Namespace) -> None:
    logger = common.get_logger("TRAIN_VECTORIZER")

    assert torch.cuda.is_available(), "need a GPU to train a custom vectorizer, but found none"
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = True

    grad_scaler = torch.cuda.amp.GradScaler()

    if args.train_jsonl_file:
        jsonl_files = []
        with open(args.train_jsonl_file, "r", encoding="utf8") as f:
            for line in f:
                line = line.strip()
                if line == "":
                    continue
                jsonl_files.append(os.path.join(os.path.dirname(args.train_jsonl_file), line))
    else:
        jsonl_files = args.train_jsonl_files

    train_batch_size = 1024
    patience = 3
    epochs_no_improvement = 0
    best_val_loss = float("inf")
    prev_best_val_loss = best_val_loss

    dictionary = io.dictionary_from_file(args.dictionary)
    train_dataset = VectorizationDataset(
        args.train_misspelling_files,
        jsonl_files,
        dictionary,
        args.seed,
        args.stem
    )
    train_loader = data.DataLoader(train_dataset,
                                   batch_size=train_batch_size,
                                   shuffle=True,
                                   drop_last=True,
                                   pin_memory=True,
                                   collate_fn=train_dataset.get_collate_fn(),
                                   num_workers=6)
    val_dataset = VectorizationDataset(
        args.val_misspelling_files,
        [],
        dictionary,
        args.seed,
        args.stem,
        labels_dict=train_dataset.labels_dict
    )
    assert train_dataset.num_labels == val_dataset.num_labels
    logger.info(f"Training dataset has {len(train_dataset)} samples and val dataset has {len(val_dataset)} samples, "
                f"number of labels with stem={args.stem} is {train_dataset.num_labels:,}, dictionary size is "
                f"{len(dictionary):,}")
    val_loader = data.DataLoader(val_dataset,
                                 batch_size=64,
                                 shuffle=False,
                                 drop_last=True,
                                 pin_memory=True,
                                 collate_fn=val_dataset.get_collate_fn())

    loss_fn = torch.nn.CrossEntropyLoss()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = index.CharTransformer()
    model = torch.nn.Sequential(
        encoder,
        torch.nn.Linear(encoder.dim, train_dataset.num_labels)
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scheduler = lr_scheduler.get_lr_scheduler_from_config(
        omegaconf.DictConfig({"type": lr_scheduler.LRSchedulers.COSINE_WITH_WARMUP}),
        optimizer,
        len(train_loader) * args.epochs
    )

    eval_per_epoch = 4
    eval_every = len(train_loader) // eval_per_epoch

    model.train()
    for e in tqdm(range(args.epochs), disable=common.disable_tqdm()):
        for i, (items, labels) in tqdm(enumerate(train_loader), total=len(train_loader), leave=False,
                                       disable=common.disable_tqdm()):
            with torch.cuda.amp.autocast():
                outputs = model(items)

                loss = loss_fn(outputs, labels.to(outputs.device))

            grad_scaler.scale(loss).backward()
            # unscale for proper gradient clipping
            grad_scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            grad_scaler.step(optimizer)
            grad_scaler.update()

            scheduler.step()

            optimizer.zero_grad(set_to_none=True)

            if (i + 1) % eval_every == 0:
                val_losses = []

                model.eval()
                for items, labels in tqdm(val_loader, total=len(val_loader), leave=False,
                                          disable=common.disable_tqdm()):
                    with torch.no_grad():
                        outputs = model(items)

                    loss = loss_fn(outputs, labels.to(outputs.device))
                    val_losses.append(loss.item())

                val_loss = sum(val_losses) / len(val_losses)
                logger.info(f"Epoch {e + 1}, step {i + 1}/{len(train_loader)}: "
                            f"val_loss={val_loss:.5f}")

                if val_loss < best_val_loss:
                    logger.info("Got new best val loss, saving vectorizer")
                    os.makedirs(os.path.dirname(args.out_path), exist_ok=True)
                    # only save encoder here
                    torch.save(encoder, args.out_path)
                    best_val_loss = val_loss

                model.train()

        if best_val_loss >= prev_best_val_loss:
            epochs_no_improvement += 1
        else:
            prev_best_val_loss = best_val_loss
            epochs_no_improvement = 0

        if epochs_no_improvement >= patience:
            logger.info(f"Early stopping after epoch {e + 1}, no improvement since {epochs_no_improvement} epochs, "
                        f"current best val_loss={best_val_loss:.5f}")
            break


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-misspelling-files", nargs="+", type=str, required=True)
    in_file_group = parser.add_mutually_exclusive_group()
    in_file_group.add_argument("--train-jsonl-files", type=str, nargs="+")
    in_file_group.add_argument("--train-jsonl-file", type=str)
    parser.add_argument("--val-misspelling-files", nargs="+", type=str, required=True)
    parser.add_argument("--dictionary", type=str, required=True)
    parser.add_argument("--seed", type=int, default=22)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--out-path", type=str, required=True)
    parser.add_argument("--stem", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
