import argparse
import collections
import itertools
import json
import os
from typing import List, Tuple, Callable, Dict, Set

import numpy as np
import omegaconf
import torch
from torch import nn
from torch.backends import cudnn
from torch.utils import data
from tqdm import tqdm

from gnn_lib.data import utils, index
from gnn_lib.data.preprocessing import edit_token
from gnn_lib.modules import lr_scheduler
from gnn_lib.utils import common, io


class VectorizationDataset(data.Dataset):
    def __init__(
            self,
            misspelling_files: List[str],
            jsonl_files: List[str],
            dictionary: Dict[str, int],
            seed: int,
            max_length: int
    ) -> None:
        self.max_length = max_length
        self.labels_dict = {w: i for i, w in enumerate(sorted(dictionary)) if len(w) <= self.max_length}
        self.joined_misspellings: Dict[str, Set[str]] = collections.defaultdict(set)

        for misspelling_file in misspelling_files:
            with open(misspelling_file, "r", encoding="utf8") as inf:
                m = json.load(inf)

            for correct, misspellings in m.items():
                if correct not in dictionary:
                    continue

                self.joined_misspellings[correct] = self.joined_misspellings[correct].union(set(misspellings))
                self.joined_misspellings[correct].add(correct)

        self.rand = np.random.default_rng(seed)
        for file in tqdm(
                jsonl_files, "Generating artifical misspellings from jsonl files", disable=common.disable_tqdm()
        ):
            with open(file, "r", encoding="utf8") as inf:
                for line in inf:
                    json_data = json.loads(line.strip())
                    words = utils.tokenize_words_regex(json_data["sequence"])[0]
                    for org_word in words:
                        if org_word not in dictionary:
                            continue

                        for _ in range(10):
                            edited_word = edit_token(org_word, self.rand)[0]
                            self.joined_misspellings[org_word].add(edited_word)

                        self.joined_misspellings[org_word].add(org_word)

        # convert dict of sets to list
        self.items: List[Tuple[str, int, str]] = [
            (misspelled, self.labels_dict[correct], correct)
            for correct in self.joined_misspellings
            for misspelled in self.joined_misspellings[correct]
            if len(misspelled) <= self.max_length
        ]

    @property
    def num_labels(self) -> int:
        return len(self.labels_dict)

    def __getitem__(self, idx: int) -> Tuple[str, int, str]:
        return self.items[idx]

    def __len__(self) -> int:
        return len(self.items)

    def get_collate_fn(self) -> Callable:
        def _join(items: List[Tuple[str, int, str]]) -> Tuple[List[str], torch.Tensor, List[str]]:
            inputs = []
            labels = []
            correct = []
            for ipt, tgt, corr in items:
                inputs.append(ipt)
                labels.append(tgt)
                correct.append(corr)
            return inputs, torch.tensor(labels, dtype=torch.long), correct

        return _join


def train(args: argparse.Namespace) -> None:
    logger = common.get_logger("TRAIN_VECTORIZER")

    assert torch.cuda.is_available(), "need a GPU to train a custom vectorizer, but found none"
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True

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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = index.CharTransformer().to(device)

    dictionary = io.dictionary_from_file(args.dictionary)
    train_dataset = VectorizationDataset(
        args.train_misspelling_files,
        jsonl_files,
        dictionary,
        args.seed,
        encoder.max_length
    )
    train_loader = data.DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        drop_last=True,
        collate_fn=train_dataset.get_collate_fn(),
        num_workers=4
    )
    val_dataset = VectorizationDataset(
        args.val_misspelling_files,
        [],
        dictionary,
        args.seed,
        encoder.max_length
    )
    assert train_dataset.num_labels == val_dataset.num_labels
    logger.info(
        f"Training dataset has {len(train_dataset):,} samples "
        f"for {len(train_dataset.joined_misspellings):,} unique words and "
        f"val dataset has {len(val_dataset):,} samples for {len(val_dataset.joined_misspellings):,} unique words, "
        f"number of labels is {train_dataset.num_labels:,}"
    )

    val_loader = data.DataLoader(
        val_dataset,
        batch_size=128,
        shuffle=False,
        drop_last=True,
        collate_fn=val_dataset.get_collate_fn()
    )

    clf_loss_fn = torch.nn.CrossEntropyLoss()
    emb_loss_fn = torch.nn.MSELoss()
    grad_scaler = torch.cuda.amp.GradScaler(enabled=False)

    classifier = nn.Linear(encoder.dim, train_dataset.num_labels).to(device)

    optimizer = torch.optim.AdamW(itertools.chain(encoder.parameters(), classifier.parameters()), lr=1e-3)
    scheduler = lr_scheduler.get_lr_scheduler_from_config(
        omegaconf.DictConfig({"type": "COSINE_WITH_WARMUP"}),
        optimizer,
        len(train_loader) * args.epochs
    )

    eval_per_epoch = 4
    eval_every = len(train_loader) // eval_per_epoch

    encoder.train()
    classifier.train()
    for e in tqdm(range(args.epochs), disable=common.disable_tqdm()):
        for i, (inputs, labels, correct) in tqdm(
                enumerate(train_loader),
                total=len(train_loader),
                leave=False,
                disable=common.disable_tqdm()
        ):
            with torch.cuda.amp.autocast(enabled=grad_scaler.is_enabled()):
                output_emb = encoder(inputs)
                correct_emb = encoder(correct)

                loss = (
                        clf_loss_fn(classifier(output_emb), labels.to(output_emb.device))
                        +
                        emb_loss_fn(output_emb, correct_emb)
                )

            optimizer.zero_grad(set_to_none=True)
            grad_scaler.scale(loss).backward()
            grad_scaler.step(optimizer)
            grad_scaler.update()

            scheduler.step()

            if (i + 1) % eval_every == 0:
                val_losses = []

                encoder.eval()
                classifier.eval()
                for inputs, labels, correct in tqdm(
                        val_loader,
                        total=len(val_loader),
                        leave=False,
                        disable=common.disable_tqdm()
                ):
                    with torch.no_grad():
                        output_emb = encoder(inputs)
                        correct_emb = encoder(correct)

                        loss = (
                                clf_loss_fn(classifier(output_emb), labels.to(output_emb.device))
                                +
                                emb_loss_fn(output_emb, correct_emb)
                        )

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

                encoder.train()
                classifier.train()

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
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
