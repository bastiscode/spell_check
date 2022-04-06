import argparse
import collections
import copy
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

from gnn_lib.data import utils, index, preprocessing
from gnn_lib.modules import lr_scheduler
from gnn_lib.utils import common, io


class VectorizationDataset(data.Dataset):
    def __init__(
            self,
            max_context_length: int,
            misspelling_files: List[str],
            jsonl_files: List[str],
            dictionary: Dict[str, int],
            seed: int,
            max_length: int,
            max_samples: int
    ) -> None:
        assert max_context_length > 0
        self.max_context_length = max_context_length
        self.context_lengths = list(range(1, self.max_context_length + 1))

        self.max_samples = max_samples
        self.max_length = max_length
        self.max_length_half = max_length // 2
        self.labels_dict = {w: i for i, w in enumerate(sorted(dictionary)) if len(w) <= self.max_length}

        self.misspellings: Dict[str, Set[str]] = collections.defaultdict(set)
        contexts: List[Tuple] = []

        for misspelling_file in misspelling_files:
            with open(misspelling_file, "r", encoding="utf8") as inf:
                m = json.load(inf)

            for correct, misspelled in m.items():
                if correct not in dictionary or len(correct) > max_length:
                    continue

                # misspelling should be smaller max length and not a real word misspelling, because then
                # the label is ambiguous
                misspelled_set = set(m for m in misspelled if len(m) <= max_length and m not in dictionary)
                self.misspellings[correct] = self.misspellings[correct].union(misspelled_set)
                self.misspellings[correct].add(correct)

        self.rand = np.random.default_rng(seed)
        for file in tqdm(
                jsonl_files,
                desc="Generating artificial contexts from jsonl files",
                leave=False,
                disable=common.disable_tqdm()
        ):
            with open(file, "r", encoding="utf8") as inf:
                lines = [json.loads(line.strip())["sequence"] for line in inf]
                for line in tqdm(
                        lines,
                        desc="Processing file",
                        leave=False,
                        disable=common.disable_tqdm()
                ):
                    words, doc = utils.tokenize_words(line, return_doc=True)
                    edited_words = preprocessing.corrupt_words(
                        copy.deepcopy(words), doc, 1.0, 1.0, "artificial", self.rand
                    )

                    whitespaces = (
                            [False] * self.max_context_length
                            + [t.whitespace_ == " " for t in doc]
                            + [False] * self.max_context_length
                    )
                    words = [""] * self.max_context_length + words + [""] * self.max_context_length
                    edited_words = [""] * self.max_context_length + edited_words + [""] * self.max_context_length

                    for i in range(self.max_context_length, len(words) - self.max_context_length):
                        if words[i] not in dictionary:
                            continue

                        context_length = self.context_lengths[self.rand.integers(0, len(self.context_lengths))]

                        correct_left_context = utils.de_tokenize_words(
                            words[i - context_length:i],
                            whitespaces[i - context_length: i]
                        ).lstrip().lower()[-self.max_length_half:]
                        correct_right_context = " " * whitespaces[i] + utils.de_tokenize_words(
                            words[i + 1: i + context_length + 1],
                            whitespaces[i + 1: i + context_length + 1]
                        ).rstrip().lower()[:self.max_length_half]
                        misspelled_left_context = utils.de_tokenize_words(
                            edited_words[i - context_length:i],
                            whitespaces[i - context_length: i]
                        ).lstrip().lower()[-self.max_length_half:]
                        misspelled_right_context = " " * whitespaces[i] + utils.de_tokenize_words(
                            edited_words[i + 1: i + context_length + 1],
                            whitespaces[i + 1: i + context_length + 1]
                        ).rstrip().lower()[:self.max_length_half]

                        # generate contexts
                        contexts.append((
                            (correct_left_context, correct_right_context),
                            (misspelled_left_context, misspelled_right_context)
                        ))

        context_indices = self.rand.permutation(len(contexts))
        context_idx = 0
        self.samples = []

        for correct in self.misspellings:
            label = self.labels_dict[correct]
            for misspelled in self.misspellings[correct]:
                self.samples.append((correct, misspelled, label, *contexts[context_indices[context_idx]]))
                context_idx = (context_idx + 1) % len(context_indices)

    @property
    def num_labels(self) -> int:
        return len(self.labels_dict)

    def __getitem__(self, idx: int) -> Tuple:
        return self.samples[idx]

    def __len__(self) -> int:
        return min(self.max_samples, len(self.samples))

    @staticmethod
    def get_collate_fn() -> Callable:
        def _join(items: List[Tuple]) -> Tuple[List, ...]:
            return tuple(zip(*items))

        return _join


def train(args: argparse.Namespace) -> None:
    logger = common.get_logger("TRAIN_VECTORIZER")

    assert torch.cuda.is_available(), "need a GPU to train a custom vectorizer, but found none"
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True

    if args.train_jsonl_file:
        train_jsonl_files = []
        with open(args.train_jsonl_file, "r", encoding="utf8") as f:
            for line in f:
                line = line.strip()
                if line == "":
                    continue
                train_jsonl_files.append(os.path.join(os.path.dirname(args.train_jsonl_file), line))
    else:
        train_jsonl_files = args.train_jsonl_files

    if args.val_jsonl_file:
        val_jsonl_files = []
        with open(args.train_jsonl_file, "r", encoding="utf8") as f:
            for line in f:
                line = line.strip()
                if line == "":
                    continue
                val_jsonl_files.append(os.path.join(os.path.dirname(args.val_jsonl_file), line))
    else:
        val_jsonl_files = args.val_jsonl_files

    train_batch_size = 1024
    patience = 3
    epochs_no_improvement = 0
    best_val_loss = float("inf")
    prev_best_val_loss = best_val_loss

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = index.CharTransformer().to(device)

    dictionary = io.dictionary_from_file(args.dictionary)
    train_dataset = VectorizationDataset(
        args.max_context_length,
        args.train_misspelling_files,
        train_jsonl_files,
        dictionary,
        args.seed,
        encoder.max_length,
        args.max_train_samples
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
        args.max_context_length,
        args.val_misspelling_files,
        val_jsonl_files,
        dictionary,
        args.seed,
        encoder.max_length,
        args.max_val_samples
    )
    assert train_dataset.num_labels == val_dataset.num_labels
    logger.info(
        f"Training dataset has {len(train_dataset):,} samples "
        f"for {len(train_dataset.misspellings):,} unique words and "
        f"val dataset has {len(val_dataset):,} samples for {len(val_dataset.misspellings):,} unique words, "
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

    eval_every = len(train_loader) // args.eval_per_epoch

    encoder.train()
    classifier.train()
    for e in tqdm(range(args.epochs), disable=common.disable_tqdm()):
        for i, (correct, misspelled, labels, correct_context, misspelled_context) in tqdm(
                enumerate(train_loader),
                total=len(train_loader),
                leave=False,
                disable=common.disable_tqdm()
        ):
            with torch.cuda.amp.autocast(enabled=grad_scaler.is_enabled()):
                correct_emb = encoder(correct)
                misspelled_emb = encoder(misspelled)

                correct_context_emb = encoder([l + index.WORD_PLACEHOLDER + r for l, r in correct_context])
                misspelled_context_emb = encoder([l + index.WORD_PLACEHOLDER + r for l, r in misspelled_context])

                loss = (
                        clf_loss_fn(
                            classifier(misspelled_emb),
                            torch.tensor(labels, dtype=torch.long, device=misspelled_emb.device)
                        )
                        +
                        emb_loss_fn(misspelled_emb, correct_emb)
                        +
                        emb_loss_fn(misspelled_context_emb, correct_context_emb)
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
                for correct, misspelled, labels, correct_context, misspelled_context in tqdm(
                        val_loader,
                        total=len(val_loader),
                        leave=False,
                        disable=common.disable_tqdm()
                ):
                    with torch.no_grad():
                        correct_emb = encoder(correct)
                        misspelled_emb = encoder(misspelled)

                        correct_context_emb = encoder([l + index.WORD_PLACEHOLDER + r for l, r in correct_context])
                        misspelled_context_emb = encoder(
                            [l + index.WORD_PLACEHOLDER + r for l, r in misspelled_context])

                        loss = (
                                clf_loss_fn(
                                    classifier(misspelled_emb),
                                    torch.tensor(labels, dtype=torch.long, device=misspelled_emb.device)
                                )
                                +
                                emb_loss_fn(misspelled_emb, correct_emb)
                                +
                                emb_loss_fn(misspelled_context_emb, correct_context_emb)
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
    train_file_group = parser.add_mutually_exclusive_group()
    train_file_group.add_argument("--train-jsonl-files", type=str, nargs="+")
    train_file_group.add_argument("--train-jsonl-file", type=str)
    val_file_group = parser.add_mutually_exclusive_group()
    val_file_group.add_argument("--val-jsonl-files", type=str, nargs="+")
    val_file_group.add_argument("--val-jsonl-file", type=str)
    parser.add_argument("--train-misspelling-files", nargs="+", type=str, required=True)
    parser.add_argument("--val-misspelling-files", nargs="+", type=str, required=True)
    parser.add_argument("--max-train-samples", type=int, required=True)
    parser.add_argument("--max-val-samples", type=int, default=10_000)
    parser.add_argument("--max-context-length", type=int, default=3)
    parser.add_argument("--dictionary", type=str, required=True)
    parser.add_argument("--seed", type=int, default=22)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--eval-per-epoch", type=int, default=4)
    parser.add_argument("--out-path", type=str, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
