import argparse
import collections
import copy
import itertools
import json
import os
import random
from typing import List, Tuple, Callable, Dict, Set

import numpy as np
import omegaconf
import torch
from torch import nn
from torch.backends import cudnn
from torch.utils import data
from tqdm import tqdm

from gnn_lib.api.utils import load_text_file
from gnn_lib.data import utils, index, preprocessing
from gnn_lib.modules import lr_scheduler
from gnn_lib.utils import common, io

logger = common.get_logger("TRAIN_VECTORIZER")


def get_misspelled_contexts(
        jsonl_files: List[str],
        dictionary: Dict[str, int],
        max_samples: int,
        max_length: int,
        context_lengths: List[int],
        max_context_length: int,
        rand: np.random.Generator
) -> List[Tuple]:
    logger.info(f"Creating {max_samples:,} misspelled contexts")
    max_length_half = max_length // 2
    contexts = []
    for file in jsonl_files:
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
                copy.deepcopy(words), doc, 1.0, 1.0, "artificial", rand
            )

            whitespaces = (
                    [False] * max_context_length
                    + [t.whitespace_ == " " for t in doc]
                    + [False] * max_context_length
            )
            words = [""] * max_context_length + words + [""] * max_context_length
            edited_words = [""] * max_context_length + edited_words + [""] * max_context_length

            for i in range(max_context_length, len(words) - max_context_length, max_context_length):
                if words[i] not in dictionary:
                    continue
                if (len(contexts) + 1) % max(1, max_samples // 10) == 0:
                    logger.info(
                        f"Created {len(contexts):,} ({100 * len(contexts) / max_samples:.1f}%) misspelled contexts")
                if len(contexts) >= max_samples:
                    return contexts

                context_length = context_lengths[rand.integers(0, len(context_lengths))]

                correct_left_context = utils.de_tokenize_words(
                    words[i - context_length:i],
                    whitespaces[i - context_length: i]
                ).lstrip().lower()[-max_length_half:]
                correct_right_context = " " * whitespaces[i] + utils.de_tokenize_words(
                    words[i + 1: i + context_length + 1],
                    whitespaces[i + 1: i + context_length + 1]
                ).rstrip().lower()[:max_length_half]
                misspelled_left_context = utils.de_tokenize_words(
                    edited_words[i - context_length:i],
                    whitespaces[i - context_length: i]
                ).lstrip().lower()[-max_length_half:]
                misspelled_right_context = " " * whitespaces[i] + utils.de_tokenize_words(
                    edited_words[i + 1: i + context_length + 1],
                    whitespaces[i + 1: i + context_length + 1]
                ).rstrip().lower()[:max_length_half]

                # generate contexts
                contexts.append((
                    (correct_left_context, correct_right_context),
                    (misspelled_left_context, misspelled_right_context)
                ))

    return contexts


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
        self.rand = np.random.default_rng(seed)

        self.max_samples = max_samples
        self.max_length = max_length
        self.labels_dict = {w: i for i, w in enumerate(sorted(dictionary)) if len(w) <= self.max_length}

        self.misspellings: Dict[str, Set[str]] = collections.defaultdict(set)

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

        self.total_misspellings = sum(len(m) for m in self.misspellings.values())
        # for every (correct, misspelled) pair get a (correct_context, misspelled_context) pair
        contexts: List[Tuple] = get_misspelled_contexts(
            jsonl_files,
            dictionary,
            self.total_misspellings,
            self.max_length,
            self.context_lengths,
            self.max_context_length,
            self.rand
        )

        context_indices = self.rand.permutation(len(contexts))
        context_idx = 0

        self.samples = []
        for correct in self.misspellings:
            label = self.labels_dict[correct]
            for misspelled in self.misspellings[correct]:
                self.samples.append((correct, misspelled, label, *contexts[context_indices[context_idx]]))
                context_idx = (context_idx + 1) % len(contexts)

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
    assert torch.cuda.is_available(), "need a GPU to train a custom vectorizer, but found none"
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True

    if args.train_jsonl_file:
        train_jsonl_files = []
        for file in args.train_jsonl_file:
            dirname = os.path.dirname(file)
            lines = load_text_file(file)
            train_jsonl_files.extend([os.path.join(dirname, line) for line in lines])
    else:
        train_jsonl_files = args.train_jsonl_files

    if args.val_jsonl_file:
        val_jsonl_files = []
        for file in args.val_jsonl_file:
            dirname = os.path.dirname(file)
            lines = load_text_file(file)
            val_jsonl_files.extend([os.path.join(dirname, line) for line in lines])
    else:
        val_jsonl_files = args.val_jsonl_files

    rand = random.Random(args.seed)
    rand.shuffle(train_jsonl_files)
    rand.shuffle(val_jsonl_files)

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
        batch_size=64,
        shuffle=False,
        drop_last=True,
        collate_fn=val_dataset.get_collate_fn()
    )

    clf_loss_fn = torch.nn.CrossEntropyLoss()
    emb_loss_fn = torch.nn.MSELoss()
    grad_scaler = torch.cuda.amp.GradScaler(enabled=args.fp16)

    classifier = nn.Linear(encoder.dim, train_dataset.num_labels).to(device)

    optimizer = torch.optim.AdamW(itertools.chain(encoder.parameters(), classifier.parameters()), lr=1e-3)
    scheduler = lr_scheduler.get_lr_scheduler_from_config(
        omegaconf.DictConfig({"type": "COSINE_WITH_WARMUP"}),
        optimizer,
        len(train_loader) * args.epochs
    )

    eval_every = max(1, len(train_loader) // args.eval_per_epoch)

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
    train_file_group.add_argument("--train-jsonl-file", type=str, nargs="+")
    val_file_group = parser.add_mutually_exclusive_group()
    val_file_group.add_argument("--val-jsonl-files", type=str, nargs="+")
    val_file_group.add_argument("--val-jsonl-file", type=str, nargs="+")
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
    parser.add_argument("--fp16", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
