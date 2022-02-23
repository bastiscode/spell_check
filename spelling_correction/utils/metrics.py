import collections
import os.path
from typing import Optional, Callable, Tuple, List, Any, Set, Dict

import torch
from Levenshtein import distance as ed

from gnn_lib.utils import io


def check_same_length(*args: Any) -> None:
    if len(args) == 0:
        return
    lengths = [len(args[0])]
    for i in range(1, len(args)):
        lengths.append(len(args[i]))
        assert lengths[0] == lengths[-1], f"expected all arguments to be of same length, got {lengths}"


def _ed(l1: List, l2: List) -> float:
    eds = [ed(s, t) for s, t in zip(l1, l2)]
    return sum(eds) / len(eds)


def _ned(l1: List, l2: List) -> float:
    ned = [ed(s, t) / max(len(s), len(t)) for s, t in zip(l1, l2)]
    return sum(ned) / len(ned)


def sequence_edit_distance(sequences: List[str], target_sequences: List[str]) -> float:
    check_same_length(sequences, target_sequences)
    return _ed(sequences, target_sequences)


def normalized_sequence_edit_distance(sequences: List[str], target_sequences: List[str]) -> float:
    """

    Normalized edit distance on strings:
        ED(A, B) / max(len(A), len(B)) with A and B being strings

    :param sequences: list of strings
    :param target_sequences: list of strings
    :return: mean distance over all pairs
    """
    check_same_length(sequences, target_sequences)
    return _ned(sequences, target_sequences)


def accuracy(pred: List,
             target: List) -> float:
    """

    What percentage out of the given sequences match the target sequences:
        1 if A == B else 0 with A and B being Lists of predictions and targets

    :param pred: List of predictions
    :param target: List of targets
    :return: mean accuracy over all prediction-target-pairs
    """
    check_same_length(pred, target)
    correct = [p == t for p, t in zip(pred, target)]
    return sum(correct) / len(correct)


def _tp_fp_fn_to_f1_prec_rec(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = ((2 * precision * recall) / (precision + recall)) if precision + recall > 0 else 0
    return f1, precision, recall


def f1_prec_rec(sequences: List[str],
                target_sequences: List[str],
                split_fn: Optional[Callable] = None) -> Tuple[float, float, float]:
    check_same_length(sequences, target_sequences)

    if split_fn is None:
        def _split_fn(s: str) -> List[str]:
            return s.split()

        split_fn = _split_fn

    tp = 0
    fp = 0
    fn = 0

    for sequence, target_sequence in zip(sequences, target_sequences):
        tokens = split_fn(sequence)
        target_tokens = split_fn(target_sequence)

        tokens_count = collections.Counter(tokens)
        target_tokens_count = collections.Counter(target_tokens)

        tokens_in_common = sum((tokens_count & target_tokens_count).values())

        tp += tokens_in_common
        fp += len(tokens) - tokens_in_common
        fn += len(target_tokens) - tokens_in_common

    return _tp_fp_fn_to_f1_prec_rec(tp, fp, fn)


METRIC_TO_FMT = {
    "f1": ".2f",
    "precision": ".2f",
    "recall": ".2f",
    "binary_f1": ".2f",
    "binary_precision": ".2f",
    "binary_recall": ".2f",
    "sequence_accuracy": ".2f",
    "word_accuracy": ".2f",
    "mned": ".4f",
    "med": ".4f",
    "detection": ".2f",
    "correction": ".2f"
}


def evaluate(groundtruth_file: str,
             predicted_file: str,
             input_file: str,
             metrics: Set[str]) -> Optional[Dict[str, float]]:
    groundtruths = []
    predictions = []
    inputs = []
    with open(groundtruth_file, "r", encoding="utf8") as gtf, \
            open(predicted_file, "r", encoding="utf8") as pf, \
            open(input_file, "r", encoding="utf8") as inf:
        for gt, p, ipt in zip(gtf, pf, inf):
            # filter for "correct" sentences, that are not all capitalized
            # ipt_words = ipt.strip().split()
            # if all(i.isupper() for i in ipt_words):  # or len(ipt_words) < 2 or not ipt_words[0].istitle():
            #     continue
            #

            if len(inputs) > 350:
                break

            groundtruths.append(gt.strip())
            predictions.append(p.strip())
            inputs.append(ipt.strip())

    result = {}
    try:
        if "f1" in metrics or "precision" in metrics or "recall" in metrics:
            f1, precision, recall = f1_prec_rec(predictions, groundtruths)
            if "f1" in metrics:
                result["f1"] = f1 * 100
            if "precision" in metrics:
                result["precision"] = precision * 100
            if "recall" in metrics:
                result["recall"] = recall * 100
        if "sequence_accuracy" in metrics:
            result["sequence_accuracy"] = accuracy(predictions, groundtruths) * 100
        if "binary_f1" in metrics or "binary_precision" in metrics or "binary_recall" in metrics:
            labels_ = []
            preds_ = []
            for ipt, pred, gt in zip(inputs, predictions, groundtruths):
                pred = [int(p) for p in pred.split(" ")]
                lab = [int(l) for l in gt.split(" ")]
                assert all(p in {0, 1} for p in pred)
                assert all(l in {0, 1} for l in lab)
                preds_.extend(pred)
                labels_.extend(lab)

            cls_labels = torch.tensor(labels_, dtype=torch.long)
            cls_predictions = torch.tensor(preds_, dtype=torch.long)

            predicted_pos_indices = cls_predictions == 1
            num_pos = predicted_pos_indices.sum()
            tp = (cls_labels[predicted_pos_indices] == cls_predictions[predicted_pos_indices]).sum()
            fp = num_pos - tp

            predicted_neg_indices = torch.logical_not(cls_predictions)
            num_neg = predicted_neg_indices.sum()
            tn = (cls_labels[predicted_neg_indices] == cls_predictions[predicted_neg_indices]).sum()
            fn = num_neg - tn

            f1, prec, rec = _tp_fp_fn_to_f1_prec_rec(tp.item(), fp.item(), fn.item())
            if "binary_f1" in metrics:
                result["binary_f1"] = f1 * 100
            if "binary_precision" in metrics:
                result["binary_precision"] = prec * 100
            if "binary_recall" in metrics:
                result["binary_recall"] = rec * 100

        if "word_accuracy" in metrics:
            all_p_words = []
            all_g_words = []
            for p, g, ipt in zip(predictions, groundtruths, inputs):
                p_words = p.split()
                g_words = g.split()
                if len(p_words) != len(g_words):
                    raise ValueError(f"Found the number of groundtruth and predicted words to differ "
                                     f"({len(p_words)}, {len(g_words)}) during word accuracy calculation")
                all_p_words.extend(p_words)
                all_g_words.extend(g_words)

            result["word_accuracy"] = accuracy(all_p_words, all_g_words) * 100

        if "mned" in metrics:
            result["mned"] = normalized_sequence_edit_distance(predictions, groundtruths)
        if "med" in metrics:
            result["med"] = sequence_edit_distance(predictions, groundtruths)
    except Exception as e:
        return None
    return result
