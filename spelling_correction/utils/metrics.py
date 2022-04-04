import collections
from typing import Optional, Callable, Tuple, List, Any, Set, Dict

import numpy as np
import torch
from Levenshtein import distance as ed


def check_same_length(*args: Any) -> None:
    if len(args) == 0:
        return
    lengths = [len(args[0])]
    for i in range(1, len(args)):
        lengths.append(len(args[i]))
        assert lengths[0] == lengths[-1], f"expected all arguments to be of same length, got {lengths}"


def _ed(l1: List, l2: List) -> float:
    eds = [ed(s, t) for s, t in zip(l1, l2)]
    return sum(eds) / max(1, len(eds))


def _ned(l1: List, l2: List) -> float:
    ned = [ed(s, t) / max(len(s), len(t)) for s, t in zip(l1, l2)]
    return sum(ned) / max(1, len(ned))


def mean_edit_distance(predictions: List[str], targets: List[str]) -> float:
    """

    Average edit distance over all pairs of predicted and target sequences

    :param predictions: list of predicted strings
    :param targets: list of target strings
    :return: mean normalized distance over all prediction-target-pairs
    """
    check_same_length(predictions, targets)
    return _ned(predictions, targets)


def mean_normalized_sequence_edit_distance(predictions: List[str], targets: List[str]) -> float:
    """

    Average normalized edit distance over all pairs of predicted and target sequences:
        ED(A, B) / max(len(A), len(B)) with A and B being strings

    :param predictions: list of predicted strings
    :param targets: list of target strings
    :return: mean normalized distance over all prediction-target-pairs
    """
    check_same_length(predictions, targets)
    return _ned(predictions, targets)


def accuracy(predictions: List, target: List) -> float:
    """

    Percentage of predictions that equal (check with pythons equality operator ==) their targets

    :param predictions: list of predictions
    :param target: list of targets
    :return: accuracy over all prediction-target-pairs
    """
    check_same_length(predictions, target)
    correct = [p == t for p, t in zip(predictions, target)]
    return sum(correct) / max(1, len(correct))


def _tp_fp_fn_to_f1_prec_rec(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = ((2 * precision * recall) / (precision + recall)) if precision + recall > 0 else 0
    return f1, precision, recall


def binary_f1_prec_rec(
        predictions: List[int],
        targets: List[int]
) -> Tuple[float, float, float]:
    """

    Calculates F1, Precision and Recall for given binary predictions and targets.

    :param predictions: list of predictions
    :param targets: list of targets
    :return: f1, precision and recall scores
    """
    check_same_length(predictions, targets)
    assert all(p in {0, 1} for p in predictions)
    assert all(t in {0, 1} for t in targets)

    predictions = np.array(predictions)
    targets = np.array(targets)

    predicted_true = predictions == 1
    tp = (targets[predicted_true] == 1).sum()
    fp = predicted_true.sum() - tp

    predicted_false = np.logical_not(predicted_true)
    tn = (targets[predicted_false] == 0).sum()
    fn = predicted_false.sum() - tn

    assert tp + fp + tn + fn == len(predictions)
    return _tp_fp_fn_to_f1_prec_rec(tp, fp, fn)


def text_f1_prec_rec(sequences: List[str],
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
