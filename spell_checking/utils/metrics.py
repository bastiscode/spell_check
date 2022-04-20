import collections
import re
import string
from typing import Optional, Callable, Tuple, List, Any, Dict, Set

import numpy as np
from Levenshtein import distance as ed

from nsc.data import utils


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


def _find_word_boundaries(s: str) -> List[Tuple[int, int]]:
    word_boundary_pattern = re.compile(r"\S+")
    matches = [(match.start(), match.end()) for match in word_boundary_pattern.finditer(s)]
    assert len(matches) == len(s.split())
    return matches


def _get_edited_words(ipt: str, tgt: str) -> Set[int]:
    tgt_word_boundaries = _find_word_boundaries(tgt)
    edit_ops = edit_operations(ipt, tgt, spaces_insert_delete_only=True)
    edited_tgt_indices = set()
    for op_code, ipt_idx, tgt_idx in edit_ops:
        word_boundary_idx = 0
        while word_boundary_idx < len(tgt_word_boundaries):
            word_start, word_end = tgt_word_boundaries[word_boundary_idx]
            if tgt_idx <= word_end:
                break
            word_boundary_idx += 1

        if op_code == "insert" and tgt[tgt_idx] == " ":
            assert word_boundary_idx < len(tgt_word_boundaries) - 1
            edited_tgt_indices.add(word_boundary_idx)
            edited_tgt_indices.add(word_boundary_idx + 1)
        else:
            edited_tgt_indices.add(word_boundary_idx)

    return edited_tgt_indices


def _match_words(pred: str, tgt: str) -> Set[int]:
    sm = difflib.SequenceMatcher(a=pred.split(), b=tgt.split())
    matching_blocks = sm.get_matching_blocks()
    matching_pred_indices = set()
    matching_tgt_indices = set()
    for matching_block in matching_blocks:
        start_pred = matching_block.a
        for idx in range(start_pred, start_pred + matching_block.size):
            matching_pred_indices.add(idx)
        start_tgt = matching_block.b
        for idx in range(start_tgt, start_tgt + matching_block.size):
            matching_tgt_indices.add(idx)
    return matching_pred_indices, matching_tgt_indices


def _group_words(
        ipt: str,
        pred: str,
        matching_in_pred: Set[int]
) -> Set[int]:
    edit_ops = edit_operations(ipt, pred, spaces_insert_delete_only=True)
    ipt_word_boundaries = find_word_boundaries(ipt)
    merged_with_next_indices = set()
    num_spaces_inserted = {}
    for op_code, ipt_idx, pred_idx in edit_ops:
        word_boundary_idx = 0
        while word_boundary_idx < len(ipt_word_boundaries):
            word_start, word_end = ipt_word_boundaries[word_boundary_idx]
            if ipt_idx <= word_end:
                break
            word_boundary_idx += 1

        if op_code == "delete" and ipt[ipt_idx] == " ":
            merged_with_next_indices.add(word_boundary_idx)

        if op_code == "insert" and pred[pred_idx] == " ":
            if word_boundary_idx not in num_spaces_inserted:
                num_spaces_inserted[word_boundary_idx] = 1
            else:
                num_spaces_inserted[word_boundary_idx] += 1

    correct = set()
    ipt_idx = 0
    pred_idx = 0
    while ipt_idx < len(ipt_word_boundaries):
        merged_word = {ipt_idx}
        total_spaces_inserted = num_spaces_inserted.get(ipt_idx, 0)
        while ipt_idx in merged_with_next_indices:
            ipt_idx += 1
            merged_word.add(ipt_idx)
            total_spaces_inserted += num_spaces_inserted.get(ipt_idx, 0)

        # find corresponding words for merged word in pred
        if all(idx in matching_in_pred for idx in range(pred_idx, pred_idx + total_spaces_inserted + 1)):
            correct = correct.union(merged_word)

        ipt_idx += 1
        pred_idx += total_spaces_inserted + 1

    assert ipt_idx == len(ipt_word_boundaries) and pred_idx == len(pred.split())
    return correct


def _correction_tp_fp_fn(pred: str, tgt: str, ipt: str) -> Tuple[int, int, int]:
    misspelled = _get_edited_words(ipt, tgt)
    changed = _get_edited_words(pred, ipt)
    matching_in_pred, restored = _match_words(pred, tgt)
    correct = _group_words(ipt, pred, matching_in_pred)
    tp_indices = misspelled.intersection(restored)
    fn_indices = misspelled.difference(restored)
    fp_indices = changed.difference(correct)
    return len(tp_indices), len(fp_indices), len(fn_indices)


def correction_f1_prec_rec(
        input_sequences: List[str],
        predicted_sequences: List[str],
        target_sequences: List[str]
) -> Tuple[float, float, float]:
    check_same_length(input_sequences, predicted_sequences, target_sequences)
    assert all(
        i.strip() == i and p.strip() == p and t.strip() == t
               for i, p, t in zip(input_sequences, predicted_sequences, target_sequences)
    ), "correction f1 score expects that all sequence contain no leading or trailing whitespaces"

    total_tp = total_fp = total_fn = 0
    for pred, tgt, ipt in zip(predicted_sequences, target_sequences, input_sequences):
        tp, fp, fn = _correction_tp_fp_fn(pred, tgt, ipt)
        total_tp += tp
        total_fp += fp
        total_fn += fn

    return _tp_fp_fn_to_f1_prec_rec(total_tp, total_fp, total_fn)


def is_real_word(word: str, dictionary: Dict[str, int]) -> bool:
    corrupt_word_split = utils.tokenize_words_regex(word)[0]
    return all(c in dictionary or c in string.punctuation for c in corrupt_word_split)
