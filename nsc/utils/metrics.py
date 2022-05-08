import string
from typing import Tuple, List, Any, Dict, Set

import numpy as np

from nsc.data import utils
from nsc.utils import tokenization_repair
from nsc.utils.edit import (
    batch_edit_distance,
    batch_edit_operations,
    batch_match_words,
    find_word_boundaries,
    get_edited_words
)


def check_same_length(*args: Any) -> None:
    if len(args) == 0:
        return
    lengths = [len(args[0])]
    for i in range(1, len(args)):
        lengths.append(len(args[i]))
        assert lengths[0] == lengths[-1], f"expected all arguments to be of same length, got {lengths}"


def _ed(l1: List, l2: List) -> float:
    eds = batch_edit_distance(l1, l2)
    return sum(eds) / max(1, len(eds))


def _ned(l1: List, l2: List) -> float:
    eds = batch_edit_distance(l1, l2)
    neds = [ed / max(len(s), len(t)) for ed, s, t in zip(eds, l1, l2)]
    return sum(neds) / max(1, len(neds))


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


def match_words(preds: List[str], tgts: List[str]) -> Tuple[List[Set[int]], List[Set[int]]]:
    match_pred_indices_list = [set() for _ in range(len(preds))]
    match_tgt_indices_list = [set() for _ in range(len(tgts))]
    for i, matching_indices in enumerate(batch_match_words(preds, tgts)):
        for pred_idx, tgt_idx in matching_indices:
            match_pred_indices_list[i].add(pred_idx)
            match_tgt_indices_list[i].add(tgt_idx)

    return match_pred_indices_list, match_tgt_indices_list


def group_words(
        ipts: List[str],
        preds: List[str],
        matching_in_preds: List[Set[int]]
) -> List[Set[int]]:
    outputs = []
    batch_edit_ops = batch_edit_operations(ipts, preds, spaces_insert_delete_only=True)
    for ipt, pred, matching_in_pred, edit_ops in zip(ipts, preds, matching_in_preds, batch_edit_ops):
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
        outputs.append(correct)

    return outputs


def correction_f1_prec_rec(
        input_sequences: List[str],
        predicted_sequences: List[str],
        target_sequences: List[str]
) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
    check_same_length(input_sequences, predicted_sequences, target_sequences)

    _, misspelled = get_edited_words(input_sequences, target_sequences)
    changed, _ = get_edited_words(input_sequences, predicted_sequences)
    matching_in_pred, restored = match_words(predicted_sequences, target_sequences)
    correct = group_words(input_sequences, predicted_sequences, matching_in_pred)

    f1s = []
    precs = []
    recs = []
    tp = fp = fn = 0
    for mis, res, cha, cor in zip(misspelled, restored, changed, correct):
        tp_indices = mis.intersection(res)
        fn_indices = mis.difference(res)
        fp_indices = cha.difference(cor)
        tp += len(tp_indices)
        fp += len(fp_indices)
        fn += len(fn_indices)

        # if the input sequence has no errors, and we did not predict any false positives, value this as correct
        # (f1 should be 1 and not 0)
        if len(mis) == 0 and len(cha) == 0:
            f1, prec, rec = (1., 1., 1.)
        else:
            f1, prec, rec = _tp_fp_fn_to_f1_prec_rec(len(tp_indices), len(fp_indices), len(fn_indices))
        f1s.append(f1)
        precs.append(prec)
        recs.append(rec)

    return (
        _tp_fp_fn_to_f1_prec_rec(tp, fp, fn),
        (float(np.mean(f1s)), float(np.mean(precs)), float(np.mean(recs)))
    )


def _insertions_and_deletions(repair_ops: List[int]) -> Set[Tuple[int, int]]:
    insertions_and_deletions = set()
    for i, op in enumerate(repair_ops):
        if op != 0:
            insertions_and_deletions.add((i, op))
    return insertions_and_deletions


def tok_rep_f1_prec_rec(
        input_sequences: List[str],
        predicted_sequences: List[str],
        target_sequences: List[str],
        mode: str = "insertions_and_deletions"
) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
    assert mode in {"insertions_and_deletions", "insertions", "deletions"}
    tp = 0
    fp = 0
    fn = 0

    f1s = []
    precs = []
    recs = []

    for pred, gt, ipt in zip(predicted_sequences, target_sequences, input_sequences):
        gt_ops = tokenization_repair.get_whitespace_operations(ipt, gt)
        pred_ops = tokenization_repair.get_whitespace_operations(ipt, pred)
        assert len(gt_ops) == len(pred_ops)

        gt_insertions_and_deletions = _insertions_and_deletions(gt_ops)
        pred_insertions_and_deletions = _insertions_and_deletions(pred_ops)

        if mode == "insertions":
            gt_insertions_and_deletions = set(filter(lambda e: e[1] == 1, gt_insertions_and_deletions))
            pred_insertions_and_deletions = set(filter(lambda e: e[1] == 1, pred_insertions_and_deletions))
        elif mode == "deletions":
            gt_insertions_and_deletions = set(filter(lambda e: e[1] == 2, gt_insertions_and_deletions))
            pred_insertions_and_deletions = set(filter(lambda e: e[1] == 2, pred_insertions_and_deletions))

        tp_ = len(gt_insertions_and_deletions.intersection(pred_insertions_and_deletions))
        fp_ = len(pred_insertions_and_deletions.difference(gt_insertions_and_deletions))
        fn_ = len(gt_insertions_and_deletions.difference(pred_insertions_and_deletions))

        tp += tp_
        fp += fp_
        fn += fn_

        # if there are no groundtruth operations (tp == fp == fn == 0) and we also did not predict any operations,
        # we count this as 1
        if len(gt_insertions_and_deletions) == 0 and len(pred_insertions_and_deletions) == 0:
            scores = (1., 1., 1.)
        else:
            scores = _tp_fp_fn_to_f1_prec_rec(tp, fp, fn)

        f1, prec, rec = scores
        f1s.append(f1)
        precs.append(prec)
        recs.append(rec)

    f1_seq, prec_seq, rec_seq = float(np.mean(f1s)), float(np.mean(precs)), float(np.mean(recs))
    f1_mic, prec_mic, rec_mic = _tp_fp_fn_to_f1_prec_rec(tp, fp, fn)

    return (f1_mic, prec_mic, rec_mic), (f1_seq, prec_seq, rec_seq)


_PUNCTUATION_SET = set(string.punctuation)


def is_real_word(word: str, dictionary: Dict[str, int]) -> bool:
    corrupt_word_split = utils.tokenize_words_regex(word)[0]
    return all(w in dictionary or all(c in _PUNCTUATION_SET for c in w) for w in corrupt_word_split)
