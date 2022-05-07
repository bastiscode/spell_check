import argparse
from typing import Set

from nsc.data.utils import clean_sequence
from nsc.utils import metrics


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "benchmark_type",
        choices=[
            "sed_sequence",
            "sed_words",
            "sec",
            "tokenization_repair"
        ]
    )
    parser.add_argument("in_file", type=str)
    parser.add_argument("gt_file", type=str)
    parser.add_argument("pred_file", type=str)
    return parser.parse_args()


def evaluate(
        corrupted_file: str,
        groundtruth_file: str,
        predicted_file: str,
        metric_names: Set[str]
) -> None:
    groundtruths = []
    predictions = []
    corrupted = []
    with open(groundtruth_file, "r", encoding="utf8") as gtf, \
            open(predicted_file, "r", encoding="utf8") as pf, \
            open(corrupted_file, "r", encoding="utf8") as cf:
        for gt, p, c in zip(gtf, pf, cf):
            groundtruths.append(clean_sequence(gt))
            corrupted.append(clean_sequence(c))
            predictions.append(clean_sequence(p))

    assert len(predictions) == len(groundtruths) == len(corrupted)

    for name in metric_names:
        if name == "binary_f1":
            binary_predictions = [int(p) for prediction in predictions for p in prediction.split()]
            binary_labels = [int(l) for label in groundtruths for l in label.split()]
            f1, prec, rec = metrics.binary_f1_prec_rec(binary_predictions, binary_labels)
            print(f"F1: {100 * f1:.2f} (Precision: {100 * prec:.2f}%, Recall: {100 * rec:.2f}%)")

        elif name == "word_accuracy":
            word_predictions = [int(p) for prediction in predictions for p in prediction.split()]
            word_groundtruths = [int(l) for label in groundtruths for l in label.split()]

            accuracy = metrics.accuracy(word_predictions, word_groundtruths)
            print(f"Word accuracy: {100 * accuracy:.2f}%")

        elif name == "sequence_accuracy":
            accuracy = metrics.accuracy(predictions, groundtruths)
            print(f"Sequence accuracy: {100 * accuracy:.2f}%")

        elif name == "mean_normalized_edit_distance":
            mned = metrics.mean_normalized_sequence_edit_distance(predictions, groundtruths)
            print(f"Mean normalized edit distance: {mned:.4f}")

        elif name == "correction_f1":
            (f1, prec, rec), _ = metrics.correction_f1_prec_rec(corrupted, predictions, groundtruths)
            print(f"Correction F1: {100 * f1:.2f} (Precision: {100 * prec:.2f}%, Recall: {100 * rec:.2f}%)")

        elif name == "tok_rep_f1":
            f1, prec, rec = 0, 0, 0
            print(f"Tokenization repair F1: {100 * f1:.2f} (Precision: {100 * prec:.2f}%, Recall: {100 * rec:.2f}%)")

        else:
            raise RuntimeError(f"unknown metric {name}")


if __name__ == "__main__":
    args = parse_args()
    if args.benchmark_type == "sed_sequence":
        message = "Evaluating spelling error detection (sequence level)"
        metric_names = {"binary_f1", "sequence_accuracy"}
    elif args.benchmark_type == "sed_words":
        message = "Evaluating spelling error detection (word level)"
        metric_names = {"binary_f1", "word_accuracy"}
    elif args.benchmark_type == "sec":
        message = "Evaluating spelling error correction"
        metric_names = {"correction_f1", "mean_normalized_edit_distance"}
    elif args.benchmark_type == "tokenization_repair":
        message = "Evaluating tokenization repair"
        metric_names = {"tok_rep_f1", "sequence_accuracy"}
    else:
        raise RuntimeError("should not happen")
    print(message)
    print("-" * len(message))
    try:
        evaluate(
            corrupted_file=args.in_file,
            groundtruth_file=args.gt_file,
            predicted_file=args.pred_file,
            metric_names=metric_names
        )
    except Exception as e:
        print(f"An exception was thrown during evaluation: '{e}'.\n"
              f"Please make sure that you passed the input, groundtruth and prediction file in the correct order "
              f"and that they have the correct format for the benchmark type you want to evaluate.")
