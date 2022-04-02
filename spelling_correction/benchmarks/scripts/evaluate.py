import argparse
import os

from tabulate import tabulate

from gnn_lib.utils import common
from gnn_lib.api import tables
from spelling_correction.utils import metrics as M

logger = common.get_logger("EVALUATE_BENCHMARKS")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--groundtruths", type=str, nargs="+", required=True)
    parser.add_argument("--inputs", type=str, nargs="+", required=True)
    parser.add_argument("--predictions", type=str, nargs="+", required=True)
    parser.add_argument("--metrics", nargs="+",
                        choices=[
                            "sequence_accuracy",
                            "word_accuracy",
                            "mned",
                            "med",
                            "f1",
                            "precision",
                            "recall",
                            "detection",
                            "correction",
                            "binary_f1",
                            "binary_precision",
                            "binary_recall"
                        ],
                        required=True)
    parser.add_argument("--dictionary", type=str, default=None)
    parser.add_argument("--save-markdown-dir", type=str, default=None)
    parser.add_argument("--save-latex-dir", type=str, default=None)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    groundtruths = sorted(args.groundtruths)
    inputs = sorted(args.inputs)
    assert len(groundtruths) == len(inputs), \
        f"expected to have the same number of groundtruth and input files, " \
        f"but got {len(groundtruths)} and {len(inputs)}"

    for groundtruth_file, input_file in zip(groundtruths, inputs):
        groundtruth_path = groundtruth_file.split("/")
        input_path = input_file.split("/")
        assert len(groundtruth_path) >= 3 and groundtruth_path[-1] == "correct.txt", \
            "expected groundtruths to point to files with paths of the format" \
            " some_path/<benchmark_name>/<benchmark_split>/correct.txt"
        assert len(input_path) >= 3 and input_path[-1] == "corrupt.txt", \
            "expected inputs to point to files with paths of the format" \
            " some_path/<benchmark_name>/<benchmark_split>/corrupt.txt"
        assert groundtruth_path[-3] == input_path[-3] and groundtruth_path[-2] == input_path[-2], \
            f"groundtruth and input seem to belong to different benchmarks ({groundtruth_path} <--> {input_path})"

        benchmark_name = groundtruth_path[-3]
        benchmark_split = groundtruth_path[-2]

        benchmark_results = []

        for predicted_file in args.predictions:
            predicted_path = predicted_file.split("/")
            assert len(predicted_path) >= 3, "expected predictions to point to files with paths of the format" \
                                             " some_path/<benchmark_name>/<benchmark_split>/<model_name>.txt"

            if predicted_path[-3] != benchmark_name or predicted_path[-2] != benchmark_split:
                continue

            model_name, _ = os.path.splitext(predicted_path[-1])

            results = M.evaluate(groundtruth_file, predicted_file, input_file, args.metrics)
            if results is None:
                continue

            benchmark_results.append([model_name] + [results[metric] for metric in args.metrics])

        if len(benchmark_results) == 0:
            continue

        benchmark_results = sorted(benchmark_results, key=lambda r: r[1], reverse=True)

        results_table = tabulate(benchmark_results,
                                 colalign=["left"] + ["decimal"] * len(args.metrics),
                                 numalign="decimal",
                                 floatfmt=[""] + [M.METRIC_TO_FMT[m] for m in args.metrics],
                                 headers=["model"] + args.metrics,
                                 tablefmt="pipe")

        logger.info(f"Benchmark: {benchmark_name}, split: {benchmark_split}:\n{results_table}\n")

        if args.save_markdown_dir is not None:
            os.makedirs(args.save_markdown_dir, exist_ok=True)
            path = os.path.join(args.save_markdown_dir, f"{benchmark_name}_{benchmark_split}.md")
            # TODO: implement this

        if args.save_latex_dir is not None:
            os.makedirs(args.save_latex_dir, exist_ok=True)
            path = os.path.join(args.save_latex_dir, f"{benchmark_name}_{benchmark_split}.tex")
            # TODO: implement this
