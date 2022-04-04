import argparse
import collections
import os
from typing import Dict, Tuple, List, Set

from tqdm import tqdm

from gnn_lib.api.utils import tables, save_text_file
from gnn_lib.utils import common, io
from spelling_correction.utils import metrics


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--benchmark-dir", type=str, required=True)
    parser.add_argument("--benchmark-type", choices=["sed_sequence", "sed_words", "sec"], required=True)
    parser.add_argument("--result-dir", type=str, required=True)

    parser.add_argument("--format", choices=["markdown", "latex", "both"], default="both")
    parser.add_argument("--save-dir", type=str, required=True)

    return parser.parse_args()


def evaluate(
        groundtruth_file: str,
        predicted_file: str,
        corrupted_file: str,
        metric_names: Set[str]
) -> Dict[str, float]:
    groundtruths = []
    predictions = []
    corrupted = []
    with open(groundtruth_file, "r", encoding="utf8") as gtf, \
            open(predicted_file, "r", encoding="utf8") as pf, \
            open(corrupted_file, "r", encoding="utf8") as cf:
        for gt, p, c in zip(gtf, pf, cf):
            groundtruths.append(gt.strip())
            predictions.append(p.strip())
            corrupted.append(c.strip())

    assert len(predictions) == len(groundtruths) and len(groundtruths) == len(corrupted)

    results = {}
    for name in metric_names:
        if name == "binary_f1":
            binary_predictions = [int(p) for prediction in predictions for p in prediction.split()]
            binary_labels = [int(l) for label in groundtruths for l in label.split()]
            f1, prec, rec = metrics.binary_f1_prec_rec(binary_predictions, binary_labels)
            results[name] = 100 * f1
        elif name == "word_accuracy":
            word_predictions = [p for prediction in predictions for p in prediction.split()]
            word_groundtruths = [l for label in groundtruths for l in label.split()]
            results[name] = 100 * metrics.accuracy(word_predictions, word_groundtruths)
        elif name == "sequence_accuracy":
            results[name] = 100 * metrics.accuracy(predictions, groundtruths)
        elif name == "mean_normalized_edit_distance":
            results[name] = metrics.mean_normalized_sequence_edit_distance(predictions, groundtruths)
        else:
            raise RuntimeError(f"unknown metric {name}")

    return results


_METRIC_TO_FMT = {
    "sequence_accuracy": ".1f",
    "binary_f1": ".1f",
    "word_accuracy": ".1f",
    "mean_normalized_edit_distance": ".4f"
}

_METRIC_TO_HIGHER_BETTER = {
    "sequence_accuracy": True,
    "binary_f1": True,
    "word_accuracy": True,
    "mean_normalized_edit_distance": False
}

Model = Tuple[str, str]


def get_sed_models_and_metrics(is_sed_words: bool) -> Tuple[Dict[int, List[Model]], Set[str]]:
    dictionary = {
        0: [
            ("aspell", "baseline_aspell"),
            ("jamspell", "baseline_jamspell"),
            ("language_tool", "baseline_languagetool"),
            ("out_of_dictionary", "baseline_ood"),
            ("do_nothing", "baseline_dummy")
        ],
        1: [
            ("neuspell_bert", "baseline_neuspell_bert")
        ],
        2: [
            ("gnn+", "gnn_cliques_wfc"),
            ("gnn++", "gnn_cliques_wfc_dep_gating"),
            ("transformer+", "transformer"),
            ("transformer", "transformer_no_feat")
        ]
    }
    metric_names = {"sequence_accuracy", "binary_f1"}
    if is_sed_words:
        dictionary[2].append(
            ("tokenization_repair+", "tokenization_repair_plus_sed")
        )
        metric_names.add("word_accuracy")
    return dictionary, metric_names


def get_sec_models_and_metrics() -> Tuple[Dict[int, List[Model]], Set[str]]:
    return {
               0: [
                   ("aspell", "baseline_aspell"),
                   ("jamspell", "baseline_jamspell"),
                   ("language_tool", "baseline_languagetool"),
                   ("close_to_dictionary", "baseline_ctd"),
                   ("do_nothing", "baseline_dummy")
               ],
               1: [
                   ("neuspell_bert", "baseline_neuspell_bert")
               ],
               2: [
                   ("transformer", "transformer_sec_nmt"),
                   ("transformer_word", "transformer_sec_words_nmt")
               ],
               3: [
                   ("sed + neuspell_bert", "baseline_neuspell_bert_with_sed"),
                   ("sed + transformer_word", "transformer_sec_words_nmt_with_sed")
               ]
           }, {"sequence_accuracy", "mean_normalized_edit_distance"}


if __name__ == "__main__":
    args = parse_args()
    logger = common.get_logger("EVALUATE_PAPER")

    benchmarks = sorted(io.glob_safe(os.path.join(args.benchmark_dir, "*", "*", "corrupt.txt")))
    benchmarks = [os.path.dirname(b) for b in benchmarks]
    benchmark_groups = [b.split("/")[-2] for b in benchmarks]
    benchmark_splits = [b.split("/")[-1] for b in benchmarks]
    benchmark_names = [f"{group}:{split}" for group, split in zip(benchmark_groups, benchmark_splits)]

    if args.benchmark_type != "sec":
        models, metric_names = get_sed_models_and_metrics(is_sed_words=args.benchmark_type == "sed_words")
    else:
        models, metric_names = get_sec_models_and_metrics()

    horizontal_lines = []
    results = collections.defaultdict(list)
    for model_group in tqdm(sorted(models), total=len(models), desc="evaluating model groups", leave=False):
        horizontal_lines.extend([False] * (len(models[model_group]) - 1))
        horizontal_lines.append(True)
        for model_name, model_file_name in tqdm(
                models[model_group], total=len(models[model_group]),
                desc=f"evaluating models in group {model_group}",
                leave=False
        ):
            model_scores = collections.defaultdict(list)
            for benchmark, benchmark_group, benchmark_split in tqdm(
                    zip(benchmarks, benchmark_groups, benchmark_splits),
                    total=len(benchmarks),
                    desc=f"evaluating model {model_name} from group {model_group} on benchmarks",
                    leave=False
            ):
                benchmark_input = os.path.join(benchmark, "corrupt.txt")
                benchmark_gt = os.path.join(benchmark, "correct.txt")
                model_prediction = os.path.join(
                    args.result_dir, benchmark_group, benchmark_split, f"{model_file_name}.txt"
                )

                if not os.path.exists(model_prediction):
                    for metric_name in metric_names:
                        model_scores[metric_name].append(None)
                    continue

                m = evaluate(
                    benchmark_gt, model_prediction, benchmark_input, metric_names
                )

                for metric_name, score in m.items():
                    model_scores[metric_name].append(score)

            for metric_name, benchmark_scores in model_scores.items():
                results[metric_name].append(
                    [model_name] + benchmark_scores
                )

    for metric_name, data in results.items():
        if len(data) == 0:
            continue

        higher_better = _METRIC_TO_HIGHER_BETTER[metric_name]
        best_scores_per_benchmark = [float("-inf")] * len(data[0])
        best_models_per_benchmark = [set()] * len(data[0])
        for i, model_scores_per_benchmark in enumerate(data):
            for j, benchmark_score in enumerate(model_scores_per_benchmark):
                if benchmark_score is None or j == 0:
                    continue

                if not higher_better:
                    benchmark_score = -benchmark_score

                if benchmark_score == best_scores_per_benchmark[j]:
                    best_models_per_benchmark[j].add(i)
                elif benchmark_score > best_scores_per_benchmark[j]:
                    best_models_per_benchmark[j] = {i}
                    best_scores_per_benchmark[j] = benchmark_score

        bold_cells = set(
            (i, j) for j, best_models in enumerate(best_models_per_benchmark) for i in best_models
        )

        # convert data to strings
        data = [
            line[:1] + [
                f"{score:{_METRIC_TO_FMT[metric_name]}}" if score is not None else "-"
                for score in line[1:]
            ]
            for line in data
        ]

        formats = [args.format] if args.format != "both" else ["markdown", "latex"]
        for fmt in formats:
            results_table = tables.generate_table(
                header=["Model"] + benchmark_names,
                data=data,
                horizontal_lines=horizontal_lines,
                bold_cells=bold_cells,
                fmt=fmt
            )

            path = os.path.join(args.save_dir, f"{metric_name}.{'md' if fmt == 'markdown' else 'tex'}")
            lines = [results_table]
            if fmt == "markdown":
                lines = [f"### {metric_name}"] + lines
            save_text_file(path, lines)
