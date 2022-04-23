import argparse
import collections
import json
import os
from typing import Dict, Tuple, List, Set, Optional, Callable

from tqdm import tqdm

from nsc.api.utils import tables, save_text_file
from nsc.utils import common, io

from spell_checking.utils import metrics


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--benchmark-dir", type=str, required=True)
    parser.add_argument(
        "--benchmark-type",
        choices=[
            "sed_sequence",
            "sed_words",
            "sec",
            "sec_advanced",
            "sec_whitespace",
            "sec_bea_cleaned",
            "tokenization_repair"
        ],
        required=True
    )
    parser.add_argument("--result-dir", type=str, required=True)
    parser.add_argument("--dictionary", type=str, default=None)

    parser.add_argument("--format", choices=["markdown", "latex", "both"], default="both")
    parser.add_argument("--save-dir", type=str, required=True)
    parser.add_argument("--overwrite", type=str, nargs="+", default=None)

    return parser.parse_args()


def evaluate(
        groundtruth_file: str,
        predicted_file: str,
        corrupted_file: str,
        metric_names: Set[str],
        dictionary: Optional[Dict[str, int]] = None
) -> Dict[str, Tuple[float, List[str]]]:
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

    assert len(predictions) == len(groundtruths) == len(corrupted)

    results = {}
    for name in metric_names:
        if name == "binary_f1":
            binary_predictions = [int(p) for prediction in predictions for p in prediction.split()]
            binary_labels = [int(l) for label in groundtruths for l in label.split()]
            f1, prec, rec = metrics.binary_f1_prec_rec(binary_predictions, binary_labels)
            results[name] = (f1, [f"{100 * f1:.2f}", f"\\footnotesize{100 * prec:.1f} {100 * rec:.1f}"])

        elif name == "word_accuracy":
            assert dictionary is not None
            word_predictions = [int(p) for prediction in predictions for p in prediction.split()]
            word_groundtruths = [int(l) for label in groundtruths for l in label.split()]
            words = [w for words in corrupted for w in words.split()]
            assert len(word_predictions) == len(word_groundtruths) == len(words)

            real_word_detections = []
            non_word_detections = []
            for w, p, l in zip(words, word_predictions, word_groundtruths):
                if l == 0:
                    continue

                if metrics.is_real_word(w, dictionary):
                    # real word error
                    real_word_detections.append(p)
                else:
                    # non word
                    non_word_detections.append(p)

            assert len(real_word_detections) + len(non_word_detections) == sum(word_groundtruths)
            accuracy = metrics.accuracy(word_predictions, word_groundtruths)
            results[name] = (
                accuracy,
                [
                    f"{100 * accuracy:.2f}",
                    f"\\footnotesize$\\frac{{{sum(real_word_detections):,}}}{{{len(real_word_detections):,}}}$ "
                    f"$\\frac{{{sum(non_word_detections):,}}}{{{len(non_word_detections):,}}}$"
                ]
            )

        elif name == "sequence_accuracy":
            accuracy = metrics.accuracy(predictions, groundtruths)
            results[name] = (accuracy, [f"{100 * accuracy:.2f}"])

        elif name == "mean_normalized_edit_distance":
            mned = metrics.mean_normalized_sequence_edit_distance(predictions, groundtruths)
            results[name] = (mned, [f"{mned:.4f}"])

        elif name == "correction_f1":
            f1, prec, rec = metrics.correction_f1_prec_rec(corrupted, predictions, groundtruths)
            results[name] = (f1, [f"{100 * f1:.2f}", f"\\footnotesize{100 * prec:.1f} {100 * rec:.1f}"])

        else:
            raise RuntimeError(f"unknown metric {name}")

    return results


_METRIC_TO_NUM_LINES = {
    "sequence_accuracy": 1,
    "binary_f1": 2,
    "word_accuracy": 2,
    "mean_normalized_edit_distance": 1,
    "correction_f1": 2
}

_METRIC_TO_HIGHER_BETTER = {
    "sequence_accuracy": True,
    "binary_f1": True,
    "word_accuracy": True,
    "mean_normalized_edit_distance": False,
    "correction_f1": True
}


def get_tokenization_repair_models_and_metrics() \
        -> Tuple[Callable[[str], bool], Dict[int, List[Tuple[str, str]]], Set[str]]:
    dictionary = {
        0: [
            ("tokenization_repair", "tokenization_repair")
        ],
        1: [
            ("tokenization_repair+", "tokenization_repair_plus_sed"),
            ("tokenization_repair+fixed", "tokenization_repair_plus_fixed"),
            ("tokenization_repair++", "tokenization_repair_plus_sec"),
        ]
    }
    metric_names = {"sequence_accuracy"}
    return lambda _: True, dictionary, metric_names


def get_sed_models_and_metrics(is_sed_words: bool) \
        -> Tuple[Callable[[str], bool], Dict[int, List[Tuple[str, str]]], Set[str]]:
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
            ("transformer", "transformer_no_feat"),
            ("gnn", "gnn_no_feat")
        ],
        3: [
            ("transformer+", "transformer"),
            ("gnn+", "gnn_cliques_wfc")
        ]
    }
    metric_names = {"binary_f1", "sequence_accuracy"}
    if is_sed_words:
        dictionary[3].extend([
            ("tokenization_repair+", "tokenization_repair_plus_sed"),
            ("tokenization_repair+fixed", "tokenization_repair_plus_fixed"),
            ("tokenization_repair++", "tokenization_repair_plus_sec")
        ])
        metric_names.add("word_accuracy")
    return lambda _: True, dictionary, metric_names


def _regular_sec_benchmark(s: str) -> bool:
    split = s.split("/")
    group, split = split[-2], split[-1]
    return group != "whitespace" and split != "bea60k_cleaned"


def get_sec_models_and_metrics() -> Tuple[Callable[[str], bool], Dict[int, List[Tuple[str, str]]], Set[str]]:
    return _regular_sec_benchmark, {
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
        ]
    }, {"sequence_accuracy", "mean_normalized_edit_distance", "correction_f1"}


def get_sec_advanced_models_and_metrics() -> Tuple[Callable[[str], bool], Dict[int, List[Tuple[str, str]]], Set[str]]:
    return _regular_sec_benchmark, {
        1: [
            ("gnn+ --> neuspell_bert", "gnn_cliques_wfc_plus_baseline_neuspell_bert")
        ],
        2: [
            ("gnn+ --> transformer", "gnn_cliques_wfc_plus_transformer_sec_nmt"),
            ("gnn+ --> transformer_word", "gnn_cliques_wfc_plus_transformer_sec_words_nmt")
        ],
        3: [
            ("transformer_with_tokenization_repair", "transformer_with_tokenization_repair_sec_nmt"),
            ("tokenization_repair++", "tokenization_repair_plus_sec")
        ]
    }, {"sequence_accuracy", "mean_normalized_edit_distance", "correction_f1"}


def get_sec_whitespace_models_and_metrics() -> Tuple[Callable[[str], bool], Dict[int, List[Tuple[str, str]]], Set[str]]:
    return lambda s: s.split("/")[-2] == "whitespace", {
        1: [
            ("transformer_with_tokenization_repair", "transformer_with_tokenization_repair_sec_nmt"),
            ("tokenization_repair --> gnn+ --> transformer_word", "tr_plus_gnn_plus_words_nmt"),
            ("tokenization_repair+ --> transformer_word", "tr_plus_plus_words_nmt"),
            ("tokenization_repair+fixed --> transformer_word", "tr_plus_fixed_plus_words_nmt")
        ]
    }, {"sequence_accuracy", "mean_normalized_edit_distance", "correction_f1"}


def get_sec_bea_cleaned_models_and_metrics() \
        -> Tuple[Callable[[str], bool], Dict[int, List[Tuple[str, str]]], Set[str]]:
    def _valid_benchmark(s: str) -> bool:
        split = s.split("/")
        group, split = split[-2], split[-1]
        return group == "neuspell" and split == "bea60k_cleaned"

    return _valid_benchmark, {
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
            ("gnn+ --> neuspell_bert", "gnn_cliques_wfc_plus_baseline_neuspell_bert")
        ],
        4: [
            ("gnn+ --> transformer", "gnn_cliques_wfc_plus_transformer_sec_nmt"),
            ("gnn+ --> transformer_word", "gnn_cliques_wfc_plus_transformer_sec_words_nmt")
        ],
        5: [
            ("transformer_with_tokenization_repair", "transformer_with_tokenization_repair_sec_nmt"),
            ("tokenization_repair++", "tokenization_repair_plus_sec")
        ]
    }, {"sequence_accuracy", "mean_normalized_edit_distance", "correction_f1"}


if __name__ == "__main__":
    args = parse_args()
    logger = common.get_logger("EVALUATE_PAPER")

    if args.dictionary is not None:
        dictionary = io.dictionary_from_file(args.dictionary)
    else:
        dictionary = None

    benchmarks = sorted(io.glob_safe(os.path.join(args.benchmark_dir, "*", "*", "corrupt.txt")))
    benchmarks = [os.path.dirname(b) for b in benchmarks]

    if args.benchmark_type in {"sed_words", "sed_sequence"}:
        filter_fn, models, metric_names = get_sed_models_and_metrics(is_sed_words=args.benchmark_type == "sed_words")
    elif args.benchmark_type == "sec":
        filter_fn, models, metric_names = get_sec_models_and_metrics()
    elif args.benchmark_type == "sec_advanced":
        filter_fn, models, metric_names = get_sec_advanced_models_and_metrics()
    elif args.benchmark_type == "sec_whitespace":
        filter_fn, models, metric_names = get_sec_whitespace_models_and_metrics()
    elif args.benchmark_type == "sec_bea_cleaned":
        filter_fn, models, metric_names = get_sec_bea_cleaned_models_and_metrics()
    else:
        filter_fn, models, metric_names = get_tokenization_repair_models_and_metrics()

    benchmarks = list(filter(filter_fn, benchmarks))
    benchmark_groups = [b.split("/")[-2] for b in benchmarks]
    benchmark_splits = [b.split("/")[-1] for b in benchmarks]

    model_groups = sorted(models)

    results = {
        metric_name: {
            model_name: [None for _ in range(len(benchmarks))]
            for model_group in model_groups for model_name, _ in models[model_group]
        }
        for metric_name in metric_names
    }

    results_file = os.path.join(args.save_dir, "results.json")
    if os.path.exists(results_file):
        with open(results_file, "r", encoding="utf8") as rf:
            results_json = json.load(rf)
            for metric_name in results_json:
                if metric_name not in results:
                    continue
                for model_name, scores in results_json[metric_name].items():
                    if model_name not in results[metric_name]:
                        continue
                    results[metric_name][model_name] = scores

    for model_group in tqdm(model_groups, desc="evaluating model groups", leave=False):
        for model_name, model_file_name in tqdm(
                models[model_group], total=len(models[model_group]),
                desc=f"evaluating models in group {model_group}",
                leave=False
        ):
            for i, (benchmark, benchmark_group, benchmark_split) in tqdm(
                    enumerate(zip(benchmarks, benchmark_groups, benchmark_splits)),
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
                    continue

                filtered_metrics = set()
                for metric_name in metric_names:
                    if (
                            metric_name in results
                            and model_name in results[metric_name]
                            and results[metric_name][model_name][i] is not None
                            and not (args.overwrite and metric_name in args.overwrite)
                    ):
                        continue
                    filtered_metrics.add(metric_name)

                m = evaluate(
                    benchmark_gt, model_prediction, benchmark_input, filtered_metrics, dictionary
                )

                for metric_name, scores in m.items():
                    results[metric_name][model_name][i] = scores

            # save intermediate results after each metric, model pair was processed
            results_dir = os.path.dirname(results_file)
            if results_dir:
                os.makedirs(results_dir, exist_ok=True)
            with open(results_file, "w", encoding="utf8") as rf:
                json.dump(results, rf)

    for metric_name, model_data in results.items():
        if len(model_data) == 0:
            continue

        data = [
            [model_name] + model_data[model_name]
            for model_group in model_groups
            for model_name, _ in models[model_group]
        ]

        num_lines_per_model = _METRIC_TO_NUM_LINES[metric_name]
        assert num_lines_per_model in {1, 2}

        horizontal_lines = []
        for model_group in model_groups:
            horizontal_lines.extend(([0] * (num_lines_per_model - 1) + [0]) * (len(models[model_group]) - 1))
            horizontal_lines.extend([0] * (num_lines_per_model - 1) + [1])

        higher_better = _METRIC_TO_HIGHER_BETTER[metric_name]
        best_scores_per_benchmark = [float("-inf")] * len(data[0])
        best_models_per_benchmark = [set()] * len(data[0])
        for i, model_scores_per_benchmark in enumerate(data):
            for j, benchmark_scores in enumerate(model_scores_per_benchmark):
                if benchmark_scores is None or j == 0:
                    continue

                benchmark_score, _ = benchmark_scores

                if not higher_better:
                    benchmark_score = -benchmark_score

                if benchmark_score == best_scores_per_benchmark[j]:
                    best_models_per_benchmark[j].add(i * num_lines_per_model)
                elif benchmark_score > best_scores_per_benchmark[j]:
                    best_models_per_benchmark[j] = {i * num_lines_per_model}
                    best_scores_per_benchmark[j] = benchmark_score

        bold_cells = set(
            (i, j) for j, best_models in enumerate(best_models_per_benchmark) for i in best_models
        )

        formatted_data = []
        for i, line in enumerate(data):
            for j in range(num_lines_per_model):
                formatted_line = []
                if j == 0:
                    formatted_line.append(line[0])
                else:
                    formatted_line.append("")
                formatted_line.extend([scores[1][j] if scores is not None else "-" for scores in line[1:]])
                formatted_data.append(formatted_line)

        formats = [args.format] if args.format != "both" else ["markdown", "latex"]
        for fmt in formats:
            results_table = tables.generate_table(
                headers=[["Model"] + benchmark_groups, [""] + benchmark_splits],
                data=formatted_data,
                horizontal_lines=horizontal_lines,
                bold_cells=bold_cells,
                fmt=fmt
            )

            path = os.path.join(args.save_dir, f"{metric_name}.{'md' if fmt == 'markdown' else 'tex'}")
            lines = [results_table]
            if fmt == "markdown":
                lines = [f"### {metric_name}"] + lines
            save_text_file(path, lines)
