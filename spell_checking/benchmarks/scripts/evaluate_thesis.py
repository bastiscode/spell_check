import argparse
import functools
import json
import os
from typing import Dict, Tuple, List, Set, Optional, Callable, Any

from tqdm import tqdm

from nsc.api.utils import tables, save_text_file
from nsc.data.utils import clean_sequence
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
            "sed_words_advanced",
            "sec",
            "sec_advanced",
            "sec_whitespace",
            "sec_spell_checking",
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
) -> Dict[str, Tuple[Any, ...]]:
    groundtruths = []
    predictions = []
    corrupted = []
    with open(groundtruth_file, "r", encoding="utf8") as gtf, \
            open(predicted_file, "r", encoding="utf8") as pf, \
            open(corrupted_file, "r", encoding="utf8") as cf:
        for gt, p, c in zip(gtf, pf, cf):
            groundtruths.append(clean_sequence(gt))
            predictions.append(clean_sequence(p))
            corrupted.append(clean_sequence(c))

    assert len(predictions) == len(groundtruths) == len(corrupted)

    results = {}
    for name in metric_names:
        if name == "binary_f1":
            binary_predictions = [int(p) for prediction in predictions for p in prediction.split()]
            binary_labels = [int(l) for label in groundtruths for l in label.split()]
            results[name] = metrics.binary_f1_prec_rec(binary_predictions, binary_labels)

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
                sum(real_word_detections),
                len(real_word_detections),
                sum(non_word_detections),
                len(non_word_detections)
            )

        elif name == "sequence_accuracy":
            accuracy = metrics.accuracy(predictions, groundtruths)
            results[name] = (accuracy,)

        elif name == "mean_normalized_edit_distance":
            mned = metrics.mean_normalized_sequence_edit_distance(predictions, groundtruths)
            results[name] = (mned,)

        elif name == "correction_f1":
            results[name] = metrics.correction_f1_prec_rec(corrupted, predictions, groundtruths)

        elif name == "bleu":
            from nltk.translate.bleu_score import corpus_bleu
            results[name] = (corpus_bleu(
                [[gt] for gt in groundtruths],
                predictions
            ),)

        else:
            raise RuntimeError(f"unknown metric {name}")

    return results


_METRIC_TO_HIGHER_BETTER = {
    "sequence_accuracy": True,
    "binary_f1": True,
    "word_accuracy": True,
    "mean_normalized_edit_distance": False,
    "correction_f1": True,
    "bleu": True
}

_METRIC_TO_NUM_COLS = {
    "sequence_accuracy": 1,
    "binary_f1": 3,
    "word_accuracy": 3,
    "mean_normalized_edit_distance": 2,
    "correction_f1": 3,
    "bleu": 1
}


def get_metric_fmt_fn(metric_name: str, **metric_kwargs: Any) -> Callable[[Any], List[str]]:
    if metric_name == "sequence_accuracy":
        def _fmt_seq_acc(acc: float, mark_bea: str = "", **fmt_kwargs: Any) -> List[str]:
            return [f"{100 * acc:.2f}{mark_bea}"]

        return _fmt_seq_acc

    elif metric_name == "binary_f1":
        def _fmt_binary_f1(f1: float, prec: float, rec: float, mark_bea: str = "", **fmt_kwargs: Any) -> List[str]:
            return [f"{100 * f1:.2f}{mark_bea}",
                    f"\\footnotesize {100 * prec:.2f}{mark_bea}",
                    f"\\footnotesize {100 * rec:.2f}{mark_bea}"]

        return _fmt_binary_f1

    elif metric_name == "word_accuracy":
        def _fmt_word_acc(
                accuracy: float,
                rw_detections: int,
                rw_total: int,
                nw_detections: int,
                nw_total: int,
                mark_bea: str = "",
                **fmt_kwargs: Any
        ) -> List[str]:
            return [
                f"{100 * accuracy:.2f}{mark_bea}",
                f"\\footnotesize {100 * rw_detections / rw_total:.1f}{mark_bea}",
                f"\\footnotesize {100 * nw_detections / nw_total:.1f}{mark_bea}"
            ]
            # f"$\\frac{{{rw_detections:,}}}{{{rw_total:,}}}$",
            # f"$\\frac{{{nw_detections:,}}}{{{nw_total:,}}}$"]

        return _fmt_word_acc

    elif metric_name == "mean_normalized_edit_distance":
        def _fmt_mned(mned: float, mark_bea: str = "", **fmt_kwargs: Any) -> List[str]:
            if fmt_kwargs["is_baseline"]:
                return ["-", f"\\footnotesize {mned:.4f}{mark_bea}"]
            else:
                baseline_mned = fmt_kwargs["baseline_scores"][0]
                return [f"{100 * (mned / baseline_mned - 1):+.1f}%{mark_bea}", f"\\footnotesize {mned:.4f}{mark_bea}"]

        return _fmt_mned

    elif metric_name == "correction_f1":
        def _fmt_correction_f1(f1: float, prec: float, rec: float, mark_bea: str = "", **fmt_kwargs: Any) -> List[str]:
            return [f"{100 * f1:.2f}{mark_bea}",
                    f"\\footnotesize {100 * prec:.2f}{mark_bea}",
                    f"\\footnotesize {100 * rec:.2f}{mark_bea}"]

        return _fmt_correction_f1

    elif metric_name == "bleu":
        def _fmt_bleu(bleu: float, mark_bea: str = "", **fmt_kwargs: Any) -> List[str]:
            return [f"{bleu:.3f}{mark_bea}"]

        return _fmt_bleu

    else:
        raise RuntimeError("should not happen")


def get_tokenization_repair_models_and_metrics() \
        -> Tuple[Callable[[str], bool], Dict[Tuple[int, str], List[Tuple[str, str]]], Set[str]]:
    dictionary = {
        (0, "baselines"): [
            ("do nothing", "baseline_dummy")
        ],
        (1, "tok_rep"): [
            ("tokenization repair", "tokenization_repair")
        ],
        (2, "tok_rep_advanced"): [
            (r"tokenization repair\textsuperscript{+}", "tokenization_repair_plus_sed"),
            (r"tokenization repair\textsuperscript{+}\textsubscript{\tiny fixed}", "tokenization_repair_plus_fixed"),
            (r"tokenization repair\textsuperscript{++}", "tokenization_repair_plus_sec"),
        ]
    }
    metric_names = {"sequence_accuracy"}
    return lambda _: True, dictionary, metric_names


def get_sed_models_and_metrics(is_sed_words: bool) \
        -> Tuple[Callable[[str], bool], Dict[Tuple[int, str], List[Tuple[str, str]]], Set[str]]:
    dictionary = {
        (0, "baselines"): [
            ("do nothing", "baseline_dummy"),
            ("out of dictionary", "baseline_ood"),
            ("aspell", "baseline_aspell"),
            ("jamspell", "baseline_jamspell"),
            ("languagetool", "baseline_languagetool")
        ],
        (1, "baselines"): [
            ("gector bert", "gector_bert"),
            ("gector xlnet", "gector_xlnet"),
            ("neuspell bert", "baseline_neuspell_bert")
        ],
        (2, "models"): [
            ("transformer", "transformer_no_feat"),
            ("transformer_no_neuspell_no_bea", "transformer_no_feat_no_neuspell_no_bea"),
            ("gnn", "gnn_no_feat"),
            ("gnn_no_neuspell_no_bea", "gnn_no_feat_no_neuspell_no_bea"),
        ],
        (3, "models+"): [
            (r"transformer\textsuperscript{+}", "transformer"),
            (r"transformer\textsuperscript{+}_no_neuspell_no_bea", "transformer_no_neuspell_no_bea"),
            (r"gnn\textsuperscript{+}", "gnn_cliques_wfc"),
            (r"gnn\rlap{\textsuperscript{+}}\textsubscript{\tiny finetuned}", "gnn_cliques_wfc_finetuned"),
            (r"gnn\textsuperscript{+}_no_neuspell_no_bea", "gnn_cliques_wfc_no_neuspell_no_bea")
        ]
    }
    metric_names = {"binary_f1"}
    if is_sed_words:
        metric_names.add("word_accuracy")
    else:
        metric_names.add("sequence_accuracy")
    return _regular_benchmark, dictionary, metric_names


def get_sed_words_advanced_models_and_metrics() \
        -> Tuple[Callable[[str], bool], Dict[Tuple[int, str], List[Tuple[str, str]]], Set[str]]:
    return functools.partial(_regular_benchmark, with_neuspell=False), {
        (0, "models_advanced"): [
            (r"tokenization repair\textsuperscript{+}", "tokenization_repair_plus_sed"),
            (r"tokenization repair\rlap{\textsuperscript{+}}\textsubscript{\tiny fixed}",
             "tokenization_repair_plus_fixed"),
            (r"tokenization repair\textsuperscript{++}", "tokenization_repair_plus_sec")
        ]
    }, {"binary_f1", "sequence_accuracy", "word_accuracy"}


def _regular_benchmark(s: str, with_neuspell: bool = True) -> bool:
    groups = {"bookcorpus", "wikidump"}
    if with_neuspell:
        groups.add("neuspell")
    group = s.split("/")[-2]
    return group in groups


def get_sec_models_and_metrics() -> Tuple[
    Callable[[str], bool], Dict[Tuple[int, str], List[Tuple[str, str]]], Set[str]]:
    return _regular_benchmark, {
        (0, "baselines"): [
            ("do nothing", "baseline_dummy"),
            ("close to dictionary", "baseline_ctd"),
            ("aspell", "baseline_aspell"),
            ("jamspell", "baseline_jamspell"),
            ("languagetool", "baseline_languagetool")
        ],
        (1, "baselines"): [
            ("gector bert", "gector_bert"),
            ("gector xlnet", "gector_xlnet"),
            ("neuspell bert", "baseline_neuspell_bert")
        ],
        (2, "models"): [
            ("transformer", "transformer_sec_nmt"),
            ("transformer_no_neuspell_no_bea", "transformer_sec_nmt_no_neuspell_no_bea"),
            (r"transformer\textsubscript{\tiny finetuned}", "transformer_sec_nmt_finetuned"),
            ("transformer word", "transformer_sec_words_nmt"),
            ("transformer word_no_neuspell_no_bea", "transformer_sec_words_nmt_no_neuspell_no_bea")
        ]
    }, {"mean_normalized_edit_distance", "correction_f1", "bleu"}


def get_sec_advanced_models_and_metrics() \
        -> Tuple[Callable[[str], bool], Dict[Tuple[int, str], List[Tuple[str, str]]], Set[str]]:
    return _regular_benchmark, {
        (0, "baselines"): [
            ("do nothing", "baseline_dummy")
        ],
        (1, "baselines"): [
            (r"gnn\textsuperscript{+} $\rightarrow$ neuspell bert", "gnn_cliques_wfc_plus_baseline_neuspell_bert"),
            (r"gnn\textsuperscript{+} $\rightarrow$ neuspell bert_no_neuspell_no_bea",
             "gnn_cliques_wfc_plus_baseline_neuspell_bert_no_neuspell_no_bea")
        ],
        (2, "models"): [
            (r"gnn\textsuperscript{+} $\rightarrow$ transformer", "gnn_cliques_wfc_plus_transformer_sec_nmt"),
            (r"gnn\rlap{\textsuperscript{+}}\textsubscript{\tiny finetuned} "
             r"$\rightarrow$ transformer\textsubscript{\tiny finetuned}",
             "gnn_cliques_wfc_plus_transformer_sec_nmt_finetuned"),
            (r"gnn\textsuperscript{+} $\rightarrow$ transformer_no_neuspell_no_bea",
             "gnn_cliques_wfc_plus_transformer_sec_nmt_no_neuspell_no_bea"),
            (r"gnn\textsuperscript{+} $\rightarrow$ transformer word",
             "gnn_cliques_wfc_plus_transformer_sec_words_nmt"),
            (r"gnn\textsuperscript{+} $\rightarrow$ transformer word_no_neuspell_no_bea",
             "gnn_cliques_wfc_plus_transformer_sec_words_nmt_no_neuspell_no_bea")
        ],
        (3, "models_advanced"): [
            (r"tokenization repair\textsuperscript{++}", "tokenization_repair_plus_sec")
        ]
    }, {"mean_normalized_edit_distance", "correction_f1", "bleu"}


def get_sec_whitespace_models_and_metrics() \
        -> Tuple[Callable[[str], bool], Dict[Tuple[int, str], List[Tuple[str, str]]], Set[str]]:
    return lambda s: s.split("/")[-2] == "whitespace", {
        (0, "baselines"): [
            ("do nothing", "baseline_dummy")
        ],
        (1, "models"): [
            ("transformer with tokenization repair", "transformer_with_tokenization_repair_sec_nmt"),
            (r"transformer with tokenization repair\textsubscript{\tiny beam}",
             "transformer_with_tokenization_repair_sec_nmt_beam"),
            (r"tokenization repair\textsuperscript{++}", "tokenization_repair_plus_sec")
        ],
        (2, "models_advanced"): [
            (r"tokenization repair $\rightarrow$ gnn\textsuperscript{+} $\rightarrow$ transformer word",
             "tr_plus_gnn_plus_words_nmt"),
            (r"tokenization repair\textsuperscript{+} $\rightarrow$ transformer word", "tr_plus_plus_words_nmt"),
            (r"tokenization repair\rlap{\textsuperscript{+}}\textsubscript{\tiny fixed} $\rightarrow$ transformer word",
             "tr_plus_fixed_plus_words_nmt")
        ]
    }, {"mean_normalized_edit_distance", "correction_f1", "bleu"}


def get_sec_spell_checking_models_and_metrics() \
        -> Tuple[Callable[[str], bool], Dict[Tuple[int, str], List[Tuple[str, str]]], Set[str]]:
    return lambda s: s.split("/")[-2] == "spell_checking", {
        (0, "baselines"): [
            ("do nothing", "baseline_dummy")
        ],
        (1, "default"): [
            ("transformer", "transformer_sec_nmt"),
            ("transformer word", "transformer_sec_words_nmt")
        ],
        (2, "beam"): [
            (r"transformer\textsubscript{\tiny beam}", "transformer_sec_nmt_beam"),
            (r"transformer word\textsubscript{\tiny beam}", "transformer_sec_words_nmt_beam")
        ]
    }, {"mean_normalized_edit_distance", "correction_f1", "bleu"}


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
    elif args.benchmark_type == "sed_words_advanced":
        filter_fn, models, metric_names = get_sed_words_advanced_models_and_metrics()
    elif args.benchmark_type == "sec":
        filter_fn, models, metric_names = get_sec_models_and_metrics()
    elif args.benchmark_type == "sec_advanced":
        filter_fn, models, metric_names = get_sec_advanced_models_and_metrics()
    elif args.benchmark_type == "sec_whitespace":
        filter_fn, models, metric_names = get_sec_whitespace_models_and_metrics()
    elif args.benchmark_type == "sec_spell_checking":
        filter_fn, models, metric_names = get_sec_spell_checking_models_and_metrics()
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

    bea_idx = 0
    for benchmark in benchmarks:
        if benchmark.endswith("neuspell/bea60k"):
            break
        bea_idx += 1
    bea_idx = None if bea_idx == len(benchmarks) else bea_idx
    bea_marks = {}

    for metric_name, model_data in results.items():
        if len(model_data) == 0:
            continue

        group_names = []
        data = []
        for model_group in model_groups:
            group_idx, group_name = model_group
            for model_name, _ in models[model_group]:
                if model_name.endswith("_no_neuspell_no_bea"):
                    continue
                scores = model_data[model_name]
                if bea_idx is not None:
                    if model_name + "_no_neuspell_no_bea" in model_data:
                        bea_marks[model_name] = r"\textsuperscript{*}"
                        scores[bea_idx] = model_data[model_name + "_no_neuspell_no_bea"][bea_idx]
                    else:
                        bea_marks[model_name] = r"\textsuperscript{$\dagger$}"

                data_line = [model_name] + model_data[model_name]
                data.append(data_line)
                group_names.append(group_name)

        num_cols_for_metric = _METRIC_TO_NUM_COLS[metric_name]

        horizontal_lines = []
        for model_group in model_groups:
            horizontal_lines.extend([0] * (sum(not model_name.endswith("_no_neuspell_no_bea")
                                               for model_name, _ in models[model_group]) - 1) + [1])

        higher_better = _METRIC_TO_HIGHER_BETTER[metric_name]
        best_scores_per_benchmark = [float("-inf")] * len(data[0])
        best_models_per_benchmark = [set()] * len(data[0])
        for i, model_scores_per_benchmark in enumerate(data):
            for j, benchmark_scores in enumerate(model_scores_per_benchmark):
                if benchmark_scores is None or j == 0:
                    continue

                benchmark_score = benchmark_scores[0]

                if not higher_better:
                    benchmark_score = -benchmark_score

                if benchmark_score == best_scores_per_benchmark[j]:
                    best_models_per_benchmark[j].add(i)
                elif benchmark_score > best_scores_per_benchmark[j]:
                    best_models_per_benchmark[j] = {i}
                    best_scores_per_benchmark[j] = benchmark_score

        bold_cells = set(
            (i, (j - 1 if j > 0 else 0) * num_cols_for_metric + 1)
            for j, best_models in enumerate(best_models_per_benchmark)
            for i in best_models
        )

        is_mned = metric_name == "mean_normalized_edit_distance"

        if args.benchmark_type == "sed_words_advanced":
            baseline_name = r"tokenization repair\textsuperscript{+}"
        elif args.benchmark_type == "sec_spell_checking":
            baseline_name = "transformer"
        else:
            baseline_name = "do nothing"
        assert baseline_name in model_data and all(s is not None for s in model_data[baseline_name])
        baseline_scores = model_data[baseline_name]

        metric_fmt_fn = get_metric_fmt_fn(metric_name)
        formatted_data = []
        for line, group_name in zip(data, group_names):
            fmt_kwargs = {}
            model_name = line[0]
            if is_mned:
                fmt_kwargs["is_baseline"] = model_name == baseline_name
            formatted_line = [line[0]]
            for i, scores in enumerate(line[1:]):
                if is_mned:
                    fmt_kwargs["baseline_scores"] = baseline_scores[i]
                else:
                    fmt_kwargs.pop("baseline_scores", None)
                if bea_idx is not None and i == bea_idx and group_name not in {"baselines"}:
                    fmt_kwargs["mark_bea"] = bea_marks.get(model_name, "")
                else:
                    fmt_kwargs.pop("mark_bea", None)
                if scores is not None:
                    formatted_line.extend(metric_fmt_fn(*scores, **fmt_kwargs))
                else:
                    formatted_line.extend(["-"] * num_cols_for_metric)
            formatted_data.append(formatted_line)

        formatted_headers = [[""]]
        if len(benchmark_groups) > 1:
            formatted_headers.append([""])

        additional_headers = []
        for b_group, b_split, scores in zip(benchmark_groups, benchmark_splits, baseline_scores):
            formatted_headers[0].extend([b_group] + [""] * (num_cols_for_metric - 1))
            if len(benchmark_groups) > 1:
                formatted_headers[1].extend([b_split] + [""] * (num_cols_for_metric - 1))

            additional_header = []
            if is_mned:
                if not len(additional_headers):
                    additional_headers += [[""]]
                additional_headers[0] += [r"\tiny Improvement", r"\tiny MNED"]
            elif metric_name == "word_accuracy":
                if not len(additional_headers):
                    additional_headers += [[""], [""]]
                additional_headers[0] += [r"\tiny Accuracy", r"\tiny Real word", r"\tiny Non word"]
                _, _, rw_total, _, nw_total = scores
                additional_headers[1] += ["", f"\\tiny {{{rw_total:,}}}", f"\\tiny {{{nw_total:,}}}"]
            elif metric_name in {"correction_f1", "binary_f1"}:
                if not len(additional_headers):
                    additional_headers += [[""]]
                additional_headers[0] += [r"\tiny F\textsubscript{1}", r"\tiny Precision", r"\tiny Recall"]

        for header in additional_headers:
            formatted_headers.append(header)

        formatted_headers = [[s.replace("_", " ") for s in line] for line in formatted_headers]

        formats = [args.format] if args.format != "both" else ["markdown", "latex"]
        for fmt in formats:
            results_table = tables.generate_table(
                headers=formatted_headers,
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
