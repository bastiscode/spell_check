import collections
import functools
import glob
import os
from typing import Dict

import altair as alt
import Levenshtein
import pandas as pd
import streamlit as st

from gnn_lib.utils import common

from spelling_correction.utils import metrics as M
from spelling_correction.demo.utils import last_n_k_path, select_benchmark_group


def show_evaluate_benchmarks(benchmarks_dir: str):
    st.write("""
    ## Evaluate various spelling error detection, correction and tokenization repair benchmarks
    """)

    benchmark_group = select_benchmark_group()

    benchmark_group = os.path.join(benchmarks_dir, benchmark_group)
    results_dir = os.path.join(benchmark_group, "results")

    benchmark_gts = glob.glob(f"{benchmark_group}/*/*/correct.txt")
    benchmark_gts = common.natural_sort(benchmark_gts)

    benchmark = st.selectbox("Select an existing benchmark groundtruth",
                                ["-"] + [last_n_k_path(b, 3, 1) for b in benchmark_gts],
                                index=0)

    if benchmark == "-":
        st.info("Please select a benchmark groundtruth to evaluate")
        st.stop()

    benchmark_gt = os.path.join(benchmark_group, benchmark, "correct.txt")
    benchmark_input = os.path.join(benchmark_group, benchmark, "corrupt.txt")

    benchmark_pred_files = glob.glob(os.path.join(results_dir, benchmark, "*.txt"))
    benchmark_pred_files = common.natural_sort(benchmark_pred_files)

    benchmark_pred_files = st.multiselect("Select existing benchmark predictions",
                                          benchmark_pred_files,
                                          format_func=lambda s: os.path.splitext(last_n_k_path(s, n=1))[0])

    if len(benchmark_pred_files) == 0:
        st.info("Please select a predictions file")
        st.stop()

    metrics = st.multiselect("Metrics",
                             options=["sequence_accuracy",
                                      "word_accuracy",
                                      "mned",
                                      "med",
                                      "f1",
                                      "precision",
                                      "recall"])
    if len(metrics) == 0:
        st.warning("Please select at least one metric before running the evaluation")
        st.stop()

    show_details = st.checkbox("Show detailed results")

    run_evaluation_button = st.button("Run evaluation")

    if run_evaluation_button:
        table_data = collections.defaultdict(dict)
        predicted_files_by_model = {}

        for benchmark_pred in benchmark_pred_files:
            model_name = os.path.splitext(os.path.basename(benchmark_pred))[0]

            results = M.evaluate(groundtruth_file=benchmark_gt,
                                 predicted_file=benchmark_pred,
                                 input_file=benchmark_input,
                                 metrics=set(metrics))
            if results is None:
                st.warning("Unexpected error during evaluation")
                st.stop()

            for metric, value in results.items():
                format_str = "{:" + M.METRIC_TO_FMT[metric] + "}"
                table_data[metric].update({model_name: format_str.format(value)})

            predicted_files_by_model[model_name] = benchmark_pred

        st.write(f"### Results on benchmark {last_n_k_path(benchmark_gt, 3)}:")
        st.dataframe(pd.DataFrame.from_dict(table_data))

        if show_details:
            with st.expander("Model comparison"):
                show_model_comparison(groundtruth_file=benchmark_gt,
                                      predicted_files_by_model=predicted_files_by_model,
                                      corrupted_file=benchmark_input)

            for key in predicted_files_by_model.keys():
                with st.expander(f"Details for model {key}"):
                    predicted_file = predicted_files_by_model[key]

                    show_details_of_model_predictions(groundtruth_file=benchmark_gt,
                                                      predicted_file=predicted_file,
                                                      corrupted_file=benchmark_input)


def _pad_str(s: str, length: int) -> str:
    return s if len(s) >= length else s + " " * (length - len(s))


def show_model_comparison(groundtruth_file: str,
                          predicted_files_by_model: Dict[str, str],
                          corrupted_file: str) -> None:
    predicted_files = {}
    for model, file in predicted_files_by_model.items():
        predicted_files[model] = open(file, "r", encoding="utf8")

    with open(groundtruth_file, "r", encoding="utf8") as gtf, open(corrupted_file, "r", encoding="utf8") as cf:
        models = sorted(predicted_files_by_model.keys())
        pred_files = [predicted_files[model] for model in models]
        pad_to = max(max(len(m) + 2 for m in models), 13)
        input_padded = _pad_str("Input:", pad_to)
        gt_padded = _pad_str("Groundtruth:", pad_to)
        models_padded = [_pad_str(f"{m}:", pad_to) for m in models]
        output_string = ""
        for i, lines in enumerate(zip(cf, gtf, *pred_files)):
            output_string += f"\nSample {i + 1}:"
            for j, line in enumerate(lines):
                line = line.strip()
                output_string += "\n"
                if j == 0:
                    output_string += input_padded
                elif j == 1:
                    output_string += gt_padded
                else:
                    output_string += models_padded[j - 2]
                output_string += line
            output_string += "\n"

        st.code(output_string, language=None)

    for file in predicted_files.values():
        file.close()


def show_details_of_model_predictions(groundtruth_file: str,
                                      predicted_file: str,
                                      corrupted_file: str):
    with open(groundtruth_file, "r", encoding="utf8") as gtf, \
            open(predicted_file, "r", encoding="utf8") as pf, \
            open(corrupted_file, "r", encoding="utf8") as cf:
        lengths = []
        equal = []
        edit_distances = []

        errors = []

        for i, (gt, p, c) in enumerate(zip(gtf, pf, cf)):
            eq = gt == p
            equal.append(eq)

            if not eq:
                errors.append((gt, p, c, i))

            lengths.append(len(c))
            edit_distances.append(Levenshtein.distance(gt, c))

        df = pd.DataFrame({"sequence length": lengths,
                           "edit distance": edit_distances,
                           "correct": equal})
        chart = alt.Chart(df, title="Correct predictions by sequence length of corrupted input").mark_bar().encode(
            x=alt.X("sequence length:Q", title="sequence length (binned)", bin=alt.Bin(step=5)),
            y="count()",
            color="correct",
            tooltip=["count()", "sequence length", "correct"]
        ).interactive()

        st.altair_chart(chart, use_container_width=True)

        chart = alt.Chart(df,
                          title="Correct predictions by edit distance "
                                "between corrupted input and groundtruth").mark_bar().encode(
            x=alt.X("edit distance:Q", title="Edit distance", bin=alt.Bin(step=1)),
            y="count()",
            color="correct",
            tooltip=["count()", "edit distance", "correct"]
        ).interactive()

        st.altair_chart(chart, use_container_width=True)

    st.write("### Erroneous predictions")
    st.write("")

    detail_string = ""
    for gt, p, c, idx in errors:
        gt = gt.strip()
        p = p.strip()
        c = c.strip()
        detail_string += f"\nSample {idx + 1}:\nInput:       {c}\nGroundtruth: {gt}\nPrediction:  {p}\n"

    st.code(detail_string, language=None)
