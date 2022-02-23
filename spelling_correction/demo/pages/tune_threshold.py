import collections
import glob
import os
import shutil
import tempfile
import time

import pandas as pd
import streamlit as st

from gnn_lib import models, tasks
from gnn_lib.data import utils
from gnn_lib.utils import common
from gnn_lib.modules.inference import inference_output_to_str

from spelling_correction.demo.utils import last_n_k_path, select_benchmark_group
from spelling_correction.utils import metrics as M


def show_tune_threshold(task: tasks.Task,
                        model: models.Model,
                        benchmarks_dir: str,
                        experiment_type: str) -> None:
    if not experiment_type.startswith("SED_"):
        st.warning("Only spelling error detection tasks are supported for threshold tuning")
        st.stop()

    st.write("## Find the best spelling error detection threshold for spelling error detection models on separate "
             "development benchmarks")

    benchmark_group = select_benchmark_group()

    benchmark_group = os.path.join(benchmarks_dir, benchmark_group)

    benchmark_files = glob.glob(f"{benchmark_group}/*/*/corrupt.txt")
    benchmark_files = common.natural_sort(benchmark_files)

    benchmark = st.selectbox("Select a tuning benchmark",
                             ["-"] + [last_n_k_path(b, 3, 1) for b in benchmark_files],
                             index=0)

    if benchmark == "-":
        st.info("Please select a benchmark")
        st.stop()

    benchmark_file = os.path.join(benchmark_group, benchmark, "corrupt.txt")
    groundtruth_file = os.path.join(benchmark_group, benchmark, "correct.txt")

    batch_size = st.slider("Batch size for benchmark", min_value=1, max_value=128, value=16)
    sort_benchmark = st.checkbox("Sort benchmark (from long to short sequences, speeds up inference)", value=True)
    threshold_step = st.selectbox("Step size between different thresholds",
                                  [0.01, 0.025, 0.05, 0.1, 0.25, 0.5],
                                  index=2)
    thresholds = [round(0 + threshold_step, 3)]
    while round(thresholds[-1] + threshold_step, 3) < 1:
        thresholds.append(round(thresholds[-1] + threshold_step, 3))

    st.code(f"Thresholds: {thresholds}")

    metric = st.selectbox(
        "Select metric to optimize",
        options=[
            "binary_f1",
            "sequence_accuracy"
        ]
    )

    run_benchmark_button = st.button("Tune threshold on benchmark")

    benchmark_dataset, benchmark_loader = utils.get_string_dataset_and_loader(
        benchmark_file, sort_benchmark, batch_size
    )

    if run_benchmark_button:
        benchmark_progress = st.progress(0.0)
        eta_str = st.empty()
        current_sequence = st.empty()
        start = time.monotonic()

        all_outputs = []
        for i, (batch, _) in enumerate(benchmark_loader):
            outputs = task.inference(
                model,
                batch,
                **{"return_logits": True}
            )

            all_outputs.extend(outputs)

            sequence_str = f"[{i + 1}/{len(benchmark_loader)}] \n"

            for s in batch:
                sequence_str += f"{s}\n"

            current_sequence.code(sequence_str, language=None)

            end = time.monotonic()
            eta_str.write(f"###### *{common.eta_seconds(end - start, i + 1, len(benchmark_loader))}*")
            benchmark_progress.progress(min(1., (i + 1) / len(benchmark_loader)))
            i += 1

        reordered_outputs = utils.reorder_data(all_outputs, benchmark_dataset.indices)

        temp_dir = tempfile.mkdtemp()
        thresholds_and_scores = []
        with st.spinner("Calculating scores for different thresholds"):
            for t in thresholds:
                t_predictions_file = os.path.join(temp_dir, f"{t}_outputs")
                with open(t_predictions_file, "w", encoding="utf8") as of:
                    for output in reordered_outputs:
                        if isinstance(output[0], list):
                            assert all(len(o) == 2 for o in output)
                            pred = [int(o[1] >= t) for o in output]
                        else:
                            assert len(output) == 2
                            pred = int(output[1]) >= t
                        of.write(inference_output_to_str(pred) + "\n")

                results = M.evaluate(groundtruth_file, t_predictions_file, benchmark_file, {metric})
                thresholds_and_scores.append((t, results[metric]))

            shutil.rmtree(temp_dir)

            thresholds_and_scores = sorted(thresholds_and_scores, key=lambda e: -e[1])

        table_data = collections.defaultdict(dict)
        format_str = "{:" + M.METRIC_TO_FMT[metric] + "}"
        table_data[metric] = {str(t): format_str.format(score) for t, score in thresholds_and_scores}
        st.dataframe(pd.DataFrame.from_dict(table_data))
