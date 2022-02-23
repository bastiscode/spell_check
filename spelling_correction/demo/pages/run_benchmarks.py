import glob
import os
import time
from typing import Any, Dict

import streamlit as st

from gnn_lib import models, tasks
from gnn_lib.data import utils
from gnn_lib.modules.inference import inference_output_to_str
from gnn_lib.utils import common
from spelling_correction.demo.utils import last_n_k_path, select_benchmark_group


def show_run_benchmarks(task: tasks.Task,
                        model: models.Model,
                        benchmarks_dir: str,
                        experiment_name: str,
                        inference_kwargs: Dict[str, Any]):
    st.write("""
        ## Run models on various spelling error detection, correction and tokenization repair benchmarks
        """)

    benchmark_group = select_benchmark_group()

    benchmark_group = os.path.join(benchmarks_dir, benchmark_group)
    results_dir = os.path.join(benchmark_group, "results")

    benchmark_files = glob.glob(f"{benchmark_group}/*/*/corrupt.txt")
    benchmark_files = common.natural_sort(benchmark_files)

    benchmark = st.selectbox("Select an existing benchmark",
                             ["-"] + [last_n_k_path(b, 3, 1) for b in benchmark_files],
                             index=0)

    if benchmark == "-":
        st.info("Please select a benchmark")
        st.stop()

    benchmark_file = os.path.join(benchmark_group, benchmark, "corrupt.txt")

    benchmark_name = st.text_input("Input a name for the benchmark output file", placeholder=f"e.g. {experiment_name}")
    if benchmark_name == "":
        st.info("Please input a benchmark output file name")
        st.stop()

    benchmark_output_file = os.path.join(results_dir, benchmark, f"{benchmark_name}.txt")

    batch_size = st.slider("Batch size for benchmark", min_value=1, max_value=128, value=16)

    sort_benchmark = st.checkbox("Sort benchmark (from long to short sequences, speeds up inference)", value=True)

    run_benchmark_button = st.button("Run benchmark")

    benchmark_dataset, benchmark_loader = utils.get_string_dataset_and_loader(
        benchmark_file, sort_benchmark, batch_size
    )

    st.write(f"*Info: Output file of benchmark will be at `{benchmark_output_file}`*")

    if os.path.exists(benchmark_output_file):
        st.warning(f"*Warning: Output file `{benchmark_output_file}` already exists, if you run the benchmark this "
                   f"file will be overridden*")

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
                **inference_kwargs
            )

            str_outputs = [inference_output_to_str(output) for output in outputs]
            all_outputs.extend(str_outputs)

            sequence_str = f"[{i + 1}/{len(benchmark_loader)}] \n"

            for s, ps in zip(batch, str_outputs):
                sequence_str += f"{s} \u2192 {ps}\n"

            current_sequence.code(sequence_str, language=None)

            end = time.monotonic()
            eta_str.write(f"###### *{common.eta_seconds(end - start, i + 1, len(benchmark_loader))}*")
            benchmark_progress.progress(min(1., (i + 1) / len(benchmark_loader)))
            i += 1

        reordered_outputs = utils.reorder_data(all_outputs, benchmark_dataset.indices)

        with open(benchmark_output_file, "w", encoding="utf8") as of:
            for output in reordered_outputs:
                of.write(str(output) + "\n")

        benchmark_progress.progress(1.0)
        end = time.monotonic()
        runtime = end - start
        eta_str.write(f"###### *Finished in {runtime: .2f} seconds*")

        file_size_kb = os.path.getsize(benchmark_file) / 1024
        kb_per_second = file_size_kb / runtime
        sequences_per_second = len(benchmark_dataset) / runtime
        st.write(f"###### *Processed {sequences_per_second:.2f} sequences/second ({kb_per_second:.2f} kb/second)*")
