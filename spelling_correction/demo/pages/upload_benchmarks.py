import os
from io import StringIO

import streamlit as st

from spelling_correction.demo.utils import select_benchmark_group


def show_upload_benchmarks(benchmarks_dir: str):
    st.write("""
    ## Upload your own spell checking benchmarks
    
    The benchmark should be provided as plain text file where each line is one input sequence that should be spell
    checked. The corresponding groundtruth file should also be a plain text file with the same number of lines as 
    the benchmark, where each line is the groundtruth output. Be sure to have the correct groundtruth output format 
    for the benchmark group your benchmark belongs to:
    
    Benchmark group | output format | example
    ---             | --- | ---
    sed_sequence    | 0 or 1 per sequence | this is rwong --> 1
    sed_words       | 0 or 1 per word in sequence (split by whitespace) | this is rwong --> 0 0 1
    sec             | string | this is rwong --> this is wrong
    
    To familiarize yourself with the format for each benchmark group you can look at existing 
    benchmark predictions by evaluating benchmarks on the page **Evaluate benchmarks** 
    and turning on **Show detailed results**.
    
    After you have uploaded your benchmark here, you can run models on it on the page 
    **Run benchmarks** and then evaluate the predictions on the page **Evaluate benchmarks**.
    """)

    benchmark_group = select_benchmark_group()

    st.write("#### Upload benchmark")

    upload_benchmark = st.file_uploader("Upload a benchmark file")

    if not upload_benchmark:
        st.info("Please upload a benchmark file")
        st.stop()

    benchmark_data = StringIO(upload_benchmark.getvalue().decode("utf8")).readlines()

    st.write("###### Benchmark preview (first 20 lines):")
    st.json(benchmark_data[:20])

    st.write("#### Upload groundtruth")

    upload_groundtruth = st.file_uploader("Upload the corresponding groundtruth for the benchmark")

    if not upload_groundtruth:
        st.info("Please upload the groundtruth file for the benchmark")
        st.stop()

    groundtruth_data = StringIO(upload_groundtruth.getvalue().decode("utf8")).readlines()

    st.write("###### Groundtruth preview (first 20 lines):")
    st.json(groundtruth_data[:20])

    if len(groundtruth_data) != len(benchmark_data):
        st.error(f"Expected benchmark and groundtruth to have the same number of lines, but got {len(benchmark_data)} "
                 f"for benchmark and {len(groundtruth_data)} for groundtruth.")
        st.stop()

    st.write("#### Give your benchmark a name")
    a, b = st.columns(2)
    with a:
        benchmark_name = st.text_input("Name of the benchmark")
    with b:
        benchmark_split = st.text_input("Split of the benchmark", placeholder="e.g. test, dev, difficult, easy, ...")

    new_benchmark_dir = os.path.join(benchmarks_dir, benchmark_group, benchmark_name, benchmark_split)

    if benchmark_name == "" or " " in benchmark_name or benchmark_split == "" or " " in benchmark_split:
        st.info("Please enter a valid benchmark name and split without whitespaces")
        st.stop()
    elif os.path.exists(new_benchmark_dir):
        st.error(f"Benchmark with name {benchmark_name} already exits. Please choose a different name.")
        st.stop()

    st.write(f"*Info: Your benchmark will be available as "
             f"`{benchmark_group}/{benchmark_name}/{benchmark_split}/corrupt.txt`, "
             f"the corresponding groundtruth as `{benchmark_group}/{benchmark_name}/{benchmark_split}/correct.txt`*")
    save_benchmark = st.button("Save benchmark now")

    if save_benchmark:
        os.makedirs(new_benchmark_dir)
        with open(os.path.join(new_benchmark_dir, "corrupt.txt"), "w", encoding="utf8") as f:
            f.writelines(benchmark_data)
        with open(os.path.join(new_benchmark_dir, "correct.txt"), "w", encoding="utf8") as f:
            f.writelines(groundtruth_data)

        st.success(
            f"*Your benchmark is now available as "
            f"`{benchmark_group}/{benchmark_name}/{benchmark_split}/corrupt.txt`, "
            f"the corresponding groundtruth as `{benchmark_group}/{benchmark_name}/{benchmark_split}/correct.txt`*"
        )
