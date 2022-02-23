import functools
import glob
import os
from collections import Counter
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import altair as alt
import streamlit as st

from gnn_lib.utils import common

from spelling_correction.demo.utils import last_n_k_path, select_benchmark_group


@st.cache(show_spinner=False)
def load_data(benchmark_ipt: str, benchmark_gt: str, models: List[str]) -> \
        Tuple[Dict[str, List[str]], List[str], List[str]]:
    with open(benchmark_ipt, "r", encoding="utf8") as inf:
        inputs = [line.strip() for line in inf]

    with open(benchmark_gt, "r", encoding="utf8") as inf:
        gts = [line.strip() for line in inf]

    pred = {}
    for i, model in enumerate(models):
        with open(model, "r", encoding="utf8") as inf:
            pred[model] = [line.strip() for line in inf]

    return pred, inputs, gts


def show_evaluation_game(benchmarks_dir: str) -> None:
    st.write("### Evaluation game")

    state = st.session_state
    rand = np.random.default_rng()

    if "game_models" not in state:
        benchmark_group = select_benchmark_group()

        benchmark_group = os.path.join(benchmarks_dir, benchmark_group)

        benchmarks = glob.glob(f"{benchmark_group}/*/*/correct.txt")
        benchmarks = common.natural_sort(benchmarks)
        benchmark = st.selectbox("Select a benchmark",
                                 ["-"] + [last_n_k_path(b, 3, 1) for b in benchmarks],
                                 index=0)
        if benchmark == "-":
            st.info("Select a benchmark")
            st.stop()

        benchmark_gt = os.path.join(benchmark_group, benchmark, "correct.txt")
        benchmark_ipt = os.path.join(benchmark_group, benchmark, "corrupt.txt")

        predictions = glob.glob(os.path.join(benchmark_group, "results", benchmark, "*.txt"))
        if len(predictions) == 0:
            st.warning(f"Could not find any predictions for benchmark {benchmark}.")
            st.stop()

        models = st.multiselect("Select models",
                                predictions,
                                format_func=lambda s: os.path.splitext(last_n_k_path(s, n=1))[0])
        if len(models) != 2:
            st.info("Select exactly two models to play with")
            st.stop()

        number_of_examples = st.slider(
            "Select the number of rounds to play", min_value=1, max_value=100, value=10, step=1
        )

        if st.button("Start game"):
            state.game_benchmark_gt = benchmark_gt
            state.game_benchmark_ipt = benchmark_ipt
            state.game_models = models
            state.game_rounds = number_of_examples
            st.experimental_rerun()
        st.stop()

    else:
        st.write(f"You are currently playing the evaluation game with models "
                 f"**{os.path.splitext(last_n_k_path(state.game_models[0], n=1))[0]}** and "
                 f"**{os.path.splitext(last_n_k_path(state.game_models[1], n=1))[0]}** "
                 f"on benchmark **{last_n_k_path(state.game_benchmark_gt, n=3, k=1)}**.")
        st.write(f"*To cancel or restart the game navigate to a different page on the sidebar and then come "
                 f"back again.*")

    pred, inputs, gts = load_data(state.game_benchmark_ipt, state.game_benchmark_gt, state.game_models)

    if "game_indices" not in state:
        with st.spinner("Preparing game..."):
            # samples where groundtruth != input are 4 times more likely to be selected (leads to more interesting
            # games)
            indices_p = [4 * (ipt != gt) + int(ipt == gt) for ipt, gt in zip(inputs, gts)]
            indices_p = np.array(indices_p) / sum(indices_p)
            indices = rand.choice(list(range(len(inputs))), state.game_rounds, p=indices_p)
            state.game_example_nr = 0
            state.game_indices = indices
            state.game_points = Counter({model: 0 for model in state.game_models})
            state.game_chosen = []

    st.write("---")

    game_finished = state.game_example_nr >= len(state.game_indices)
    if not game_finished:
        st.write("##### Choose the model prediction you think is the best for the given input")

    st.write(f"######  [Example {min(state.game_example_nr + 1, len(state.game_indices))}/{len(state.game_indices)}] "
             f"{len(state.game_indices) - state.game_example_nr} to go")
    st.progress(state.game_example_nr / len(state.game_indices))

    if game_finished:
        st.write(f"#### Game finished!")

        game_points = {
            "Models": [],
            "Points": []
        }
        for model in state.game_models:
            game_points["Models"].append(os.path.splitext(last_n_k_path(model, n=1))[0])
            game_points["Points"].append(state.game_points[model])

        game_points["Models"].extend(["both good", "both bad"])
        game_points["Points"].extend([state.game_points.get("both_good", 0), state.game_points.get("both_bad", 0)])

        df = pd.DataFrame(game_points)
        chart = alt.Chart(df, title="Point distribution", width=500, height=500).mark_bar().encode(
            x=alt.X(
                "Models",
                title="Models",
                axis=alt.Axis(labelAngle=30, labelFontWeight=800, labelFontSize=16)
            ),
            y=alt.Y("Points", axis=alt.Axis(tickMinStep=1)),
            color="Points",
            tooltip=["Points"]
        ).configure_title(fontSize=20).interactive()
        st.altair_chart(chart)

        model_1_name = os.path.splitext(last_n_k_path(state.game_models[0], n=1))[0]
        model_2_name = os.path.splitext(last_n_k_path(state.game_models[1], n=1))[0]

        with st.expander("Game recap"):
            for i, idx in enumerate(state.game_indices):
                st.write(f"**Example {i + 1}**")

                st.write(f"""
                    Input | {inputs[idx]}
                    --- | ---
                    **Groundtruth**    | **{gts[idx]}**
                    {model_1_name} | {pred[state.game_models[0]][idx]}
                    {model_2_name} | {pred[state.game_models[1]][idx]}
                    Chosen         | {state.game_chosen[i]}
                """)
                if i < len(state.game_indices) - 1:
                    st.write("---")
                else:
                    st.write("")
        st.stop()

    idx = state.game_indices[state.game_example_nr]
    st.write(f"###### Input\n\n{inputs[idx]}")
    st.write(f"###### Model predictions")
    # swap models randomly so user has no idea which is which
    for model in (state.game_models if rand.random() < 0.5 else reversed(state.game_models)):
        clicked = st.button(pred[model][idx], key=f"{model}_{idx}")
        if clicked:
            state.game_points[model] += 1
            state.game_example_nr += 1
            state.game_chosen.append(os.path.splitext(last_n_k_path(model, n=1))[0])
            st.experimental_rerun()

    st.write("###### Other options")
    if st.button("Both are good", key=f"both_good_{idx}"):
        state.game_points["both_good"] += 1
        state.game_example_nr += 1
        state.game_chosen.append("both good")
        st.experimental_rerun()

    if st.button("Both are bad", key=f"both_bad_{idx}"):
        state.game_points["both_bad"] += 1
        state.game_example_nr += 1
        state.game_chosen.append("both bad")
        st.experimental_rerun()
