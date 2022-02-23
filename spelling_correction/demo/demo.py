import argparse
import os
from typing import Tuple, Any

import omegaconf
import streamlit as st
import torch

from gnn_lib import models
from gnn_lib import tasks
from gnn_lib.utils import config

from spelling_correction.demo.pages.evaluate_benchmarks import show_evaluate_benchmarks
from spelling_correction.demo.pages.evaluation_game import show_evaluation_game
from spelling_correction.demo.pages.home import show_home
from spelling_correction.demo.pages.inference import show_inference
from spelling_correction.demo.pages.info import show_info
from spelling_correction.demo.pages.run_benchmarks import show_run_benchmarks
from spelling_correction.demo.pages.upload_benchmarks import show_upload_benchmarks
from spelling_correction.demo.pages.variants import show_dataset_variant
from spelling_correction.demo.pages.tune_threshold import show_tune_threshold
from spelling_correction.demo.sidebar import show_sidebar
from spelling_correction.demo.utils import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--experiments",
                        required=True,
                        help="Path to the experiments dir")
    parser.add_argument("-d", "--data",
                        required=True,
                        help="Path to data dir")
    parser.add_argument("-c", "--configs",
                        required=True,
                        help="Path to configs dir")
    parser.add_argument("-bt", "--test-benchmarks",
                        required=True,
                        help="Path to the benchmarks dir")
    parser.add_argument("-bd", "--dev-benchmarks",
                        required=True,
                        help="Path to the benchmarks dir")
    return parser.parse_args()


@st.cache(allow_output_mutation=True, hash_funcs={tasks.Task: lambda _: None, models.Model: lambda _: None})
def load_experiment(
        experiment: str,
        device: torch.device,
        **kwargs: Any) -> Tuple[config.TrainConfig, tasks.Task, models.Model]:
    cfg: config.TrainConfig = load_config(experiment, kwargs.get("override_env_vars"))

    task = tasks.get_task(
        checkpoint_dir=os.path.join(experiment, "checkpoints"),
        variant_cfg=cfg.variant,
        seed=cfg.seed
    )
    sample_g, _ = task.generate_sample_inputs(2)
    model = task.get_model(
        sample_g=sample_g,
        cfg=cfg.model,
        device=device
    )
    task.load_best(model)
    return cfg, task, model


def run_demo(args: argparse.Namespace) -> None:
    st.markdown(
        f"""
        <style>
            .reportview-container .main .block-container{{
                max-width: {1600}px;
            }}
        </style>
        """,
        unsafe_allow_html=True
    )

    st.write(
        """
        # Spell checking with Graph Neural Networks
        ###### *by Sebastian Walter, last updated on 12/2021*
        ---
        """
    )

    # override some env variables
    override_env_vars = {
        "GNN_LIB_DATA_DIR": args.data,
        "GNN_LIB_EXPERIMENT_DIR": args.experiments,
        "GNN_LIB_CONFIG_DIR": args.configs
    }

    page, experiment, device, inference_kwargs = show_sidebar(
        experiment_dir=args.experiments,
        override_env_vars=override_env_vars
    )
    if st.session_state.get("page") is None or st.session_state.page != page:
        # clear state on page switch
        st.session_state.clear()
        st.session_state.page = page

    if experiment is not None:
        cfg, task, model = load_experiment(experiment=experiment, device=device, override_env_vars=override_env_vars)
        omegaconf.OmegaConf.resolve(cfg)
    else:
        cfg = task = model = None

    if page == "Data variants":
        if experiment is None:
            st.warning("Please select an experiment first")
            st.stop()
        show_dataset_variant(task.variant)
    elif page == "Home":
        show_home()
    elif page == "Inference":
        if experiment is None:
            st.warning("Please select an experiment first")
            st.stop()
        show_inference(task, model, inference_kwargs)
    elif page == "Info":
        if experiment is None:
            st.warning("Please select an experiment first")
            st.stop()
        show_info(model, cfg)
    elif page == "Evaluate benchmarks":
        show_evaluate_benchmarks(args.test_benchmarks)
    elif page == "Upload benchmarks":
        show_upload_benchmarks(args.test_benchmarks)
    elif page == "Run benchmarks":
        if experiment is None:
            st.warning("Please select an experiment first")
            st.stop()
        show_run_benchmarks(task, model, args.test_benchmarks, cfg.experiment_name, inference_kwargs)
    elif page == "Evaluation game":
        show_evaluation_game(args.test_benchmarks)
    elif page == "Tune threshold":
        if experiment is None:
            st.warning("Please select an experiment first")
            st.stop()
        show_tune_threshold(task, model, args.dev_benchmarks, cfg.variant.type)


if __name__ == "__main__":
    run_demo(parse_args())
