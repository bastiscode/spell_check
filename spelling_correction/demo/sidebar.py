import glob
import os
from typing import Tuple, Any, Dict

import streamlit as st
import torch

from spelling_correction.demo.utils import load_config


def _load_experiment_name_and_type(experiment: str, **kwargs: Any) -> Tuple[str, str]:
    cfg = load_config(experiment, **kwargs)
    return cfg.experiment_name, cfg.variant.type


def _format_experiment(e: str, **kwargs: Any) -> str:
    name, variant_type = _load_experiment_name_and_type(e, **kwargs)
    return f"{name} ({variant_type})"


def show_sidebar(experiment_dir: str, **kwargs: Any) -> Tuple[str, str, torch.device, Dict[str, Any]]:
    experiments = glob.glob(os.path.join(experiment_dir, "*/*"))

    st.sidebar.write("# Navigation")
    st.sidebar.write("##### Select a page")
    page = st.sidebar.selectbox(
        "Pages",
        ["Home",
         "Data variants",
         "Inference",
         "Info",
         "Upload benchmarks",
         "Evaluate benchmarks",
         "Run benchmarks",
         "Tune threshold",
         "Evaluation game"],
        index=0
    )
    st.sidebar.write("---")

    st.sidebar.write("# Experiments")
    st.sidebar.write("##### Select an experiment")
    if len(experiments) > 0:
        experiments = ["-"] + experiments
        experiment = st.sidebar.selectbox(
            "Experiments",
            experiments,
            format_func=lambda e: "-" if e == "-" else _format_experiment(e, **kwargs)
        )
        if experiment == "-":
            experiment = None
        else:
            experiment_name, variant_type = _load_experiment_name_and_type(experiment, **kwargs)
            date = ".".join(os.path.basename(experiment).split("_")[-6:-3])
            time = ":".join(os.path.basename(experiment).split("_")[-3:])
            st.sidebar.write("##### Selected experiment")
            st.sidebar.json(
                {
                    "experiment_name": experiment_name,
                    "variant_type": variant_type,
                    "date": date,
                    "time": time
                }
            )
    else:
        st.sidebar.warning(f"Could not find any experiments")
        experiment = None
    st.sidebar.write("---")

    st.sidebar.write("# Options")
    st.sidebar.write("##### Select a device to run the model on")
    num_gpus = torch.cuda.device_count()
    gpu_names = [
        f"{torch.cuda.get_device_name(i)} (cuda:{i})"
        for i in range(num_gpus)
    ]
    select_device = st.sidebar.selectbox(
        "Devices",
        [-1] + list(range(num_gpus)),
        format_func=lambda i: "CPU" if i == -1 else gpu_names[i]
    )
    device = torch.device(
        "cpu" if select_device < 0
        else f"cuda:{select_device}"
    )

    inference_kwargs = {}
    if experiment is not None:
        experiment_name, experiment_type = _load_experiment_name_and_type(experiment, **kwargs)
        if experiment_type.endswith("_NMT"):
            st.sidebar.write("##### Select an inference mode")
            select_inference_method = st.sidebar.selectbox(
                "Inference modes",
                ["greedy", "sample", "beam", "best_first"],
            )

            sample_top_k = None
            beam_width = None
            best_first_top_k = None
            inference_kwargs = {
                "inference_mode": select_inference_method,
                "sample_top_k": sample_top_k,
                "beam_width": beam_width,
                "best_first_top_k": best_first_top_k
            }

            if select_inference_method == "sample":
                sample_top_k = st.sidebar.slider("Sample from top k predictions",
                                                 min_value=1,
                                                 max_value=10,
                                                 value=5)
                inference_kwargs["sample_top_k"] = sample_top_k
            elif select_inference_method == "beam":
                beam_width = st.sidebar.slider("Specify the beam width",
                                               min_value=1,
                                               max_value=10,
                                               value=5)
                inference_kwargs["beam_width"] = beam_width
            elif select_inference_method == "best_first":
                beam_width = st.sidebar.slider("Search for the top k solutions",
                                               min_value=1,
                                               max_value=5,
                                               value=3)
                inference_kwargs["best_first_top_k"] = beam_width

        if experiment_type.startswith("SED_"):
            threshold = st.sidebar.slider("Set the threshold for detecting spelling errors",
                                          min_value=0.,
                                          max_value=1.,
                                          value=0.5,
                                          step=0.05)
            inference_kwargs = {"threshold": threshold}

    return page, experiment, device, inference_kwargs
