import streamlit as st
from omegaconf import OmegaConf

from gnn_lib import models
from gnn_lib.utils import config


def show_info(model: models.Model, cfg: config.TrainConfig) -> None:
    st.write("### Training config")
    st.write("###### *Note: Some values in the config might be of the incorrect type (e.g. string instead of int), "
             "because they are interpolated from environment variables and only type checked and converted to "
             "the correct types when used during training*")
    st.code(OmegaConf.to_yaml(cfg, resolve=True), language="yaml")

    st.write("### Model specification")
    st.code(model)
