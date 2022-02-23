import os
import pickle
from typing import Optional, Dict

import omegaconf

from gnn_lib.data import tokenization
from gnn_lib.utils import io, config

import streamlit as st


def load_config(experiment: str, override_env_vars: Optional[Dict[str, str]] = None) -> omegaconf.DictConfig:
    with open(os.path.join(experiment, "cfg.pkl"), "rb") as inf:
        cfg, env_vars = pickle.load(inf)

    env_vars.update(override_env_vars)
    config.set_gnn_lib_env_vars(env_vars)
    return cfg


def select_dictionary_file(text: str, data_dir: str) -> str:
    dictionary_files = sorted(io.glob_safe(os.path.join(data_dir, "dictionaries/*100k.txt")))
    dictionary_file = st.selectbox(
        text,
        [df for df in dictionary_files],
        format_func=lambda s: "/".join(s.split("/")[-2:])
    )
    if dictionary_file == "":
        st.error("You must select a dictionary file")
        st.stop()
    return dictionary_file


def select_tokenizer_dialog(text: str, data_dir: str) -> tokenization.TokenizerConfig:
    file_path = select_tokenizer_file(text, data_dir)
    tokenizer_type = get_tokenizer_type_from_tokenizer_file(file_path)
    return tokenization.TokenizerConfig(type=tokenizer_type, file_path=file_path)


def select_tokenizer_file(text: str, data_dir: str) -> str:
    tokenizer_files = sorted(io.glob_safe(os.path.join(data_dir, "tokenizers/*/*.pkl")))
    tokenizer_file = st.selectbox(
        text,
        [tf for tf in tokenizer_files],
        format_func=lambda s: "/".join(s.split("/")[-2:])
    )
    if tokenizer_file == "":
        st.error("You must select a tokenizer file")
        st.stop()
    return tokenizer_file


def get_tokenizer_type_from_tokenizer_file(file_path: str) -> tokenization.Tokenizers:
    split_path = file_path.split("/")
    if file_path == "char":
        return tokenization.Tokenizers.CHAR
    elif split_path[-2] == "bpe":
        return tokenization.Tokenizers.BPE
    elif split_path[-2] == "word":
        return tokenization.Tokenizers.WORD
    else:
        st.error(f"Tokenizer file '{file_path}' is not in one of the "
                 f"expected subdirectories 'tokenizers/bpe' or 'tokenizers/word' "
                 f"in the data directory "
                 f"which are relevant to determine the tokenizer type.")
        st.stop()


def last_n_k_path(path: str, n: int = 3, k: int = None) -> str:
    if k is None:
        return "/".join(path.split("/")[-n:])
    else:
        return "/".join(path.split("/")[-n:-k])


def select_benchmark_group() -> str:
    benchmark_group = st.selectbox(
        "Select a benchmark group",
        options=["-", "sec", "sed_sequence", "sed_words", "tokenization_repair"]
    )
    if benchmark_group == "-":
        st.info("Please select a benchmark group to evaluate")
        st.stop()

    return benchmark_group
