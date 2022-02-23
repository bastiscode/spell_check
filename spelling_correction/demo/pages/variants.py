import omegaconf
import pandas as pd
import streamlit as st
from streamlit_network_graph import network_graph

from gnn_lib.data import variants
from gnn_lib.utils import graph as graph_utils


def show_dataset_variant(variant: variants.DatasetVariant) -> None:
    is_inference = st.checkbox("Inference mode", value=True)
    st.write("*In inference mode the input sequence will not be corrupted and no labels will be generated*")
    sequence = st.text_input(
        "Enter a sequence to represent as a graph"
    )
    if sequence == "":
        st.stop()

    show_variant(sequence, variant, is_inference)


def show_variant(
        sequence: str,
        variant: variants.DatasetVariant,
        is_inference: bool) -> None:
    sequence, noised_sequence = variant.noise_fn(sequence) if not is_inference else sequence, sequence
    graph, info = variant.prepare_sequence(
        noised_sequence,
        is_inference=True
    )

    nxg = graph_utils.dgl_to_networkx(graph)

    st.write(f"##### Input sequence: {noised_sequence}")
    network_graph(nxg, layout="kamada_kawai", key="graph")

    # col1, col2 = st.columns(2)
    # with col1:
    #     st.write("##### Additional information")
    #     st.write(info)
    #
    # with col2:
    #     if "word" in graph.ntypes:
    #         feature_explanations = []
    #         if variant.cfg.add_word_features:
    #             feature_explanations.extend([
    #                 "is punctuation",
    #                 "is currency",
    #                 "is digit or like a number",
    #                 "is like url",
    #                 "is like email",
    #                 "has trailing whitespace",
    #                 "is title case",
    #                 "is upper case",
    #                 "is lower case",
    #                 "is stop word",
    #                 "is alphanumeric"
    #             ])
    #         if variant.cfg.dictionary_file is not None:
    #             feature_explanations.extend([
    #                 "word in dictionary",
    #                 "word to lowercase in dictionary"
    #             ])
    #         feature_explanations.extend([
    #             "is sentence start",
    #             "is sentence end"
    #         ])
    #         st.write("##### Graph contains word nodes, "
    #                  "to each word the following features are added:")
    #         st.json(feature_explanations)

    data_dict = {
        "Total nodes": str(graph.num_nodes()),
        "Total edges": str(graph.num_edges()),
        "Noised sequence": noised_sequence,
        "Original sequence": sequence
    }
    for e_type in graph.canonical_etypes:
        data_dict[f"{e_type} edges"] = str(graph.num_edges(e_type))
    for n_type in graph.ntypes:
        data_dict[f"{n_type} nodes"] = str(graph.num_nodes(n_type))

    st.write("##### Graph information")
    st.dataframe(
        pd.DataFrame.from_dict({
            "Info": data_dict
        })
    )

    st.write("##### Data variant config")
    variant_cfg = variant.cfg
    omegaconf.OmegaConf.resolve(variant_cfg)
    st.code(
        omegaconf.OmegaConf.to_yaml(variant_cfg),
        "yaml"
    )
