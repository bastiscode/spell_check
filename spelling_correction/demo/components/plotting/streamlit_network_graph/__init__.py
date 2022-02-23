import itertools
import os
import json
import random
from typing import Optional, Any, Dict

import streamlit.components.v1 as components

import networkx as nx

__all__ = ["network_graph"]

_RELEASE = True

_SHAPES = ["circle", "ellipse", "polygon"]
_NODE_COLORS = ["blue", "green", "red", "grey", "purple", "orange", "black"]
_EDGE_COLORS = ["grey", "darkblue", "brown", "darkgreen"]


def network_graph(
        graph: nx.DiGraph,
        layout: str = "kamada_kawai",
        key: Optional[str] = None) -> None:
    """Display a networkx graph in Streamlit.

    Parameters
    ----------
    graph: nx.DiGraph
        The networkx directed graph you want to visualize
    layout: str
        The layout used for determining the node positions.
        One of "kamada_kawai" or "spring". Default is "kamada_kawai".
    key: str or None
        An optional key that uniquely identifies this component. If this is
        None, and the component's arguments are changed, the component will
        be re-mounted in the Streamlit frontend and lose its current state.

    """
    # convert networkx graph to a python dict here which is json compatible
    graph_json: Dict[str, Any] = {
        "nodes": [],
        "edges": []
    }

    node_types = set()
    for n in graph.nodes:
        node = graph.nodes[n]
        if "type" in node:
            node_types.add(node["type"])
    node_shapes = {
        node_type: _SHAPES[i % len(_SHAPES)]
        for i, node_type in enumerate(node_types)
    }
    node_colors = {
        node_type: _NODE_COLORS[i % len(_NODE_COLORS)]
        for i, node_type in enumerate(node_types)
    }

    scale = 400
    center = (scale, scale)
    if layout == "spring":
        layout_ = nx.spring_layout(
            graph,
            scale=scale,
            center=center,
            seed=22
        )
    elif layout == "kamada_kawai":
        layout_ = nx.kamada_kawai_layout(
            graph,
            scale=scale,
            center=center
        )
    else:
        raise ValueError(f"Unknown layout {layout}")

    edge_types = set()
    for e in graph.edges:
        edge = graph.edges[e]
        if "type" in edge:
            edge_types.add(edge["type"])
    edge_colors = {
        edge_type: _EDGE_COLORS[i % len(_EDGE_COLORS)]
        for i, edge_type in enumerate(edge_types)
    }

    def scale_weight(x: float, min_: float, max_: float) -> float:
        return (max_ - min_) * x + min_

    for n in graph.nodes:
        node = graph.nodes[n]
        style: Dict[str, Any] = {}
        label = ""
        node_type = ""
        shape = "circle"
        if "type" in node:
            style["fill"] = node_colors[node["type"]]
            shape = node_shapes[node["type"]]
            node_type = node["type"]
            label = node_type[0]
        graph_json["nodes"].append({
            "id": str(n),
            "label": label,
            "nodeType": node_type,
            "x": layout_[n][0],
            "y": layout_[n][1],
            "style": style,
            "type": shape,
            "size": 15 if shape == "circle" else (25, 15),
            "features": {
                k: v for k, v in node.items()
                if k != "type"
            }
        })

    for fn, tn in graph.edges:
        edge = graph.edges[(fn, tn)]
        style = {}
        label = ""
        edge_type = ""
        weight = "unweighted"
        if "type" in edge:
            style["stroke"] = edge_colors[edge["type"]]
            edge_type = edge["type"]
            label = edge_type[0]
        if "weight" in edge:
            style["lineWidth"] = scale_weight(edge["weight"], 0.2, 1.2)
            style["opacity"] = scale_weight(edge["weight"], 0.5, 1.0)
            weight = str(round(edge["weight"], 4))
        graph_json["edges"].append({
            "source": str(fn),
            "target": str(tn),
            "label": label,
            "edgeType": edge_type,
            "style": style,
            "weight": weight,
            "features": {
                k: v for k, v in edge.items()
                if k not in {"type", "weight"}
            }
        })

    key = str(hash(
        json.dumps(graph_json, sort_keys=True)
    )) if key is None else key

    _component_func(
        graphJson=graph_json,
        key=key
    )


if not _RELEASE:
    _component_func = components.declare_component(
        "streamlit_network_graph",
        url="http://localhost:3001",
    )

    import streamlit as st

    st.subheader("Component with Gnp random graph")

    gnp = nx.fast_gnp_random_graph(
        n=50,
        p=0.1,
        directed=True
    )

    for n in gnp.nodes:
        if n % 2 == 0:
            gnp.nodes[n]["type"] = "blu"
        else:
            gnp.nodes[n]["type"] = "bla"

    # Create an instance of our component with a constant `name` arg, and
    # print its output value.
    network_graph(graph=gnp)

    st.markdown("---")
    st.subheader("Second component with Gnp random graph")

    fc = nx.DiGraph()
    nodes = ["A", "B", "C", "D"]
    fc.add_nodes_from(nodes)
    fc.add_edges_from(itertools.permutations(nodes, 2))
    for e in fc.edges:
        fc.edges[e]["weight"] = random.random()
        fc.edges[e]["type"] = random.choice(["bla", "blu", "bli"])
        fc.edges[e]["some feat"] = 123
    network_graph(graph=fc)

else:
    parent_dir = os.path.dirname(os.path.abspath(__file__))
    build_dir = os.path.join(parent_dir, "frontend/build")
    _component_func = components.declare_component(
        "streamlit_network_graph",
        path=build_dir
    )
