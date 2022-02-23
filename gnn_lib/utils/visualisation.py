import dgl
import networkx as nx
import matplotlib.pyplot as plt

from gnn_lib.utils import graph


def visualise_graph_matplotlib(
        g: dgl.DGLHeteroGraph) -> plt.Figure:
    nxg = graph.dgl_to_networkx(g)
    pos = nx.kamada_kawai_layout(nxg)

    fig, ax = plt.subplots(figsize=(20, 10))
    if g.is_homogeneous:
        nx.draw(nxg, pos=pos)
        return fig

    color_map = plt.get_cmap("jet")

    ntypes = list(set(nxg.nodes[n]["type"]
                      for n in nxg.nodes))
    node_map = {ntypes[i]: i
                for i in range(len(ntypes))}
    etypes = list(set(nxg.edges[e]["type"]
                      for e in nxg.edges))
    edge_map = {etypes[i]: i + len(ntypes)
                for i in range(len(etypes))}

    nx.draw_networkx_edges(nxg,
                           pos=pos,
                           edge_color=[
                               edge_map[nxg.edges[e]["type"]]
                               for e in nxg.edges
                           ],
                           edge_cmap=color_map)
    # edge_labels = nx.get_edge_attributes(nxg, "type")
    # nx.draw_networkx_edge_labels(nxg, pos=pos, edge_labels=edge_labels)

    nx.draw_networkx_nodes(nxg,
                           pos=pos,
                           node_size=20,
                           nodelist=nxg.nodes,
                           node_color=[
                               node_map[nxg.nodes[n]["type"]]
                               for n in nxg.nodes
                           ],
                           cmap=color_map)
    node_labels = nx.get_node_attributes(nxg, "type")
    nx.draw_networkx_labels(nxg, pos=pos, labels=node_labels, font_size=8)

    return fig


def visualize_graph_graphviz(g: dgl.DGLHeteroGraph) -> str:
    nxg = graph.dgl_to_networkx(g)
    dot_graph = nx.nx_pydot.to_pydot(nxg)

    dot_graph.set_colorscheme("dark28")

    num_colors = 8  # dark 28 has 8 different colors
    ntypes = set(node.get("type")
                 for node in dot_graph.get_nodes())
    ntype_to_color = {
        ntype: i % num_colors + 1
        for i, ntype in enumerate(ntypes)
    }
    etypes = set(edge.get("type")
                 for edge in dot_graph.get_edges())
    etype_to_color = {
        etype: i % num_colors + 1
        for i, etype in enumerate(etypes)
    }

    for node in dot_graph.get_nodes():
        node.set_colorscheme("dark28")
        node.set_color(ntype_to_color[node.get("type")])
        node.set_style("filled")
        node.set_label(node.get("type"))

    for edge in dot_graph.get_edges():
        edge.set_colorscheme("dark28")
        edge.set_color(etype_to_color[edge.get("type")])
        edge.set_style("filled")
        edge.set_label(edge.get("type"))

    return dot_graph.to_string()
