import dgl
import networkx as nx

from gnn_lib.modules.utils import tensor_to_python


def _homogeneous_to_networkx(g: dgl.DGLGraph) -> nx.DiGraph:
    nxg = nx.DiGraph()
    for n in g.nodes():
        node_data = {"type": "n"}
        for k, v in g.nodes[n].data.items():
            node_data[k] = tensor_to_python(v)
        nxg.add_node(
            n.tolist(),
            **node_data
        )

    sources, targets, eids = g.edges(form="all")
    for from_, to_, eid in \
            zip(sources.tolist(),
                targets.tolist(),
                eids.tolist()):
        edge_data = {"type": "e"}
        for k, v in g.edges[eid].data.items():
            edge_data[k] = tensor_to_python(v)
        nxg.add_edge(
            from_,
            to_,
            **edge_data
        )
    return nxg


def _heterogeneous_to_networkx(g: dgl.DGLHeteroGraph) -> nx.DiGraph:
    nxg = nx.DiGraph()
    for from_type, edge_type, to_type in g.canonical_etypes:
        edges = g.all_edges(
            form="all",
            etype=(from_type, edge_type, to_type)
        )
        edges = tuple(v.tolist() for v in edges)

        from_features = g.node_attr_schemes(from_type).keys()
        to_features = g.node_attr_schemes(to_type).keys()
        edge_features = g.edge_attr_schemes(
            (from_type, edge_type, to_type)
        ).keys()
        for from_, to_, eid_ in zip(*edges):
            from_node = (from_type, from_)
            to_node = (to_type, to_)

            if not nxg.has_node(from_node):
                node_data = {"type": from_type}
                for feat in from_features:
                    node_data[feat] = tensor_to_python(
                        g.ndata[feat][
                            from_type
                        ][from_]
                    )
                nxg.add_node(
                    from_node,
                    **node_data
                )
            if not nxg.has_node(to_node):
                node_data = {"type": to_type}
                for feat in to_features:
                    node_data[feat] = tensor_to_python(
                        g.ndata[feat][
                            to_type
                        ][to_]
                    )
                nxg.add_node(
                    to_node,
                    **node_data
                )

            edge_data = {"type": edge_type}
            for feat in edge_features:
                edge_data[feat] = tensor_to_python(
                    g.edata[feat][
                        (from_type, edge_type, to_type)
                    ][eid_]
                )
            nxg.add_edge(
                from_node,
                to_node,
                **edge_data
            )

    return nxg


def dgl_to_networkx(g: dgl.DGLHeteroGraph) -> nx.DiGraph:
    return _homogeneous_to_networkx(g) if g.is_homogeneous else _heterogeneous_to_networkx(g)
