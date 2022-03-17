from collections import defaultdict
from typing import Tuple, Dict, Set, Any, List, Optional, Union

import dgl
import numpy as np
import torch

from gnn_lib.data import tokenization, utils


# class DGLData:
#     def __init__(self) -> None:
#         self.from_nodes: List[int] = []
#         self.to_nodes: List[int] = []
#         self.node_data: Dict[str, Dict[int, Any]] = defaultdict(dict)
#
#     def add(self, from_node: int, to_node: int, **kwargs: Any) -> None:
#         self.from_nodes.append(from_node)
#         self.to_nodes.append(to_node)
#
#     def add_node_data(self, feat: str, node: int, value: Any) -> None:
#         self.node_data[feat][node] = value
#
#     def get_data(self) -> Tuple[Iterable[int], Iterable[int]]:
#         return self.from_nodes, self.to_nodes
#
#     def get_num_nodes(self) -> int:
#         return len(set(self.from_nodes).union(set(self.to_nodes)))
#
#     def get_node_data(self) -> Dict[str, torch.Tensor]:
#         node_data = {}
#         for feat, value_dict in self.node_data.items():
#             values = [value_dict[i]
#                       for i in range(self.get_num_nodes())]
#             node_data[feat] = torch.tensor(values)
#
#         return node_data


class HeterogeneousDGLData:
    def __init__(self,
                 types: Set[Tuple[str, str, str]]) -> None:
        self.data_dict: Dict[Tuple[str, str, str], Tuple[List[int], List[int]]] = {}
        self.node_data: Dict[str, Dict[str, Dict[int, Any]]] = {}
        self.edge_data: Dict[str, Dict[Tuple[str, str, str], Dict[int, Any]]] = {}

        self.canonical_types = types
        self.node_types = set(t[0] for t in self.canonical_types).union(set(t[2] for t in self.canonical_types))

        self.num_nodes_dict: Dict[str, Dict[int, int]] = defaultdict(dict)
        self.num_edges_dict: Dict[Tuple[str, str, str], int] = defaultdict(int)
        self.edge_to_id_dict: Dict[Tuple[str, str, str], Dict[Tuple[int, int], int]] = defaultdict(dict)

    def add_type(self, type_tuple: Tuple[str, str, str]) -> None:
        self.canonical_types.add(type_tuple)

        self.node_types.add(type_tuple[0])
        self.node_types.add(type_tuple[2])

    def add_edge(self,
                 from_node: int,
                 to_node: int,
                 type_tuple: Tuple[str, str, str]) -> None:
        if type_tuple not in self.data_dict:
            self.data_dict[type_tuple] = ([], [])

        self.data_dict[type_tuple][0].append(from_node)
        self.data_dict[type_tuple][1].append(to_node)

        self.edge_to_id_dict[type_tuple][(from_node, to_node)] = self.num_edges_dict[type_tuple]
        self.num_edges_dict[type_tuple] += 1
        self.num_nodes_dict[type_tuple[0]][from_node] = 1
        self.num_nodes_dict[type_tuple[2]][to_node] = 1

    def add_edge_data(self,
                      edge_type: Tuple[str, str, str],
                      edge: Union[int, Tuple[int, int]],
                      feat_dict: Dict[str, Any]):
        if isinstance(edge, tuple):
            assert edge in self.edge_to_id_dict[edge_type], \
                f"could not find edge {edge} of type {edge_type} in {self.edge_to_id_dict}"
            edge = self.edge_to_id_dict[edge_type][edge]
        for feat, value in feat_dict.items():
            if feat not in self.edge_data:
                self.edge_data[feat] = {}
            if edge_type not in self.edge_data[feat]:
                self.edge_data[feat][edge_type] = {}
            self.edge_data[feat][edge_type][edge] = value

    def add_node_data(self,
                      node_type: str,
                      node: int,
                      feat_dict: Dict[str, Any]) -> None:
        for feat, value in feat_dict.items():
            if feat not in self.node_data:
                self.node_data[feat] = {}
            if node_type not in self.node_data[feat]:
                self.node_data[feat][node_type] = {}
            self.node_data[feat][node_type][node] = value

    def get_data(self) -> Dict[Tuple[str, str, str],
                               Tuple[List[int], List[int]]]:
        data = self.data_dict
        for type_tuple in self.canonical_types:
            if type_tuple not in data:
                data[type_tuple] = ([], [])
        return data

    def get_types(self) -> Set[Tuple[str, str, str]]:
        return self.canonical_types

    def get_num_nodes(self) -> Dict[str, int]:
        return {
            node_type: len(self.num_nodes_dict[node_type])
            for node_type in self.node_types
        }

    def get_num_edges(self) -> Dict[Tuple[str, str, str], int]:
        return {
            edge_type: self.num_edges_dict[edge_type]
            for edge_type in self.canonical_types
        }

    def get_edge_data(self) -> Union[Dict[str, torch.Tensor], Dict[str, Dict[str, torch.Tensor]]]:
        only_edge_type = None if len(self.canonical_types) > 1 else list(self.canonical_types)[0]

        if only_edge_type is not None:
            edge_data: Dict[str, torch.Tensor] = {}
        else:
            edge_data: Dict[str, Dict[Tuple[str, str, str], torch.Tensor]] = defaultdict(dict)

        num_edges_dict = self.get_num_edges()
        for feat, edge_type_dict in self.edge_data.items():
            for edge_type, num_edges in num_edges_dict.items():
                if edge_type not in edge_type_dict:
                    continue
                values = [edge_type_dict[edge_type][i]
                          for i in range(num_edges)]
                if only_edge_type is not None:
                    edge_data[feat] = torch.tensor(values)
                else:
                    edge_data[feat][edge_type] = torch.tensor(values)
        return edge_data

    def get_node_data(self) -> Union[Dict[str, torch.Tensor], Dict[str, Dict[str, torch.Tensor]]]:
        only_node_type = None if len(self.node_types) > 1 else list(self.node_types)[0]

        if only_node_type is not None:
            node_data: Dict[str, torch.Tensor] = {}
        else:
            node_data: Dict[str, Dict[str, torch.Tensor]] = defaultdict(dict)

        num_nodes_dict = self.get_num_nodes()
        for feat, node_type_dict in self.node_data.items():
            for node_type, num_nodes in num_nodes_dict.items():
                if node_type not in node_type_dict:
                    continue
                values = [node_type_dict[node_type][i]
                          for i in range(num_nodes)]
                if only_node_type is not None:
                    node_data[feat] = torch.tensor(values)
                else:
                    node_data[feat][node_type] = torch.tensor(values)
        return node_data

    def to_heterograph(self) -> dgl.DGLHeteroGraph:
        g = dgl.heterograph(
            data_dict=self.get_data(),
            num_nodes_dict=self.get_num_nodes(),
            idtype=torch.int32
        )

        for feat, feat_dict in self.get_node_data().items():
            g.ndata[feat] = feat_dict

        for feat, feat_dict in self.get_edge_data().items():
            g.edata[feat] = feat_dict

        return g


def sequence_to_token_graph(
        sample: utils.SAMPLE
) -> HeterogeneousDGLData:
    types = {
        ("token", "connects_to", "token")
    }
    data = HeterogeneousDGLData(types=types)

    tokens = [t for word_tokens in sample.tokens for t in word_tokens]

    for i, token_id in enumerate(tokens):
        for j, _ in enumerate(tokens):
            data.add_edge(i, j, ("token", "connects_to", "token"))

        feat_dict = {
            "position": i,
            "token_id": token_id
        }

        data.add_node_data(
            node_type="token",
            node=i,
            feat_dict=feat_dict
        )

    return data


def _indices_to_sequence_edge_type(from_idx: int, to_idx: int) -> str:
    if from_idx == to_idx:
        return "self"
    else:
        return "connects_to"


def sequence_to_word_graph(
        sample: utils.SAMPLE,
        tokenizer: tokenization.Tokenizer,
        add_sequence_level_node: bool = False,
        dictionary: Optional[Dict[str, int]] = None,
        add_word_features: bool = False,
        add_ner_features: bool = False,
        add_pos_tag_features: bool = False,
        add_dependency_info: bool = False,
        add_num_neighbors: Optional[int] = None,
        token_fully_connected: bool = False,
        word_fully_connected: bool = True,
        add_self_loops: bool = True,
        scheme: str = "token_to_word"
) -> HeterogeneousDGLData:
    assert scheme in {"token_to_word", "word_to_token", "token_bidi_word"}
    types = {
        ("token", "connects_to_token", "token")
    }
    if add_self_loops:
        types.add(("token", "self", "token"))
        types.add(("word", "self", "word"))
        if add_sequence_level_node:
            types.add(("sequence", "self", "sequence"))
        if add_num_neighbors is not None:
            types.add(("neighbor", "self", "neighbor"))
    if word_fully_connected:
        types.add(("word", "connects_to_word", "word"))
    if scheme != "token_to_word":
        types.add(("word", "contains", "token"))
    if scheme != "word_to_token":
        types.add(("token", "in", "word"))

    if add_sequence_level_node:
        if scheme == "word_to_token":
            types.add(("token", "in", "sequence"))
        else:
            types.add(("word", "in", "sequence"))

    if add_dependency_info:
        types.add(("word", "dep", "word"))
        types.add(("word", "head", "word"))
        # for tag in utils.SPACY_DEP_TAG_MAP.keys():
        #     types.add(("word", f"{tag}_dep", "word"))
        #     types.add(("word", f"{tag}_head", "word"))

    data = HeterogeneousDGLData(types=types)

    token_start_indices = [0] + list(np.cumsum([len(tokens) for tokens in sample.tokens[:-1]]))
    dependencies = {}
    for from_word_idx, (word, tokens) in enumerate(zip(sample.doc, sample.tokens)):
        token_start_idx = token_start_indices[from_word_idx]
        for from_token_idx, token_id in enumerate(tokens):
            for to_token_idx in range(len(tokens)):
                # add edges between tokens from a word (optional self-loops)
                if from_token_idx == to_token_idx and not add_self_loops:
                    continue

                data.add_edge(
                    token_start_idx + from_token_idx,
                    token_start_idx + to_token_idx,
                    ("token",
                     "self" if from_token_idx == to_token_idx else "connects_to_token",
                     "token")
                )

            if token_fully_connected:
                # add edges to all other tokens in other words
                for to_word_idx in range(len(sample.doc)):
                    if from_word_idx == to_word_idx:
                        continue

                    for to_token_idx in range(len(sample.tokens[to_word_idx])):
                        from_idx = token_start_idx + from_token_idx
                        to_idx = token_start_indices[to_word_idx] + to_token_idx
                        data.add_edge(
                            from_idx,
                            to_idx,
                            ("token",
                             "connects_to_token",
                             "token")
                        )

            if scheme != "token_to_word":
                # add edges from word to token node
                data.add_edge(from_word_idx,
                              token_start_idx + from_token_idx,
                              ("word", "contains", "token"))
            if scheme != "word_to_token":
                data.add_edge(token_start_idx + from_token_idx,
                              from_word_idx,
                              ("token", "in", "word"))

            data.add_node_data(
                node_type="token",
                node=token_start_idx + from_token_idx,
                feat_dict={
                    "position": token_start_idx + from_token_idx,
                    "token_id": token_id
                }
            )

        # fully connect words
        if word_fully_connected:
            for to_word_idx in range(len(sample.doc)):
                if from_word_idx == to_word_idx and not add_self_loops:
                    continue

                data.add_edge(from_word_idx,
                              to_word_idx,
                              (
                                  "word",
                                  "self" if from_word_idx == to_word_idx else "connects_to_word",
                                  "word"
                              )
                              )
        elif add_self_loops:
            data.add_edge(
                from_word_idx,
                from_word_idx,
                (
                    "word",
                    "self",
                    "word"
                )
            )

        feat_dict = {}

        features = []
        if add_word_features:
            features += utils.token_flags(word, dictionary)
        if add_ner_features:
            ner_id = utils.SPACY_NER_MAP.get(word.ent_type_, -1)
            features += utils.one_hot_encode(ner_id, utils.SPACY_NUM_NER_TYPES)
        if add_pos_tag_features:
            pos_tag_id = utils.SPACY_POS_TAG_MAP.get(word.pos_, -1)
            features += utils.one_hot_encode(pos_tag_id, utils.SPACY_NUM_POS_TAGS)
        if add_dependency_info:
            dep_tag_id = utils.SPACY_DEP_TAG_MAP.get(word.dep_, -1)
            dependencies[from_word_idx] = (word.head.i, dep_tag_id)
            features += utils.parser_token_flags(word)
        if len(features) > 0:
            feat_dict["features"] = features

        if len(feat_dict) > 0:
            data.add_node_data(
                node_type="word",
                node=from_word_idx,
                feat_dict=feat_dict
            )

    if len(dependencies) > 0:
        for word_idx, (depends_on_idx, tag) in dependencies.items():
            feat_dict = {"features": utils.one_hot_encode(tag, utils.SPACY_NUM_DEP_TAGS)}
            data.add_edge(word_idx,
                          depends_on_idx,
                          ("word", "dep", "word"))
            data.add_edge_data(("word", "dep", "word"),
                               (word_idx, depends_on_idx),
                               feat_dict)
            data.add_edge(depends_on_idx,
                          word_idx,
                          ("word", "head", "word"))
            data.add_edge_data(("word", "head", "word"),
                               (depends_on_idx, word_idx),
                               feat_dict)

    if add_sequence_level_node:
        if scheme == "word_to_token":
            for token_idx in range(data.get_num_nodes()["token"]):
                data.add_edge(token_idx,
                              0,
                              ("token", "in", "sequence"))
        else:
            for word_idx in range(data.get_num_nodes()["word"]):
                data.add_edge(word_idx,
                              0,
                              ("word", "in", "sequence"))
        if add_self_loops:
            data.add_edge(
                0, 0, ("sequence", "self", "sequence")
            )

    if add_num_neighbors is not None:
        assert sample.neighbors_list is not None
        data.add_type(("token", "in", "neighbor"))
        data.add_type(("neighbor", "neighbor_of", "word"))
        token_start_idx = data.get_num_nodes()["token"]
        neighbor_start_idx = 0
        for from_word_idx, neighbors in enumerate(sample.neighbors_list):
            distances = np.array(neighbors.distances[:add_num_neighbors])
            distance_sum = np.sum(distances)
            # normalize distances across all neighbors
            distances = distances / (distance_sum * (distance_sum > 0) + (distance_sum <= 0) * 1)
            for i, word in enumerate(neighbors.words[:add_num_neighbors]):
                neighbor_tokens = tokenizer.tokenize(word)
                for j, token in enumerate(neighbor_tokens):
                    for k in range(len(neighbor_tokens)):
                        # add edges between tokens from a word (optional self-loops)
                        if j == k and not add_self_loops:
                            continue

                        data.add_edge(
                            token_start_idx + j,
                            token_start_idx + k,
                            ("token",
                             "self" if j == k else "connects_to_token",  # _indices_to_sequence_edge_type(j, k),
                             "token")
                        )

                    feat_dict = {"position": token_start_indices[from_word_idx] + j,
                                 "token_id": token}
                    data.add_edge(
                        token_start_idx + j,
                        neighbor_start_idx + i,
                        ("token", "in", "neighbor")
                    )
                    data.add_node_data(
                        "token",
                        token_start_idx + j,
                        feat_dict=feat_dict
                    )
                token_start_idx += len(neighbor_tokens)
                data.add_edge(neighbor_start_idx + i,
                              from_word_idx,
                              ("neighbor", "neighbor_of", "word"))
                data.add_edge_data(
                    ("neighbor", "neighbor_of", "word"),
                    (neighbor_start_idx + i, from_word_idx),
                    feat_dict={"features": [distances[i]]}
                )
                if add_self_loops:
                    data.add_edge(neighbor_start_idx + i,
                                  neighbor_start_idx + i,
                                  ("neighbor", "self", "neighbor"))

            neighbor_start_idx += len(neighbors.words[:add_num_neighbors])

    return data


def add_graph2seq_decoder_nodes_to_word_graph(
        word_graph_data: HeterogeneousDGLData,
        decoder_input_ids: List[int],
        add_self_loops: bool = True,
        scheme: str = "token_to_word",
        decoder_target_ids: Optional[List[int]] = None
) -> HeterogeneousDGLData:
    word_graph_data.add_type(("token", "token_to_dec", "dec"))
    if scheme != "word_to_token":
        word_graph_data.add_type(("word", "word_to_dec", "dec"))
    word_graph_data.add_type(("dec", "connects_to", "dec"))

    num_nodes = word_graph_data.get_num_nodes()
    for i, decoder_input_id in enumerate(decoder_input_ids):
        if scheme != "word_to_token":
            for word_idx in range(num_nodes["word"]):
                word_graph_data.add_edge(word_idx, i, ("word", "word_to_dec", "dec"))

        for token_idx in range(num_nodes["token"]):
            word_graph_data.add_edge(token_idx, i, ("token", "token_to_dec", "dec"))

        if add_self_loops:
            word_graph_data.add_edge(i, i, ("dec", "connects_to", "dec"))

        feat_dict = {"position": i,
                     "dec_id": decoder_input_id}
        if decoder_target_ids is not None:
            feat_dict["label"] = decoder_target_ids[i]

        word_graph_data.add_node_data("dec",
                                      i,
                                      feat_dict=feat_dict)

        # connect all previous nodes to current node
        for j in range(i):
            word_graph_data.add_edge(j, i, ("dec", "connects_to", "dec"))

    return word_graph_data
