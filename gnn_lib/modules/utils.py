import time
from typing import Tuple, Any, Dict, List, Optional, Union, Set

import dgl
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import PackedSequence


class EncoderMixin(nn.Module):
    def encode(self, g: dgl.DGLHeteroGraph, **kwargs: Any) -> dgl.DGLHeteroGraph:
        raise NotImplementedError


class DecoderMixin(nn.Module):
    def decode(self,
               decoder_inputs: torch.Tensor,
               **kwargs: Any) -> torch.Tensor:
        raise NotImplementedError


def device_from_model(model: nn.Module) -> torch.device:
    return next(model.parameters()).device


def cuda_or_cpu(force_cpu: bool = False) -> torch.device:
    from torch import distributed
    return torch.device(
        "cpu" if force_cpu or not torch.cuda.is_available()
        else (f"cuda:{torch.distributed.get_rank()}"
              if torch.distributed.is_available() and torch.distributed.is_initialized()
              else "cuda")
    )


def tensor_to_python(t: torch.Tensor, force_list: bool = False) -> Any:
    return t.tolist() if t.numel() > 1 or force_list else t.item()


def split(items: Union[List, torch.Tensor], lengths: Union[List, torch.Tensor]) -> List:
    assert len(items) == sum(lengths)
    split_items = []
    running_size = 0
    for length in lengths:
        split_items.append(items[running_size:running_size + length])
        running_size += length
    return split_items


def pad(inputs: List[torch.Tensor], val: float = 0) -> torch.Tensor:
    return torch.nn.utils.rnn.pad_sequence(inputs, batch_first=True, padding_value=val)


def split_and_pad(inputs: torch.Tensor, splits: Union[List[int], torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    inputs = torch.split(
        inputs,
        tensor_to_python(splits, force_list=True) if isinstance(splits, torch.Tensor) else splits
    )
    x = pad(inputs)
    return x, splits if isinstance(splits, torch.Tensor) else torch.tensor(splits, device=x.device, dtype=torch.long)


def split_and_pack(inputs: torch.Tensor, splits: Union[List[int], torch.Tensor]) -> PackedSequence:
    inputs = torch.split(inputs,
                         tensor_to_python(splits, force_list=True) if isinstance(splits, torch.Tensor) else splits)
    return torch.nn.utils.rnn.pack_sequence(inputs, enforce_sorted=False)


def unpack(packed: PackedSequence) -> Tuple[torch.Tensor, torch.Tensor]:
    unpacked, lengths = torch.nn.utils.rnn.pad_packed_sequence(packed, batch_first=True)
    return unpacked, lengths


def join(x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    outputs = []
    for i in range(len(x)):
        outputs.append(x[i, :lengths[i]])
    return torch.cat(outputs)


_MASK_VALUE = -10_000.0


def padding_mask(x: torch.Tensor, lengths: Union[torch.Tensor, List[int]]) -> torch.Tensor:
    assert x.shape[0] == len(lengths)
    mask = np.zeros(x.shape[:2], dtype=bool)
    for i, l in enumerate(lengths):
        mask[i, l:] = True
    return torch.from_numpy(mask).to(x.device)


def square_causal_mask(length: int, device: torch.device) -> torch.Tensor:
    return torch.triu(torch.ones(length, length, dtype=torch.bool, device=device), diagonal=1) * _MASK_VALUE


# def square_causal_block_mask(length: int, lengths: torch.Tensor, device: torch.device) -> torch.Tensor:
#     triangular_mask = square_causal_mask(length, device)
#
#     block_mask = torch.ones(length, length, dtype=torch.bool, device=device)
#     cum_lengths = torch.cumsum(lengths, dim=0)
#     assert cum_lengths[-1] <= length, f"{cum_lengths[-1]} > {length}"
#     for i in range(len(lengths)):
#         block_mask[:cum_lengths[i], cum_lengths[i] - lengths[i]:] = 0
#
#     return torch.logical_or(triangular_mask, block_mask) * _MASK_VALUE


def square_causal_block_mask(length: int, lengths: torch.Tensor, device: torch.device) -> torch.Tensor:
    triangular_mask = np.triu(np.ones((length, length), dtype=bool), 1)

    block_mask = np.ones((length, length), dtype=bool)
    cum_lengths = np.cumsum(lengths, 0)
    assert cum_lengths[-1] <= length, f"{cum_lengths[-1]} > {length}"
    lower_indices = cum_lengths - lengths
    for i, (l, u) in enumerate(zip(lower_indices, cum_lengths)):
        block_mask[:u, l:] = False

    mask = np.logical_or(triangular_mask, block_mask) * _MASK_VALUE
    return torch.from_numpy(mask.astype(np.float32)).to(device)


# def rectangular_block_mask(
#         target_length: int,
#         source_length: int,
#         target_lengths: torch.Tensor,
#         source_lengths: torch.Tensor,
#         device: torch.device) -> torch.Tensor:
#     block_mask = torch.ones(target_length, source_length, dtype=torch.bool, device=device)
#     target_cum_lengths = torch.cumsum(target_lengths, 0)
#     source_cum_lengths = torch.cumsum(source_lengths, 0)
#     assert len(target_lengths) == len(source_lengths)
#     assert target_cum_lengths[-1] <= target_length, f"{target_cum_lengths[-1]} > {target_length}"
#     assert source_cum_lengths[-1] <= source_length, f"{source_cum_lengths[-1]} > {source_length}"
#     for i in range(len(source_lengths)):
#         tl = target_cum_lengths[i] - target_lengths[i]
#         tu = target_cum_lengths[i]
#         sl = source_cum_lengths[i] - source_lengths[i]
#         su = source_cum_lengths[i]
#         block_mask[tl:tu, sl:su] = 0
#     return block_mask * _MASK_VALUE


def rectangular_block_mask(
        target_length: int,
        source_length: int,
        target_lengths: np.ndarray,
        source_lengths: np.ndarray,
        device: torch.device) -> torch.Tensor:
    block_mask = np.ones((target_length, source_length), dtype=np.float32)
    target_cum_lengths = np.cumsum(target_lengths, 0)
    source_cum_lengths = np.cumsum(source_lengths, 0)
    assert len(target_lengths) == len(source_lengths)
    assert target_cum_lengths[-1] <= target_length, f"{target_cum_lengths[-1]} > {target_length}"
    assert source_cum_lengths[-1] <= source_length, f"{source_cum_lengths[-1]} > {source_length}"
    lower_target_indices = target_cum_lengths - target_lengths
    lower_source_indices = source_cum_lengths - source_lengths
    for i, (tl, tu, sl, su) in enumerate(zip(
            lower_target_indices,
            target_cum_lengths,
            lower_source_indices,
            source_cum_lengths
    )):
        block_mask[tl:tu, sl:su] = 0
    block_mask *= _MASK_VALUE
    return torch.from_numpy(block_mask).to(device=device)


def multi_node2seq_encoder_outputs_from_graph(
        g: dgl.DGLHeteroGraph,
        decoder_node_type: str,
        context_node_types: List[str],
        hidden_feature: str,
        align_positions_with: Optional[str] = None
) -> Tuple[Dict[str, List[List[torch.Tensor]]], Dict[str, List[torch.Tensor]], Optional[List[torch.Tensor]]]:
    encoder_outputs = {}
    encoder_lengths = {}
    encoder_positions = None

    decoder_nodes = g.nodes(decoder_node_type)
    assert len(decoder_nodes) == decoder_nodes[-1] + 1

    if align_positions_with is not None:
        assert align_positions_with in context_node_types or align_positions_with == decoder_node_type, \
            f"expected the node type to align position with {align_positions_with} " \
            f"to be in the context node types {context_node_types}"
        assert "position" in g.node_attr_schemes(align_positions_with), \
            f"need position attribute for node type {align_positions_with} to get the position"

    # handle when decoder node type is a grouped node type
    if "group" in g.node_attr_schemes(decoder_node_type):
        # start = time.perf_counter()
        batch_enc_feat = []
        batch_enc_len = []
        batch_num_nodes = tensor_to_python(g.batch_num_nodes(decoder_node_type), force_list=True)
        batch_groups = torch.split(g.nodes[decoder_node_type].data["group"], batch_num_nodes)
        batch_feat = torch.split(g.nodes[decoder_node_type].data[hidden_feature], batch_num_nodes)
        for groups, num_nodes, feats in zip(batch_groups, batch_num_nodes, batch_feat):
            unique_groups = torch.unique(groups, sorted=True)
            assert len(unique_groups) == unique_groups[-1] + 1
            grouped = [[] for _ in range(len(unique_groups))]

            for group, feat in zip(groups, feats):
                grouped[group].append(feat)

            enc_feat = []
            enc_len = []
            for i, group_feats in enumerate(grouped):
                group_feats = torch.stack(group_feats)
                enc_feat.append(group_feats)
                enc_len.append(len(group_feats))

            batch_enc_feat.append(enc_feat)
            batch_enc_len.append(torch.tensor(enc_len, device=g.device, dtype=torch.long))

        # end = time.perf_counter()
        # print(f"grouping decoder feat took {(end - start) * 1000:.2f}ms")

        group_offsets = torch.repeat_interleave(
            torch.cumsum(
                torch.tensor([0] + [groups[-1] + 1 for groups in batch_groups[:-1]], device=g.device, dtype=torch.long),
                0
            ),
            g.batch_num_nodes(decoder_node_type),
            0
        )
        node_id_to_group = tensor_to_python(g.nodes[decoder_node_type].data["group"] + group_offsets, force_list=True)

        if align_positions_with is not None and align_positions_with == decoder_node_type:
            encoder_positions = [
                torch.stack([pos.min() for pos in torch.split(positions, tensor_to_python(lengths, force_list=True))])
                for lengths, positions in zip(
                    batch_enc_len, torch.split(g.nodes[decoder_node_type].data["position"], batch_num_nodes)
                )
            ]

    else:
        # split nodes to decode as single nodes
        batch_num_nodes = tensor_to_python(g.batch_num_nodes(decoder_node_type), force_list=True)
        batch_enc_len = torch.split(
            torch.ones(g.num_nodes(decoder_node_type), device=g.device, dtype=torch.long),
            batch_num_nodes
        )
        batch_enc_feat = [
            torch.split(feat, [1] * len(feat))
            for feat in
            torch.split(
                g.nodes[decoder_node_type].data[hidden_feature],
                batch_num_nodes
            )
        ]
        node_id_to_group = tensor_to_python(decoder_nodes, force_list=True)

        if align_positions_with is not None and align_positions_with == decoder_node_type:
            encoder_positions = torch.split(g.nodes[decoder_node_type].data["position"], batch_num_nodes)

    encoder_outputs[decoder_node_type] = batch_enc_feat
    encoder_lengths[decoder_node_type] = batch_enc_len

    batch_num_decoder_nodes = [len(enc_feat) for enc_feat in batch_enc_feat]
    assert sum(batch_num_decoder_nodes) == node_id_to_group[-1] + 1

    for node_type in context_node_types:
        if node_type == decoder_node_type:  # ignore decoder node type
            continue
        if g.num_nodes(node_type) == 0:
            continue

        # find correct edge type
        edge_types = []
        for from_, e_, to_ in g.canonical_etypes:
            if from_ == node_type and to_ == decoder_node_type:
                edge_types.append((from_, e_, to_))
        assert len(edge_types) == 1, \
            f"expected that there is exactly one edge type from {node_type} to " \
            f"{decoder_node_type}, but got {len(edge_types)}"
        e_type = edge_types[0]

        # get in-edges and split src node features accordingly
        src, dst = g.in_edges(decoder_nodes, etype=e_type)
        enc_feat = [[] for _ in range(sum(batch_num_decoder_nodes))]
        for feat, node_id in zip(g.nodes[node_type].data[hidden_feature][src.long()], dst):
            enc_feat[node_id_to_group[node_id]].append(feat)

        encoder_outputs[node_type] = split(
            [torch.stack(f) for f in enc_feat],
            batch_num_decoder_nodes
        )
        encoder_lengths[node_type] = split(
            torch.tensor([len(f) for f in enc_feat], device=g.device, dtype=torch.long),
            batch_num_decoder_nodes
        )

        if align_positions_with is not None and align_positions_with == node_type:
            positions = [[] for _ in range(sum(batch_num_decoder_nodes))]
            for pos, node_id in zip(g.nodes[node_type].data["position"][src.long()], dst):
                positions[node_id_to_group[node_id]].append(pos)
            positions = torch.stack([torch.stack(pos).min() for pos in positions])
            encoder_positions = torch.split(
                positions,
                batch_num_decoder_nodes
            )

    return encoder_outputs, encoder_lengths, None if align_positions_with is None else encoder_positions


def encoder_positions_from_graph(
        g: dgl.DGLHeteroGraph,
        decoder_node_type: str,
        align_positions_with: str
) -> List[torch.Tensor]:
    batch_num_nodes = tensor_to_python(g.batch_num_nodes(align_positions_with), force_list=True)
    batch_positions = torch.split(g.nodes[align_positions_with].data["position"], batch_num_nodes)
    decode_grouped = "group" in g.node_attr_schemes(decoder_node_type)
    if align_positions_with == decoder_node_type:
        if decode_grouped:
            batch_groups = torch.split(g.nodes[decoder_node_type].data["group"], batch_num_nodes)
        else:
            return batch_positions
    else:
        if decode_grouped:
            batch_groups = []
        else:
            batch_groups = [list(range(num_nodes)) for num_nodes in batch_num_nodes]

    batch_pos = []
    for pos, groups in zip(batch_positions, batch_groups):
        _, group_lengths = torch.unique(groups, sorted=True, return_counts=True)
        batch_pos.append(torch.split(pos, tensor_to_python(group_lengths, force_list=True)))

    return [
        torch.stack([torch.amin(pos, 0) for pos in positions])
        for positions in batch_pos
    ]


def multi_node2seq_encoder_outputs_from_graph2(
        g: dgl.DGLHeteroGraph,
        decoder_node_type: str,
        context_node_types: List[str],
        hidden_feature: str,
        align_positions_with: Optional[str] = None
) -> Tuple[Dict[str, List[List[torch.Tensor]]], Dict[str, List[torch.Tensor]], Optional[List[torch.Tensor]]]:
    encoder_outputs = {}
    encoder_lengths = {}
    encoder_positions = None

    decoder_nodes = g.nodes(decoder_node_type)
    assert len(decoder_nodes) == decoder_nodes[-1] + 1

    if align_positions_with is not None:
        assert align_positions_with in context_node_types or align_positions_with == decoder_node_type, \
            f"expected the node type to align position with {align_positions_with} " \
            f"to be in the context node types {context_node_types} or equal to the decoder node type {decoder_node_type}"
        assert "position" in g.node_attr_schemes(align_positions_with), \
            f"need position attribute for node type {align_positions_with} to get the position"

    # handle when decoder node type is a grouped node type
    decode_grouped = "group" in g.node_attr_schemes(decoder_node_type)
    if decode_grouped:
        # start = time.perf_counter()
        batch_enc_feat = []
        batch_enc_len = []
        batch_num_nodes = tensor_to_python(g.batch_num_nodes(decoder_node_type), force_list=True)
        batch_groups = torch.split(g.nodes[decoder_node_type].data["group"], batch_num_nodes)
        batch_group_lengths = []
        batch_feat = torch.split(g.nodes[decoder_node_type].data[hidden_feature], batch_num_nodes)

        for groups, feat in zip(batch_groups, batch_feat):
            _, group_lengths = torch.unique(groups, sorted=True, return_counts=True)
            batch_group_lengths.append(group_lengths)
            assert sum(group_lengths) == len(groups)
            batch_enc_feat.append(torch.split(feat, tensor_to_python(group_lengths, force_list=True)))
            batch_enc_len.append(group_lengths)

        # end = time.perf_counter()
        # print(f"grouping decoder feat took {(end - start) * 1000:.2f}ms")
        if align_positions_with == decoder_node_type:
            batch_positions = torch.split(g.nodes[decoder_node_type].data["position"], batch_num_nodes)
            encoder_positions = [
                positions[
                    torch.cat([
                        torch.tensor([0], device=group_lengths.device), torch.cumsum(group_lengths[:-1], 0)
                    ])
                ]
                for positions, group_lengths in zip(batch_positions, batch_group_lengths)
            ]

    else:
        # split nodes to decode as single nodes
        batch_num_nodes = tensor_to_python(g.batch_num_nodes(decoder_node_type), force_list=True)
        batch_enc_len = torch.split(
            torch.ones(g.num_nodes(decoder_node_type), device=g.device, dtype=torch.long),
            batch_num_nodes
        )
        batch_enc_feat = [
            torch.split(feat, [1] * len(feat))
            for feat in
            torch.split(
                g.nodes[decoder_node_type].data[hidden_feature],
                batch_num_nodes
            )
        ]
        if align_positions_with == decoder_node_type:
            encoder_positions = torch.split(g.nodes[align_positions_with].data["position"], batch_num_nodes)

    encoder_outputs[decoder_node_type] = batch_enc_feat
    encoder_lengths[decoder_node_type] = batch_enc_len

    for node_type in context_node_types:
        if node_type == decoder_node_type:  # ignore decoder node type
            continue
        if g.num_nodes(node_type) == 0:
            continue

        # find correct edge type
        edge_types = []
        for from_, e_, to_ in g.canonical_etypes:
            if from_ == node_type and to_ == decoder_node_type:
                edge_types.append((from_, e_, to_))
        assert len(edge_types) == 1, \
            f"expected that there is exactly one edge type from {node_type} to " \
            f"{decoder_node_type}, but got {len(edge_types)}"
        e_type = edge_types[0]

        src, dst = g.in_edges(decoder_nodes, etype=e_type)
        src = src.long()
        dst = dst.long()

        if decode_grouped:
            dst_groups = g.nodes[decoder_node_type].data["group"][dst]
        else:
            dst_groups = dst

        batch_num_nodes = tensor_to_python(g.batch_num_nodes(node_type), force_list=True)
        # there should be exactly one edge form every src node to a decoder node
        assert len(src) == sum(batch_num_nodes) and src[-1] == sum(batch_num_nodes) - 1
        assert sum(batch_num_nodes) == len(dst_groups)

        batch_dst_groups = torch.split(dst_groups, batch_num_nodes)
        batch_dst_group_lengths = []
        batch_feat = torch.split(g.nodes[node_type].data[hidden_feature], batch_num_nodes)

        batch_enc_feat = []
        batch_enc_len = []
        for feat, dst_groups in zip(batch_feat, batch_dst_groups):
            _, dst_group_lengths = torch.unique(dst_groups, sorted=True, return_counts=True)
            batch_dst_group_lengths.append(dst_group_lengths)
            assert sum(dst_group_lengths) == len(dst_groups)
            batch_enc_feat.append(torch.split(feat, tensor_to_python(dst_group_lengths, force_list=True)))
            batch_enc_len.append(dst_group_lengths)

        encoder_outputs[node_type] = batch_enc_feat
        encoder_lengths[node_type] = batch_enc_len

        if align_positions_with == node_type:
            batch_positions = torch.split(g.nodes[node_type].data["position"], batch_num_nodes)
            # assuming positions always increase within a group, this is faster than splitting into groups and
            # finding the min position
            encoder_positions = [
                positions[
                    torch.cat([
                        torch.tensor([0], device=group_lengths.device),
                        torch.cumsum(group_lengths[:-1], 0)
                    ])
                ]
                for positions, group_lengths in zip(batch_positions, batch_dst_group_lengths)
            ]

    return encoder_outputs, encoder_lengths, encoder_positions


def graph2seq_encoder_outputs_from_graph(
        g: dgl.DGLHeteroGraph,
        node_types: List[str],
        hidden_feature: str) -> \
        Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    encoder_outputs = {}
    encoder_lengths = {}
    for node_type in node_types:
        if g.num_nodes(node_type) == 0:
            continue
        enc_out, enc_len = split_and_pad(
            g.nodes[node_type].data[hidden_feature],
            g.batch_num_nodes(node_type)
        )
        encoder_outputs[node_type] = enc_out
        encoder_lengths[node_type] = enc_len

    return encoder_outputs, encoder_lengths


def flatten_and_pad(
        feat: torch.Tensor,
        max_num_features: Optional[int] = None) -> torch.Tensor:
    feat = torch.flatten(feat, 1)
    if max_num_features is not None:
        feat = F.pad(feat, (0, max_num_features - feat.shape[1]))
    return feat


def node_type_subgraph(g: dgl.DGLHeteroGraph, node_types: Set[str]) -> dgl.DGLHeteroGraph:
    batch_num_nodes_dict = {n_type: g.batch_num_nodes(n_type) for n_type in node_types}
    batch_num_edges_dict = {e_type: g.batch_num_edges(e_type) for e_type in g.canonical_etypes
                            if e_type[0] in node_types or e_type[1] in node_types}
    g: dgl.DGLHeteroGraph = dgl.node_type_subgraph(g, ntypes=list(node_types))
    g.set_batch_num_nodes(batch_num_nodes_dict)
    g.set_batch_num_edges(batch_num_edges_dict)
    return g


def edge_type_subgraph(g: dgl.DGLHeteroGraph, edge_types: Set[Tuple[str, str, str]]) -> dgl.DGLHeteroGraph:
    node_types = set()
    for from_, _, to_ in edge_types:
        node_types.add(from_)
        node_types.add(to_)
    batch_num_nodes_dict = {n_type: g.batch_num_nodes(n_type) for n_type in node_types}
    batch_num_edges_dict = {e_type: g.batch_num_edges(e_type) for e_type in edge_types}
    g: dgl.DGLHeteroGraph = dgl.edge_type_subgraph(g, etypes=list(edge_types))
    g.set_batch_num_nodes(batch_num_nodes_dict)
    g.set_batch_num_edges(batch_num_edges_dict)
    return g


def transfer_features(from_g: dgl.DGLHeteroGraph, to_g: dgl.DGLHeteroGraph, feat: str) -> dgl.DGLHeteroGraph:
    to_g.ndata[feat] = from_g.ndata[feat]
    return to_g
