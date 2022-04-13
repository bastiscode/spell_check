from typing import Tuple, Any, Dict, List, Optional, Union, Set

import dgl
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import PackedSequence

from gnn_lib.utils import to


class GraphEncoderMixin(nn.Module):
    def encode(self, g: dgl.DGLHeteroGraph, **kwargs: Any) -> dgl.DGLHeteroGraph:
        raise NotImplementedError


class TensorEncoderMixin(nn.Module):
    def encode(self, x: torch.Tensor, **kwargs: Any) -> torch.Tensor:
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
        else (
            f"cuda:{torch.distributed.get_rank()}"
            if torch.distributed.is_available() and torch.distributed.is_initialized()
            else "cuda"
        )
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
    return x, splits.cpu() if isinstance(splits, torch.Tensor) else torch.tensor(splits, dtype=torch.long)


def split_and_pack(inputs: torch.Tensor, splits: Union[List[int], torch.Tensor]) -> PackedSequence:
    inputs = torch.split(inputs,
                         tensor_to_python(splits, force_list=True) if isinstance(splits, torch.Tensor) else splits)
    return pack(inputs)


def pack(inputs: Union[torch.Tensor, List[torch.Tensor]], lengths: Optional[torch.Tensor] = None) -> PackedSequence:
    if lengths is not None:
        return torch.nn.utils.rnn.pack_padded_sequence(inputs, lengths, batch_first=True, enforce_sorted=False)
    else:
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
    mask = torch.zeros(x.shape[:2], dtype=torch.bool)
    for i, l in enumerate(lengths):
        mask[i, l:] = True
    return mask.to(x.device)


def square_causal_mask(length: int, device: torch.device) -> torch.Tensor:
    return torch.triu(torch.ones(length, length, dtype=torch.bool, device=device), diagonal=1) * _MASK_VALUE


def square_causal_block_mask(length: int, lengths: torch.Tensor, device: torch.device) -> torch.Tensor:
    triangular_mask = torch.triu(torch.ones((length, length), dtype=torch.bool), diagonal=1)

    block_mask = torch.ones((length, length), dtype=torch.bool)
    cum_lengths = torch.cumsum(lengths, 0)
    assert cum_lengths[-1] <= length, f"{cum_lengths[-1]} > {length}"
    lower_indices = cum_lengths - lengths
    for i, (l, u) in enumerate(zip(lower_indices, cum_lengths)):
        block_mask[:u, l:] = False

    mask = torch.logical_or(triangular_mask, block_mask) * _MASK_VALUE
    return mask.to(dtype=torch.float, device=device)


def rectangular_block_mask(
        target_length: int,
        source_length: int,
        target_lengths: torch.Tensor,
        source_lengths: torch.Tensor,
        device: torch.device) -> torch.Tensor:
    block_mask = torch.ones((target_length, source_length), dtype=torch.float)
    target_cum_lengths = torch.cumsum(target_lengths, 0)
    source_cum_lengths = torch.cumsum(source_lengths, 0)
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
    return block_mask.to(device=device, non_blocking=True)


def encoder_outputs_from_graph(
        g: dgl.DGLHeteroGraph,
        node_types: List[str],
        hidden_feature: str
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    encoder_outputs = {}
    encoder_lengths = {}
    for node_type in node_types:
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


def get_additional_features_and_aggregations_from_group_stages(
        stages: List[Dict[str, Any]],
        default_aggregation: str = "mean"
) -> Tuple[Dict[str, int], Dict[str, str]]:
    additional_features = {}
    aggregation = {}
    for stage in stages:
        stage_name = stage["stage"]
        if "features" in stage:
            additional_features[stage_name] = stage["features"].shape[1]
        aggregation[stage_name] = stage.get("aggregation", default_aggregation)

    return additional_features, aggregation


def group_features(
        grouped_feats: List[torch.Tensor],
        groups: List[Dict[str, Any]],
        additional_feature_encoder: Optional[nn.Module] = None,
        aggregation: Optional[str] = None
) -> List[Union[torch.Tensor, List[torch.Tensor]]]:
    aggregation = aggregation or "mean"

    has_additional_features = additional_feature_encoder is not None
    batch_groups = []
    batch_group_lengths = []
    stage_features = []
    for group in groups:
        batch_groups.append(group["groups"])
        batch_group_lengths.append(len(group["groups"]))
        if has_additional_features:
            stage_features.append(group["features"])

    if has_additional_features:
        all_grouped_feats = torch.cat(grouped_feats, dim=0)
        grouped_feats = torch.split(
            additional_feature_encoder(
                torch.cat(
                    [
                        all_grouped_feats,
                        to(torch.cat(stage_features, dim=0), all_grouped_feats.device)
                    ], dim=1)
            ),
            batch_group_lengths
        )

    assert all(
        len(group_feat) == group_length for group_feat, group_length in zip(grouped_feats, batch_group_lengths)
    )

    new_grouped_feats = []
    for batch_group, batch_grouped_feat in zip(batch_groups, grouped_feats):
        # filter out invalid groups (marked with -1 or anything else smaller zero)
        valid_groups = batch_group >= 0
        batch_group = batch_group[valid_groups]
        batch_grouped_feat = batch_grouped_feat[valid_groups]
        _, batch_group_splits = torch.unique(batch_group, return_counts=True, sorted=True)
        new_batch_grouped_feat = []
        for group_feat in torch.split(batch_grouped_feat, batch_group_splits.tolist()):
            if aggregation == "mean":
                group_feat = torch.mean(group_feat, dim=0, keepdim=True)
            elif aggregation == "max":
                group_feat = torch.max(group_feat, dim=0, keepdim=True).values
            elif aggregation == "sum":
                group_feat = torch.sum(group_feat, dim=0, keepdim=True)
            elif aggregation == "stack":
                pass
            else:
                raise ValueError(f"unknown group aggregation {aggregation}")
            new_batch_grouped_feat.append(group_feat)
        if aggregation == "stack":
            new_grouped_feats.append(new_batch_grouped_feat)
        else:
            new_grouped_feats.append(torch.cat(new_batch_grouped_feat, dim=0))

    return new_grouped_feats
