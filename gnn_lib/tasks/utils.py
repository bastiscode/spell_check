import collections
import copy
from typing import List, Union, Optional, Set, Dict, Any

import dgl
import torch
from torch import nn

from gnn_lib.modules.utils import tensor_to_python, split
from gnn_lib.utils import DATA_INPUT, BATCH

SAMPLE_SEQUENCE = \
    "But I must explain to you how all this mistaken idea of denouncing pleasure and praising pain was " \
    "born and I will give you a complete account of the system, and expound the actual teachings of the great" \
    " explorer of the truth, the master-builder of human happiness. No one rejects, dislikes, or avoids pleas" \
    "ure itself, because it is pleasure, but because those who do not know how to pursue pleasure rationally" \
    " encounter consequences that are extremely painful."


def class_predictions(logits: torch.Tensor, threshold: float = 0.5, temperature: float = 1.0) -> torch.Tensor:
    if logits.shape[-1] > 2:
        # multi-class classification
        return torch.argmax(logits, dim=-1).long()
    else:
        # binary classification
        softmax = torch.softmax(logits / temperature, dim=-1)
        return (softmax[..., -1] >= threshold).long()


def class_voting(
        logits: List[Union[List, torch.Tensor]],
        method: str,
        thresholds: Optional[List[float]] = None,
        temperatures: Optional[List[float]] = None
) -> torch.Tensor:
    if temperatures is None:
        temperatures = [1.0] * len(logits)
    else:
        assert len(temperatures) == len(logits)
    if thresholds is None:
        thresholds = [0.5] * len(logits)
    else:
        assert len(thresholds) == len(logits)

    logits = [torch.tensor(l, dtype=torch.float) if not isinstance(l, torch.Tensor) else l for l in logits]

    if method == "hard":  # predict the class with the most individual predictions
        predictions = torch.stack([
            class_predictions(l, threshold=thres, temperature=temp)
            for l, thres, temp in zip(logits, thresholds, temperatures)
        ], dim=-1)
        most_freq_classes = torch.mode(predictions, dim=-1).values
        return most_freq_classes
    elif method == "soft":  # predict the class with the highest total probability (requires good calibrated models)
        soft_predictions = [torch.softmax(l / t, dim=-1) for l, t in zip(logits, temperatures)]
        summed_predictions = torch.stack(soft_predictions).sum(dim=0)
        return torch.argmax(summed_predictions, dim=-1).long()
    else:
        raise ValueError(f"Unknown method {method}, should be one of [hard, soft]")


class EMA:
    def __init__(self, model: nn.Module, ema_factor: float) -> None:
        self.model = model
        self.ema_model: nn.Module = copy.deepcopy(model)
        for param in self.ema_model.parameters():
            param.detach_()
        self.ema_model = self.ema_model.eval()

        self.ema_factor = ema_factor
        self.one_minus_ema_factor = 1 - ema_factor

    def update(self, overwrite: bool = False):
        for ema_param, model_param in zip(self.ema_model.parameters(), self.model.parameters()):
            if overwrite:
                ema_param.data = model_param.data
            else:
                ema_param.mul_(self.ema_factor).add_(model_param.data, alpha=self.one_minus_ema_factor)

        for ema_buffer, model_buffer in zip(self.ema_model.buffers(), self.model.buffers()):
            ema_buffer.data = model_buffer.data


def get_token_ids_from_graphs(graph: dgl.DGLHeteroGraph) -> List[List[int]]:
    return split(
        tensor_to_python(graph.nodes["token"].data["token_id"], force_list=True),
        tensor_to_python(graph.batch_num_nodes("token"), force_list=True)
    )


def get_batch_size_from_data(data: DATA_INPUT) -> int:
    if isinstance(data, dgl.DGLHeteroGraph):
        return data.batch_size
    else:
        return len(data)


def is_string_input(inputs: Union[List[str], BATCH]):
    return isinstance(inputs, list) and all(isinstance(ipt, str) for ipt in inputs)


def get_unused_parameters(model: nn.Module, **inputs: Any) -> Set[str]:
    def _sum(item: Union[torch.Tensor, List, Dict]) -> Union[torch.Tensor, List, Dict]:
        if isinstance(item, torch.Tensor):
            return item.sum()
        elif isinstance(item, list):
            return sum([_sum(i) for i in item])
        elif isinstance(item, dict):
            return sum(_sum(v) for v in item.values())
        else:
            raise RuntimeError("expected model output to be any nesting of dicts, lists and tensors, "
                               f"but got {type(item)}")

    outputs, _ = model(**inputs)
    loss = _sum(outputs)

    loss.backward()

    unused_parameters = set()
    for name, p in model.named_parameters():
        if p.grad is None and p.requires_grad:
            unused_parameters.add(name)

    return unused_parameters
