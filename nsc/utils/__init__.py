from typing import List, Union, Dict, Any

import dgl
import torch

TensorInput = torch.Tensor  # maybe extend this with Dict[str, torch.Tensor] in the future
DataInput = Union[List[TensorInput], dgl.DGLHeteroGraph]
InfoInput = Dict[str, List[Any]]


def pin(item: Union[dgl.DGLHeteroGraph, torch.Tensor, Dict, List]) \
        -> Union[dgl.DGLHeteroGraph, torch.Tensor, Dict, List]:
    if isinstance(item, dgl.DGLHeteroGraph):
        # item.pin_memory_() # causes some weird errors, uncomment for now
        return item
    elif isinstance(item, torch.Tensor):
        return item.pin_memory()
    elif isinstance(item, list):
        return [pin(i) for i in item]
    elif isinstance(item, dict):
        return {k: pin(v) for k, v in item.items()}
    else:
        return item


def to(item: Union[dgl.DGLHeteroGraph, torch.Tensor, Dict, List], device: torch.device) \
        -> Union[dgl.DGLHeteroGraph, torch.Tensor, Dict, List]:
    if isinstance(item, dgl.DGLHeteroGraph):
        return item.to(device, non_blocking=True)
    elif isinstance(item, torch.Tensor):
        return item.to(device, non_blocking=True)
    elif isinstance(item, list):
        return [to(i, device) for i in item]
    elif isinstance(item, dict):
        return {k: to(v, device) for k, v in item.items()}
    else:
        return item


class Batch:
    def __init__(self, data: DataInput, info: InfoInput) -> None:
        self.data = data
        self.info = info

    # define a pin memory method, so we can also use this class in a pytorch dataloader with pin_memory=True
    def pin_memory(self):
        self.data = pin(self.data)
        self.info = pin(self.info)
        return self
