from typing import List, Union, Dict, Any, Tuple

import dgl
import torch

TENSOR_INPUT = torch.Tensor  # maybe extend this with Dict[str, torch.Tensor] in the future
DATA_INPUT = Union[List[TENSOR_INPUT], dgl.DGLHeteroGraph]
INFO_INPUT = Dict[str, List[Any]]
MODEL_INPUTS = Tuple[DATA_INPUT, INFO_INPUT]
