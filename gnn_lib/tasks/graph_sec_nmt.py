from typing import Union, List, Any

import dgl
import torch

from gnn_lib import models
from gnn_lib.tasks.graph2seq import Graph2Seq


class GraphSECNMT(Graph2Seq):
    @torch.inference_mode()
    def inference(self,
                  model: models.ModelForGraph2Seq,
                  inputs: Union[List[str], dgl.DGLHeteroGraph],
                  **kwargs: Any) -> List[List[str]]:
        assert isinstance(inputs, list) and isinstance(inputs[0], str)
        kwargs.update({
            "input_strings": inputs
        })

        return super().inference(model, inputs, **kwargs)
