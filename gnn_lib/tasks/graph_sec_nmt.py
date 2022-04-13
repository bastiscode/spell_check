from typing import List, Any, Union

import torch

from gnn_lib import models
from gnn_lib.data.utils import Sample
from gnn_lib.tasks.graph2seq import Graph2Seq


class GraphSECNMT(Graph2Seq):
    @torch.inference_mode()
    def inference(self,
                  model: models.ModelForGraph2Seq,
                  inputs: List[Union[str, Sample]],
                  **kwargs: Any) -> List[List[str]]:
        kwargs.update({
            "input_strings": [model.input_tokenizers["token"].normalize(str(ipt)) for ipt in inputs]
        })

        return super().inference(model, inputs, **kwargs)
