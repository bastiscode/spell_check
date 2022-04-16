from typing import List, Any, Union, Optional

import torch

from gnn_lib import models
from gnn_lib.data.utils import Sample
from gnn_lib.tasks.graph2seq import Graph2Seq
from gnn_lib.utils import Batch


class GraphSECNMT(Graph2Seq):
    @torch.inference_mode()
    def inference(
            self,
            model: models.ModelForGraph2Seq,
            inputs: Union[Batch, List[Union[str, Sample]]],
            input_strings: Optional[List[str]] = None,
            **kwargs: Any
    ) -> List[List[str]]:
        if isinstance(inputs, Batch):
            assert input_strings is not None
            batch = inputs
            input_strings = [model.input_tokenizers["token"].normalize(ipt) for ipt in input_strings]
        else:
            batch = self._batch_sequences_for_inference(inputs)
            input_strings = [model.input_tokenizers["token"].normalize(str(ipt)) for ipt in inputs]

        return super().inference(model, batch, input_strings=input_strings, **kwargs)
