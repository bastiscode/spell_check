from typing import List, Any

import torch
from gnn_lib import models
from gnn_lib.tasks.token2seq import Token2Seq


class SECWordsNMT(Token2Seq):
    @torch.inference_mode()
    def inference(self,
                  model: models.ModelForToken2Seq,
                  inputs: List[str],
                  **kwargs: Any) -> List[List[str]]:
        assert isinstance(inputs, list) and isinstance(inputs[0], str)

        raise NotImplementedError
