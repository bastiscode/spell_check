from typing import Dict, Any

import torch

from gnn_lib import models
from gnn_lib.data import tokenization
from gnn_lib.tasks import utils as task_utils
from gnn_lib.tasks.graph_classification import GraphClassification
from gnn_lib.utils import data_containers


class SEDSequence(GraphClassification):
    def _get_additional_stats(self, model: models.ModelForGraphClassification) -> \
            Dict[str, data_containers.DataContainer]:
        stats = super()._get_additional_stats(model)
        stats["text"] = data_containers.MultiTextContainer(name="text_samples", max_samples=4)
        return stats

    def _update_stats(self,
                      model: models.ModelForGraphClassification,
                      inputs: Dict[str, Any],
                      labels: torch.Tensor,
                      model_output: torch.Tensor,
                      stats: Dict[str, data_containers.DataContainer],
                      step: int,
                      log_every: int) -> None:
        super()._update_stats(model, inputs, labels, model_output, stats, step, log_every)
        text_container = stats["text"]
        if step % max(log_every // text_container.max_samples, 1) != 0 or \
                len(text_container.samples) >= text_container.max_samples:
            return

        token_ids = task_utils.get_token_ids_from_graphs(inputs["g"])
        tokenizer: tokenization.Tokenizer = model.tokenizers["token"]
        input_text = tokenizer.de_tokenize(token_ids[0])

        labels_str = str(labels[0].item())
        pred_str = str(torch.argmax(model_output[0], 0).item())

        text_container.add(
            f""""
    input:\t{input_text}
    label:\t{labels_str}
    pred:\t{pred_str}
---
            """
        )
