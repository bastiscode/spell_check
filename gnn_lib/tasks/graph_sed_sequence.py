from typing import Dict, Any

import torch

from gnn_lib import models
from gnn_lib.tasks.graph_classification import GraphClassification
from gnn_lib.utils import data_containers


class GraphSEDSequence(GraphClassification):
    def _get_additional_stats(self, model: models.ModelForGraphClassification) -> \
            Dict[str, data_containers.DataContainer]:
        stats = super()._get_additional_stats(model)
        stats["fpr"] = data_containers.F1PrecRecContainer(name="f1_prec_rec", class_names={1: "sequence"})
        return stats

    def _update_stats(self,
                      model: models.ModelForGraphClassification,
                      inputs: Dict[str, Any],
                      labels: torch.Tensor,
                      model_output: torch.Tensor,
                      stats: Dict[str, data_containers.DataContainer],
                      step: int,
                      total_steps: int) -> None:
        super()._update_stats(model, inputs, labels, model_output, stats, step, total_steps)

        predictions = torch.argmax(model_output, dim=1)
        stats["fpr"].add((labels.cpu(), predictions.cpu()))
