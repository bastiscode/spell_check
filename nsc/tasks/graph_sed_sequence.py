from typing import Dict, Any, Tuple, List

import torch

from nsc import models
from nsc.tasks.graph_classification import GraphClassification
from nsc.utils import data_containers
from nsc.data import utils as data_utils
from nsc.tasks import utils as task_utils


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

    def _split_sample_for_inference(
            self,
            sample: data_utils.Sample,
            max_length: int,
            context_length: int,
            **kwargs: Any
    ) -> List[Tuple[int, int, int, int]]:
        return task_utils.get_word_windows(sample, max_length, context_length)

    def _merge_inference_outputs(
            self,
            sequence: str,
            infos: List[data_utils.InferenceInfo],
            predictions: List[int],
            **kwargs: Any
    ) -> int:
        return task_utils.merge_sed_sequence_outputs(predictions)
