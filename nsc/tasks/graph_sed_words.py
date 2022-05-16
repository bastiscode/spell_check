from typing import Dict, List, Union, Any, Tuple

import torch

from nsc import models
from nsc.data.utils import Sample, InferenceInfo
from nsc.tasks import utils as task_utils
from nsc.tasks.multi_node_classification import MultiNodeClassification
from nsc.utils import data_containers, Batch


class GraphSEDWords(MultiNodeClassification):
    def _get_additional_stats(self, model: models.ModelForMultiNodeClassification) \
            -> Dict[str, data_containers.DataContainer]:
        node_type = "word" if self.variant.cfg.data_scheme == "word_graph" else "token"

        stats = {
            f"{node_type}_accuracy": data_containers.AverageScalarContainer(name="word_accuracy"),
            f"{node_type}_fpr": data_containers.F1PrecRecContainer(
                name="f1_prec_rec",
                class_names={1: "word"}
            )
        }
        return stats

    def _update_stats(self,
                      model: models.ModelForMultiNodeClassification,
                      inputs: Dict[str, Any],
                      labels: Dict[str, torch.Tensor],
                      model_output: Dict[str, torch.Tensor],
                      stats: Dict[str, data_containers.DataContainer],
                      step: int,
                      total_steps: int) -> None:
        super()._update_stats(model, inputs, labels, model_output, stats, step, total_steps)
        for node_type, pred in model_output.items():
            predictions = torch.argmax(pred, dim=1)
            stats[f"{node_type}_fpr"].add((labels[node_type].cpu(), predictions.cpu()))

    @torch.inference_mode()
    def inference(
            self,
            model: models.ModelForMultiNodeClassification,
            inputs: Union[Batch, List[Union[str, Sample]]],
            **kwargs: Any
    ) -> List[List[int]]:
        detections_dicts = super().inference(model, inputs, **kwargs)
        return [detections_dict[list(detections_dict)[0]] for detections_dict in detections_dicts]

    def _split_sample_for_inference(
            self,
            sample: Sample,
            max_length: int,
            context_length: int,
            **kwargs: Any
    ) -> List[Tuple[int, int, int, int]]:
        return task_utils.get_word_windows(sample, max_length, context_length)

    def _merge_inference_outputs(
            self,
            sequence: str,
            infos: List[InferenceInfo],
            predictions: List[List[int]],
            **kwargs: Any
    ) -> List[int]:
        return task_utils.merge_sed_words_outputs(sequence, infos, predictions)
