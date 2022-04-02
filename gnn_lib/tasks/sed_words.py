from typing import Dict, List, Any, Tuple

import torch

from gnn_lib import models
from gnn_lib.data.utils import Sample, InferenceInfo
from gnn_lib.tasks.token_classification import TokenClassification
from gnn_lib.utils import data_containers
from gnn_lib.tasks import utils as task_utils


class SEDWords(TokenClassification):
    def _get_additional_stats(self, model: models.ModelForTokenClassification) \
            -> Dict[str, data_containers.DataContainer]:
        stats = super()._get_additional_stats(model)
        stats["fpr"] = data_containers.F1PrecRecContainer(name="f1_prec_rec", class_names={1: "word"})
        return stats

    def _update_stats(self,
                      model: models.ModelForTokenClassification,
                      inputs: Dict[str, Any],
                      labels: torch.Tensor,
                      model_output: List[torch.Tensor],
                      stats: Dict[str, data_containers.DataContainer],
                      step: int,
                      total_steps: int) -> None:
        super()._update_stats(model, inputs, labels, model_output, stats, step, total_steps)

        predictions = torch.argmax(torch.cat(model_output, dim=0), dim=1)
        stats["fpr"].add((labels.cpu(), predictions.cpu()))

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
        assert all(p in {0, 1} for prediction in predictions for p in prediction)
        merged_prediction = []
        for info, prediction in zip(infos, predictions):
            assert len(sequence[info.ctx_start:info.ctx_end].split()) == len(prediction)
            num_left_context_words = len(sequence[info.ctx_start:info.window_start].split())
            num_window_words = len(sequence[info.window_start:info.window_end].split())
            merged_prediction.extend(prediction[num_left_context_words:num_left_context_words + num_window_words])
        assert len(merged_prediction) == len(sequence.split())
        return merged_prediction
