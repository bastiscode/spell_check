from typing import Dict, List, Any, Union

import torch

from gnn_lib import models
from gnn_lib.tasks.token_classification import TokenClassification
from gnn_lib.utils import data_containers, BATCH
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

    def inference(
            self,
            model: models.ModelForTokenClassification,
            inputs: Union[List[str], BATCH],
            **kwargs: Any) -> List[List[int]]:
        assert task_utils.is_string_input(inputs)
        batch = self.variant.prepare_sequences_for_inference(inputs)

        oversized = [len(t) > model.cfg.max_length for t in batch.data]
        if sum(oversized):
            # print(f"found {sum(oversized)} sequences that are too long: {[len(t) for t in batch.data]}")
            data = [t for i, t in enumerate(batch.data) if not oversized[i]]
            info = {k: [v_ for i, v_ in enumerate(v) if not oversized[i]] for k, v in batch.info.items()}
            batch = BATCH(data, info)

        predictions = super().inference(model, batch, **kwargs)

        prediction_idx = 0
        outputs = []
        for ipt, is_oversized in zip(inputs, oversized):
            if not is_oversized:
                outputs.append(predictions[prediction_idx])
                prediction_idx += 1
            else:
                outputs.append([0] * len(ipt.split()))
        assert prediction_idx == len(predictions)
        return outputs
