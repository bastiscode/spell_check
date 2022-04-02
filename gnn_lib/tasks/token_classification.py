from typing import Dict, List, Union, Any, Tuple

import torch
from torch.nn import functional as F

from gnn_lib import tasks, models
from gnn_lib.data.utils import Sample
from gnn_lib.modules import utils
from gnn_lib.tasks import utils as task_utils
from gnn_lib.utils import data_containers, Batch, to


class TokenClassification(tasks.Task):
    expected_models = models.ModelForTokenClassification

    def _get_additional_stats(self, model: models.ModelForTokenClassification) \
            -> Dict[str, data_containers.DataContainer]:
        stats = {
            "accuracy": data_containers.AverageScalarContainer(name="accuracy"),
            "seq_length": data_containers.HistogramContainer(
                name="input_sequence_length"
            )
        }
        return stats

    def _prepare_inputs_and_labels(
            self,
            batch: Batch,
            device: torch.device
    ) -> Tuple[Dict[str, Any], Any]:
        # extract labels from info dict
        labels = to(torch.cat(batch.info.pop("label")), device)

        return {"x": batch.data, **batch.info}, labels

    def _calc_loss(self,
                   labels: torch.Tensor,
                   model_output: List[torch.Tensor],
                   additional_losses: Dict[str, torch.Tensor]) -> torch.Tensor:
        return F.cross_entropy(torch.cat(model_output, dim=0), labels) + sum(additional_losses.values())

    def _update_stats(self,
                      model: models.ModelForTokenClassification,
                      inputs: Dict[str, Any],
                      labels: torch.Tensor,
                      model_output: List[torch.Tensor],
                      stats: Dict[str, data_containers.DataContainer],
                      step: int,
                      total_steps: int) -> None:
        sequence_length_container = stats["seq_length"]
        sequence_length_container.add([len(t) for t in inputs["x"]])

        predictions = torch.argmax(torch.cat(model_output, dim=0), dim=1)
        stats["accuracy"].add((labels == predictions).cpu())

    @torch.inference_mode()
    def inference(
            self,
            model: models.ModelForTokenClassification,
            inputs: List[Union[str, Sample]],
            **kwargs: Any
    ) -> List[List]:
        self._check_model(model)
        model = model.eval()

        batch = self._batch_sequences_for_inference(inputs)
        outputs, _ = model(batch.data, **batch.info)

        return_logits = kwargs.get("return_logits", False)
        if return_logits:
            predictions = [utils.tensor_to_python(output, force_list=True) for output in outputs]
        else:
            threshold = kwargs.get("threshold", 0.5)
            temperature = kwargs.get("temperature", 1.0)
            predictions = [
                utils.tensor_to_python(
                    task_utils.class_predictions(
                        output,
                        threshold,
                        temperature
                    ),
                    force_list=True
                ) for output in outputs
            ]

        return predictions
