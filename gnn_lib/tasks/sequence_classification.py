from typing import Dict, List, Union, Any, Tuple

import torch
from torch.nn import functional as F

from gnn_lib import tasks, models
from gnn_lib.modules import utils
from gnn_lib.tasks import utils as task_utils
from gnn_lib.utils import data_containers, BATCH, to


class SequenceClassification(tasks.Task):
    expected_models = models.ModelForSequenceClassification

    def _get_additional_stats(self, model: models.ModelForSequenceClassification) \
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
            batch: BATCH,
            device: torch.device
    ) -> Tuple[Dict[str, Any], Any]:
        # extract labels from info dict
        labels = to(torch.cat(batch.info.pop("label")), device)

        return {"x": batch.data, **batch.info}, labels

    def _calc_loss(self,
                   labels: torch.Tensor,
                   model_output: torch.Tensor,
                   additional_losses: Dict[str, torch.Tensor]) -> torch.Tensor:
        return F.cross_entropy(model_output, labels) + sum(additional_losses.values())

    def _update_stats(self,
                      model: models.ModelForSequenceClassification,
                      inputs: Dict[str, Any],
                      labels: torch.Tensor,
                      model_output: torch.Tensor,
                      stats: Dict[str, data_containers.DataContainer],
                      step: int,
                      total_steps: int) -> None:
        sequence_length_container = stats["seq_length"]
        sequence_length_container.add([len(t) for t in inputs["x"]])

        predictions = torch.argmax(model_output, dim=1)
        stats["accuracy"].add((labels == predictions).cpu())

    @torch.inference_mode()
    def inference(
            self,
            model: models.ModelForTokenClassification,
            inputs: Union[List[str], BATCH],
            **kwargs: Any
    ) -> List[Dict[str, List]]:
        self._check_model(model)
        model = model.eval()

        threshold = kwargs.get("threshold", 0.5)
        temperature = kwargs.get("temperature", 1.0)

        got_str_input = task_utils.is_string_input(inputs)
        if got_str_input:
            batch = self.variant.prepare_sequences_for_inference(inputs)
        else:
            batch = inputs

        oversized = [len(t) > model.cfg.max_length for t in batch.data]
        if sum(oversized):
            # print(f"found {sum(oversized)} sequences that are too long: {[len(t) for t in batch.data]}")
            data = [t for i, t in enumerate(batch.data) if not oversized[i]]
            info = {k: [v_ for i, v_ in enumerate(v) if not oversized[i]] for k, v in batch.info.items()}
            batch = BATCH(data, info)

        outputs, _ = model(batch.data, **batch.info)

        return_logits = kwargs.get("return_logits", False)

        predictions = utils.tensor_to_python(
            task_utils.class_predictions(outputs, threshold, temperature), force_list=True
        )
        prediction_idx = 0
        outputs = []
        for is_oversized in oversized:
            if not is_oversized:
                outputs.append(predictions[prediction_idx])
                prediction_idx += 1
            else:
                outputs.append(0)
        assert prediction_idx == len(predictions)
        return outputs
