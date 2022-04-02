from typing import Dict, Union, List, Any, Tuple

import dgl
import torch
from torch.nn import functional as F

from gnn_lib import tasks, models
from gnn_lib.data import utils as data_utils
from gnn_lib.data.utils import Sample
from gnn_lib.modules import utils
from gnn_lib.tasks import utils as task_utils
from gnn_lib.utils import data_containers, Batch, to
from gnn_lib.utils.distributed import DistributedDevice


class GraphClassification(tasks.Task):
    expected_models = models.ModelForGraphClassification

    def _get_additional_stats(self, model: models.ModelForGraphClassification) -> \
            Dict[str, data_containers.DataContainer]:
        return {
            "accuracy": data_containers.AverageScalarContainer(name="accuracy")
        }

    def _prepare_inputs_and_labels(self,
                                   batch: Batch,
                                   device: DistributedDevice) -> Tuple[Dict[str, Any], Any]:
        labels = to(torch.cat(batch.info.pop("label")), device.device)

        return {"g": batch.data, **batch.info}, labels

    def _calc_loss(self,
                   labels: torch.Tensor,
                   model_output: torch.Tensor,
                   additional_losses: Dict[str, torch.Tensor]) -> torch.Tensor:
        return F.cross_entropy(model_output, labels) + sum(additional_losses.values())

    def _update_stats(self,
                      model: models.ModelForGraphClassification,
                      inputs: Dict[str, Any],
                      labels: torch.Tensor,
                      model_output: torch.Tensor,
                      stats: Dict[str, data_containers.DataContainer],
                      step: int,
                      total_steps: int) -> None:
        predictions = torch.argmax(model_output, dim=1)
        stats["accuracy"].add((labels == predictions).cpu())

    @torch.inference_mode()
    def inference(
            self,
            model: models.ModelForGraphClassification,
            inputs: List[Union[str, Sample]],
            **kwargs: Any
    ) -> List:
        self._check_model(model)
        model = model.eval()

        threshold = kwargs.get("threshold", 0.5)
        temperature = kwargs.get("temperature", 1.0)

        batch = self._batch_sequences_for_inference(inputs)
        outputs, _ = model(batch.data, **batch.info)

        return_logits = kwargs.get("return_logits", False)
        if return_logits:
            return utils.tensor_to_python(outputs, force_list=True)
        else:
            return utils.tensor_to_python(
                task_utils.class_predictions(outputs, threshold, temperature), force_list=True
            )
