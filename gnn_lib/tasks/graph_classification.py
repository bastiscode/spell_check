from typing import Dict, Union, List, Any, Tuple

import dgl
import torch
from torch.nn import functional as F

from gnn_lib import tasks, models
from gnn_lib.data import utils as data_utils
from gnn_lib.modules import utils
from gnn_lib.tasks import utils as task_utils
from gnn_lib.utils import data_containers, BATCH, to
from gnn_lib.utils.distributed import DistributedDevice


class GraphClassification(tasks.Task):
    expected_models = models.ModelForGraphClassification

    def _get_additional_stats(self, model: models.ModelForGraphClassification) -> \
            Dict[str, data_containers.DataContainer]:
        return {
            "accuracy": data_containers.AverageScalarContainer(name="accuracy")
        }

    def _prepare_inputs_and_labels(self,
                                   batch: BATCH,
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
            inputs: Union[List[str], BATCH],
            **kwargs: Any
    ) -> List:
        self._check_model(model)
        model = model.eval()

        threshold = kwargs.get("threshold", 0.5)
        temperature = kwargs.get("temperature", 1.0)

        got_str_input = isinstance(inputs, list) and isinstance(inputs[0], str)
        if got_str_input:
            batch = self.variant.prepare_sequences_for_inference(inputs)
        else:
            batch = inputs

        oversized = (batch.data.batch_num_nodes("token") > model.cfg.max_length).tolist()

        if sum(oversized):
            # print(f"found {sum(oversized)} sequences that are too long: {batch.data.batch_num_nodes('token')}")
            data = dgl.batch([graph for i, graph in enumerate(dgl.unbatch(batch.data)) if not oversized[i]])
            info = {k: [v_ for i, v_ in enumerate(v) if not oversized[i]] for k, v in batch.info.items()}
            batch = BATCH(data, info)

        outputs, _ = model(batch.data, **batch.info)

        return_logits = kwargs.get("return_logits", False)
        if return_logits:
            return utils.tensor_to_python(outputs, force_list=True)
        else:
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
