from typing import Dict, List, Union, Any, Tuple

import torch
from torch.nn import functional as F

from gnn_lib import tasks, models
from gnn_lib.modules import utils
from gnn_lib.tasks import utils as task_utils
from gnn_lib.utils import data_containers, BATCH, to
from gnn_lib.utils.distributed import DistributedDevice


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

    def _prepare_inputs_and_labels(self,
                                   batch: BATCH,
                                   device: DistributedDevice) -> Tuple[Dict[str, Any], Any]:
        # extract labels from info dict
        labels = to(torch.cat(batch.info.pop("label")), device.device)

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

        got_str_input = isinstance(inputs, list) and isinstance(inputs[0], str)
        if got_str_input:
            g, infos = self.variant.prepare_sequences_for_inference(inputs)
        else:
            g, infos = inputs

        outputs, _ = model(g, **infos)

        return_logits = kwargs.get("return_logits", False)

        batch_predictions_dict = {}
        for node_type in model.cfg.num_classes:
            num_nodes = g.batch_num_nodes(node_type)
            if "groups" in infos and node_type in infos["groups"][0]:
                num_nodes = [max(group[node_type][-1]["groups"]) + 1 for group in infos["groups"]]

            if return_logits:
                predictions = utils.tensor_to_python(outputs[node_type], force_list=True)
            else:
                threshold = kwargs.get(f"{node_type}_threshold", kwargs.get("threshold", 0.5))
                temperature = kwargs.get(f"{node_type}_temperature", kwargs.get("temperature", 1.0))
                predictions = utils.tensor_to_python(
                    task_utils.class_predictions(
                        outputs[node_type],
                        threshold,
                        temperature
                    ),
                    force_list=True
                )

            predictions = utils.split(
                predictions,
                num_nodes
            )
            batch_predictions_dict[node_type] = predictions

        return [
            {node_type: predictions[i] for node_type, predictions in batch_predictions_dict.items()}
            for i in range(g.batch_size)
        ]
