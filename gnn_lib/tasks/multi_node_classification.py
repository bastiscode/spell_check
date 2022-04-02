import collections
from typing import Dict, List, Union, Any, Tuple

import dgl
import torch
from torch.nn import functional as F

from gnn_lib import tasks, models
from gnn_lib.data.utils import Sample
from gnn_lib.models import Batch
from gnn_lib.modules import utils
from gnn_lib.tasks import utils as task_utils
from gnn_lib.utils import data_containers, to


class MultiNodeClassification(tasks.Task):
    expected_models = models.ModelForMultiNodeClassification

    def _get_additional_stats(self, model: models.ModelForMultiNodeClassification) \
            -> Dict[str, data_containers.DataContainer]:
        return {
            f"{node_type}_accuracy": data_containers.AverageScalarContainer(name=f"{node_type}_accuracy")
            for node_type, num_classes in model.cfg.num_classes.items()
        }

    def _prepare_inputs_and_labels(
            self,
            batch: Batch,
            device: torch.device
    ) -> Tuple[Dict[str, Any], Any]:
        # extract labels from info dict
        label_dict = collections.defaultdict(list)
        for labels in batch.info.pop("label"):
            for node_type, label in labels.items():
                label_dict[node_type].append(label)
        label_dict = {k: to(torch.cat(v, dim=0), device) for k, v in label_dict.items()}

        return {"g": batch.data, **batch.info}, label_dict

    def _calc_loss(self,
                   labels: Dict[str, torch.Tensor],
                   model_output: Dict[str, torch.Tensor],
                   additional_losses: Dict[str, torch.Tensor]) -> torch.Tensor:
        return sum(
            F.cross_entropy(pred, labels[node_type])
            for node_type, pred in model_output.items()
        ) + sum(additional_losses.values())

    def _update_stats(self,
                      model: models.ModelForMultiNodeClassification,
                      inputs: Dict[str, Any],
                      labels: Dict[str, torch.Tensor],
                      model_output: Dict[str, torch.Tensor],
                      stats: Dict[str, data_containers.DataContainer],
                      step: int,
                      total_steps: int) -> None:
        for node_type, pred in model_output.items():
            predictions = torch.argmax(pred, dim=1)
            stats[f"{node_type}_accuracy"].add((labels[node_type] == predictions).cpu())
            stats[f"{node_type}_f1_prec_rec"].add((labels[node_type].cpu(), predictions.cpu()))

    @torch.inference_mode()
    def inference(
            self,
            model: models.ModelForMultiNodeClassification,
            inputs: List[Union[str, Sample]],
            **kwargs: Any
    ) -> List[Dict[str, List]]:
        self._check_model(model)
        model = model.eval()

        batch = self._batch_sequences_for_inference(inputs)
        outputs, _ = model(batch.data, **batch.info)

        return_logits = kwargs.get("return_logits", False)

        batch_predictions_dict = {}
        for node_type in model.cfg.num_classes:
            num_nodes = batch.data.batch_num_nodes(node_type)
            if "groups" in batch.info and node_type in batch.info["groups"][0]:
                num_nodes = [max(group[node_type][-1]["groups"]) + 1 for group in batch.info["groups"]]

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
            for i in range(batch.data.batch_size)
        ]
