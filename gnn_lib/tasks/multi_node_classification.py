import collections
from typing import Dict, List, Union, Any, Tuple

import dgl
import torch
from torch.nn import functional as F

from gnn_lib import tasks, models
from gnn_lib.models import MODEL_INPUTS
from gnn_lib.modules import utils
from gnn_lib.tasks import utils as task_utils
from gnn_lib.utils import data_containers
from gnn_lib.utils.distributed import DistributedDevice


class MultiNodeClassification(tasks.Task):
    expected_model = models.ModelForMultiNodeClassification

    def _get_additional_stats(self, model: models.ModelForMultiNodeClassification) \
            -> Dict[str, data_containers.DataContainer]:
        stats = {}
        # for multi node classification, by default record accuracy and 1 vs all precision, recall and f1 scores
        for node_type, num_classes in model.cfg.num_classes.items():
            stats[f"{node_type}_accuracy"] = data_containers.AverageScalarContainer(name=f"{node_type}_accuracy")
            stats[f"{node_type}_f1_prec_rec"] = data_containers.F1PrecRecContainer(
                name="fpr",
                class_names={i: f"{node_type}_{i}" for i in range(1, num_classes)}
            )
        return stats

    def _prepare_inputs_and_labels(self,
                                   batch: MODEL_INPUTS,
                                   device: DistributedDevice) -> Tuple[Dict[str, Any], Any]:
        g, info = batch
        g: dgl.DGLHeteroGraph = g.to(device.device)

        # extract labels from info dict
        label_dict = collections.defaultdict(list)
        for labels in info.pop("label"):
            for node_type, label in labels.items():
                label_dict[node_type].extend(label)
        label_dict = {k: torch.tensor(v, device=device.device, dtype=torch.long) for k, v in label_dict.items()}

        return {"g": g, **info}, label_dict

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
            inputs: Union[List[str], Tuple[dgl.DGLHeteroGraph, List[Dict[str, Any]]]],
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
                num_nodes = [group[node_type][-1]["groups"][-1] + 1 for group in infos["groups"]]

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
