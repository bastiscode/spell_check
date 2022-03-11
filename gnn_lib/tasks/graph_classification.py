from typing import Dict, Union, List, Any, Tuple

import dgl
import torch
from torch.nn import functional as F

from gnn_lib import tasks, models
from gnn_lib.modules import utils
from gnn_lib.tasks import utils as task_utils
from gnn_lib.utils import data_containers
from gnn_lib.utils.distributed import DistributedDevice


class GraphClassification(tasks.Task):
    expected_model = models.ModelForGraphClassification

    def _get_additional_stats(self, model: models.ModelForGraphClassification) -> \
            Dict[str, data_containers.DataContainer]:
        stats = {
            "accuracy": data_containers.AverageScalarContainer(name="accuracy"),
            "f1_prec_rec": data_containers.F1PrecRecContainer(
                name="fpr",
                class_names={i: str(i) for i in range(1, model.cfg.num_classes)}
            )
        }
        return stats

    def _prepare_inputs_and_labels(self,
                                   batch: Tuple[dgl.DGLHeteroGraph, List[Dict[str, Any]]],
                                   device: DistributedDevice) -> Tuple[Dict[str, Any], Any]:
        g, info = batch
        labels = torch.tensor([i["label"] for i in info], device=device.device, dtype=torch.long)
        return {"g": g.to(device.device)}, labels

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
        stats["f1_prec_rec"].add((labels.cpu(), predictions.cpu()))

    @torch.inference_mode()
    def inference(
            self,
            model: models.ModelForGraphClassification,
            inputs: Union[List[str], Tuple[dgl.DGLHeteroGraph, List[Dict[str, Any]]]],
            **kwargs: Any
    ) -> List:
        self._check_model(model)
        model = model.eval()

        threshold = kwargs.get("threshold", 0.5)
        temperature = kwargs.get("temperature", 1.0)

        got_str_input = isinstance(inputs, list) and isinstance(inputs[0], str)
        if got_str_input:
            g, infos = self.variant.prepare_sequences_for_inference(inputs)
        else:
            g, infos = inputs

        outputs, _ = model(g, **infos)

        return_logits = kwargs.get("return_logits", False)
        if return_logits:
            return utils.tensor_to_python(outputs, force_list=True)
        else:
            return utils.tensor_to_python(
                task_utils.class_predictions(outputs, threshold, temperature), force_list=True
            )
