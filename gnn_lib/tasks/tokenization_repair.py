from typing import Union, List, Dict, Any

import torch
from torch.nn import functional as F

from gnn_lib import models
from gnn_lib.tasks import utils as task_utils
from gnn_lib.tasks.token_classification import TokenClassification
from gnn_lib.utils import tokenization_repair, data_containers


class TokenizationRepair(TokenClassification):
    def _get_additional_stats(self, model: models.ModelForTokenClassification) \
            -> Dict[str, data_containers.DataContainer]:
        stats = super()._get_additional_stats(model)
        stats["fpr"] = data_containers.F1PrecRecContainer(name="f1_prec_rec", class_names={1: "insert", 2: "delete"})
        return stats

    def _calc_loss(self,
                   labels: torch.Tensor,
                   model_output: List[torch.Tensor],
                   additional_losses: Dict[str, torch.Tensor]) -> torch.Tensor:
        return F.cross_entropy(
            input=torch.cat(model_output, dim=0),
            target=labels,
            # weight insertions and deletions more
            weight=torch.tensor([1, 5, 5], device=labels.device, dtype=torch.float)
        ) + sum(additional_losses.values())

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

    @torch.inference_mode()
    def inference(
            self,
            model: models.ModelForTokenClassification,
            inputs: List[str],
            **kwargs: Any
    ) -> Union[List[str], List[List[int]]]:
        assert task_utils.is_string_input(inputs)
        batch = self.variant.batch_sequences_for_inference(inputs)

        repair_tokens_list = super().inference(model, batch, **kwargs)

        return [
            tokenization_repair.repair_whitespace(
                ipt,
                repair_tokens
            )
            for ipt, repair_tokens in zip(inputs, repair_tokens_list)
        ]
