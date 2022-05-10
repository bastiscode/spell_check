from typing import Union, List, Dict, Any, Tuple, Optional

import torch
from torch.nn import functional as F

from nsc import models
from nsc.data.utils import Sample, InferenceInfo
from nsc.models import Model
from nsc.tasks import utils as task_utils
from nsc.tasks.token_classification import TokenClassification
from nsc.utils import tokenization_repair, data_containers, Batch


class TokenizationRepair(TokenClassification):
    def _get_additional_stats(
            self, model: models.ModelForTokenClassification
    ) -> Dict[str, data_containers.DataContainer]:
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

    def get_max_input_length(self, model: Model) -> int:
        return model.cfg.max_length - 2 * self.variant.cfg.add_bos_eos

    @torch.inference_mode()
    def inference(
            self,
            model: models.ModelForTokenClassification,
            inputs: Union[Batch, List[Union[str, Sample]]],
            input_strings: Optional[List[str]] = None,
            **kwargs: Any
    ) -> List[str]:
        if isinstance(inputs, Batch):
            assert input_strings is not None
        else:
            input_strings = [str(ipt) for ipt in inputs]
        repair_tokens_list = super().inference(model, inputs, **kwargs)
        return [
            tokenization_repair.repair_whitespace(
                ipt,
                repair_tokens[1:-1] if self.variant.cfg.add_bos_eos else repair_tokens
            )
            for ipt, repair_tokens in zip(input_strings, repair_tokens_list)
        ]

    def _split_sample_for_inference(
            self,
            sample: Sample,
            max_length: int,
            context_length: int,
            **kwargs: Any
    ) -> List[Tuple[int, int, int, int]]:
        if self.variant.cfg.input_type == "char":
            return task_utils.get_character_windows(
                sample,
                max_length,
                context_length
            )
        elif self.variant.cfg.input_type == "byte":
            return task_utils.get_byte_windows(
                sample,
                max_length,
                context_length
            )
        else:
            raise RuntimeError("should not happen")

    def _merge_inference_outputs(
            self,
            sequence: str,
            infos: List[InferenceInfo],
            predictions: List[str],
            **kwargs: Any
    ) -> str:
        return task_utils.merge_tokenization_repair_outputs(sequence, infos, predictions)
