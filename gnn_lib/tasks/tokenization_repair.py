from typing import Union, List, Dict, Any

import torch
from torch.nn import functional as F

from gnn_lib import models
from gnn_lib.data import variants, tokenization
from gnn_lib.tasks import utils as task_utils
from gnn_lib.tasks.multi_node_classification import MultiNodeClassification
from gnn_lib.utils import tokenization_repair, data_containers, BATCH


class TokenizationRepair(MultiNodeClassification):
    def __init__(self, variant_cfg: variants.DatasetVariantConfig, checkpoint_dir: str, seed: int):
        super().__init__(variant_cfg, checkpoint_dir, seed)
        self.cls_weight = torch.tensor([1, 5, 5], dtype=torch.float)

    def _get_additional_stats(self, model: models.ModelForMultiNodeClassification) \
            -> Dict[str, data_containers.DataContainer]:
        stats = {
            "accuracy": data_containers.AverageScalarContainer(name="accuracy"),
            "f1_prec_rec": data_containers.F1PrecRecContainer(
                name="fpr",
                class_names={1: "insert",
                             2: "delete"}
            ),
            "text": data_containers.MultiTextContainer(
                name="text_samples",
                max_samples=4
            )
        }
        return stats

    def _calc_loss(self,
                   labels: Dict[str, torch.Tensor],
                   model_output: Dict[str, torch.Tensor],
                   additional_losses: Dict[str, torch.Tensor]) -> torch.Tensor:
        return sum(
            F.cross_entropy(
                input=pred,
                target=labels[node_type],
                weight=self.cls_weight.to(pred.device)
            )
            for node_type, pred in model_output.items()
        ) + sum(additional_losses.values())

    def _update_stats(self,
                      model: models.ModelForMultiNodeClassification,
                      inputs: Dict[str, Any],
                      labels: Dict[str, torch.Tensor],
                      model_output: Dict[str, torch.Tensor],
                      stats: Dict[str, data_containers.DataContainer],
                      step: int,
                      log_every: int) -> None:
        predictions = torch.argmax(model_output["token"], dim=1)
        labels = labels["token"]

        stats["accuracy"].add((labels == predictions).cpu())
        stats["f1_prec_rec"].add((labels.cpu(), predictions.cpu()))

        text_container = stats["text"]
        if step % max(log_every // text_container.max_samples, 1) != 0 or \
                len(text_container.samples) >= text_container.max_samples:
            return

        token_ids = task_utils.get_token_ids_from_graphs(inputs["g"])
        tokenizer: tokenization.Tokenizer = model.tokenizers["token"]
        input_text = tokenizer.de_tokenize(token_ids[0])

        batch_num_tokens = inputs["g"].batch_num_nodes("token")
        labels = labels[:batch_num_tokens[0]].tolist()
        labels_str = "".join(str(l) for l in labels)
        pred = predictions[:batch_num_tokens[0]].tolist()
        pred_str = "".join(str(p) for p in pred)

        text_container.add(
            f"""
    input:\t{input_text}
    label:\t{labels_str}
    pred:\t{pred_str}
---
            """
        )

    @torch.inference_mode()
    def inference(self,
                  model: models.ModelForMultiNodeClassification,
                  inputs: Union[List[str], BATCH],
                  **kwargs: Any) -> \
            Union[List[str], List[List[int]]]:
        got_str_input = isinstance(inputs, list) and isinstance(inputs[0], str)
        if got_str_input:
            g, infos = self.variant.prepare_sequences_for_inference(inputs)
        else:
            g, infos = inputs

        repair_tokens_dicts = super().inference(model, g, **kwargs)

        if got_str_input:
            return [
                tokenization_repair.repair_whitespace(
                    ipt,
                    repair_tokens_dicts[i]["token"]
                )
                for i, ipt in enumerate(inputs)
            ]
        else:
            return [repair_tokens_dict["token"] for repair_tokens_dict in repair_tokens_dicts]
