from typing import Union, List, Any, Dict

import dgl
import torch
from torch.nn import functional as F

from gnn_lib import models
from gnn_lib.data import variants
from gnn_lib.modules import inference
from gnn_lib.tasks.graph2seq import Graph2Seq
from gnn_lib.utils import tokenization_repair


def _tok_rep_nmt_score_fn(bos_token_id: int, eos_token_id: int) -> inference.SCORE_FN:
    map_score = inference.map_score()

    def score(beam: inference.Beam) -> float:
        assert beam.token_ids[0] == bos_token_id
        if beam.is_eos(eos_token_id) and len(beam.token_ids):
            return map_score(beam)
        return float("-inf")

    return score


class TokenizationRepairNMT(Graph2Seq):
    expected_model = models.ModelForGraph2Seq  # type: ignore

    def __init__(self, variant_cfg: variants.DatasetVariantConfig, checkpoint_dir: str, seed: int):
        super().__init__(variant_cfg, checkpoint_dir, seed)
        self.cls_weight = torch.tensor([1, 5, 5, 1, 1, 1, 1], dtype=torch.float)

    def _calc_loss(self,
                   labels: Dict[str, Any],
                   model_output: torch.Tensor,
                   additional_losses: Dict[str, torch.Tensor]) -> torch.Tensor:
        return F.cross_entropy(
            input=model_output.reshape(-1, model_output.shape[-1]),
            target=labels["labels"].reshape(-1),
            weight=self.cls_weight.to(model_output.device),
            ignore_index=labels["pad_token_id"]
        ) + sum(additional_losses.values())

    @torch.inference_mode()
    def inference(self,
                  model: models.ModelForGraph2Seq,
                  inputs: Union[List[str], dgl.DGLHeteroGraph],
                  **kwargs: Any) -> List[List[str]]:
        kwargs.update({"inference_mode": "beam"})
        tok_repair_sequences = super().inference(model, inputs, **kwargs)

        output_tokenizer = model.tokenizers["output_tokenizer"]

        all_outputs = []
        for ipt, repair_sequences in zip(inputs, tok_repair_sequences):
            outputs = []
            for repair_sequence in repair_sequences:
                repair_tokens = output_tokenizer.tokenize(repair_sequence)
                repaired = tokenization_repair.repair_whitespace(ipt, repair_tokens)
                outputs.append(repaired)
            all_outputs.append(outputs)
        return all_outputs
