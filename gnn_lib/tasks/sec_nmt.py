from typing import Union, List, Any, Optional, Callable

import dgl
import torch

from gnn_lib import models
from gnn_lib.data import utils, index
from gnn_lib.modules import inference
from gnn_lib.tasks.graph2seq import Graph2Seq


def beam_score(normalize_by_length: bool = True, alpha: float = 1.0) -> inference.SCORE_FN:
    def score(beam: inference.Beam,
              bos_token_id: int,
              eos_token_id: int,
              input_str: Optional[str] = None,
              de_tok_fn: Optional[Callable[[List[int]], str]] = None,
              prefix_index: Optional[index.PrefixIndex] = None) -> float:
        # start = time.perf_counter()
        assert beam.token_ids[0] == bos_token_id
        token_ids = beam.token_ids[1:]  # strip bos token
        if beam.is_eos(eos_token_id):
            token_ids = token_ids[:-1]

        pred_str = de_tok_fn(token_ids)
        pred_str_split = pred_str.split()

        if len(pred_str_split) > 0:
            input_words, input_ws = utils.tokenize_words_regex(input_str)
            pred_words, pred_ws = utils.tokenize_words_regex(pred_str_split[-1])

            assert prefix_index is not None
            for pred_w in pred_words:
                if (
                        len(prefix_index.retrieve(pred_w)) > 0 or
                        len(prefix_index.retrieve(pred_w.lower())) > 0 or
                        any(ipt_w.startswith(pred_w) for ipt_w in input_words) or
                        any(ipt_w.lower().startswith(pred_w) for ipt_w in input_words)
                ):
                    continue
                return -10_000.

        s = sum(beam.log_prob)
        if normalize_by_length:
            s = s / (len(beam.log_prob) ** alpha)

        # end = time.perf_counter()
        # print(f"scoring beam took {(end - start) * 1e6:.2f}us: {input_words} -> {pred_words} (score={s:.4f})")
        return s

    return score


class SECNMT(Graph2Seq):
    @torch.inference_mode()
    def inference(self,
                  model: models.ModelForGraph2Seq,
                  inputs: Union[List[str], dgl.DGLHeteroGraph],
                  **kwargs: Any) -> List[List[str]]:
        assert isinstance(inputs, list) and isinstance(inputs[0], str)
        kwargs.update({
            "input_strings": inputs
        })

        return super().inference(model, inputs, **kwargs)
