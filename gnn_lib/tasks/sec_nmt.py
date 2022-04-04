from typing import List, Any, Union, Tuple

import torch
from gnn_lib import models
from gnn_lib.data.utils import Sample, InferenceInfo
from gnn_lib.tasks.seq2seq import Seq2Seq
from gnn_lib.tasks import utils as task_utils


class SECNMT(Seq2Seq):
    @torch.inference_mode()
    def inference(
            self,
            model: models.ModelForSeq2Seq,
            inputs: List[Union[str, Sample]],
            **kwargs: Any
    ) -> List[List[str]]:
        kwargs.update({
            "input_strings": [str(ipt) for ipt in inputs]
        })

        return super().inference(model, inputs, **kwargs)

    def _split_sample_for_inference(
            self,
            sample: Sample,
            max_length: int,
            context_length: int,
            **kwargs: Any
    ) -> List[Tuple[int, int, int, int]]:
        return task_utils.get_word_windows(sample, max_length, 0)

    def _merge_inference_outputs(
            self,
            sequence: str,
            infos: List[InferenceInfo],
            predictions: List[List[str]],
            **kwargs: Any
    ) -> List[str]:
        return task_utils.merge_sec_nmt_outputs(predictions)
