from typing import List, Any, Union, Optional, Tuple

import torch

from nsc import models
from nsc.data.utils import Sample, InferenceInfo
from nsc.tasks.graph2seq import Graph2Seq
from nsc.utils import Batch
from nsc.tasks import utils as task_utils


class GraphSECNMT(Graph2Seq):
    @torch.inference_mode()
    def inference(
            self,
            model: models.ModelForGraph2Seq,
            inputs: Union[Batch, List[Union[str, Sample]]],
            input_strings: Optional[List[str]] = None,
            **kwargs: Any
    ) -> List[List[str]]:
        if isinstance(inputs, Batch):
            assert input_strings is not None
            batch = inputs
            input_strings = [model.input_tokenizers["token"].normalize(ipt) for ipt in input_strings]
        else:
            batch = self._batch_sequences_for_inference(inputs)
            input_strings = [model.input_tokenizers["token"].normalize(str(ipt)) for ipt in inputs]

        return super().inference(model, batch, input_strings=input_strings, **kwargs)

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
