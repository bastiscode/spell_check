from typing import List, Any, Union, Tuple, Optional

import torch
from gnn_lib import models
from gnn_lib.data import tokenization
from gnn_lib.data.utils import Sample, InferenceInfo
from gnn_lib.modules import inference
from gnn_lib.tasks.seq2seq import Seq2Seq
from gnn_lib.tasks import utils as task_utils


def stop_on_ws(tokenizer: tokenization.Tokenizer) -> inference.StopFn:
    bos_token_id = tokenizer.token_to_id(tokenization.BOS)
    eos_token_id = tokenizer.token_to_id(tokenization.EOS)
    de_tok_fn = inference.get_de_tok_fn(tokenizer, bos_token_id, eos_token_id)
    assert isinstance(tokenizer, tokenization.BPETokenizer), \
        "this decoding early stop function has to be used with a BPETokenizer"

    def _stop(token_ids: List[int], output_string: Optional[str] = None) -> bool:
        assert output_string is not None
        if token_ids[-1] == eos_token_id:
            return True
        new_output_str = de_tok_fn(token_ids)
        return len(new_output_str.split()) > len(output_string.split()) + 1

    return _stop


class SECNMT(Seq2Seq):
    @torch.inference_mode()
    def inference(
            self,
            model: models.ModelForSeq2Seq,
            inputs: List[Union[str, Sample]],
            **kwargs: Any
    ) -> List[List[str]]:
        if "detections" not in kwargs:
            kwargs.update({
                "input_strings": [str(ipt) for ipt in inputs]
            })
            return super().inference(model, inputs, **kwargs)
        else:
            self._check_model(model)
            model = model.eval()
            model_cfg: models.ModelForSeq2SeqConfig = model.cfg

            batch = self._batch_sequences_for_inference(inputs)
            inputs = [str(ipt) for ipt in inputs]

            encoder_inputs, encoder_padding_mask = model.pad_inputs(batch.data)
            encoder_outputs = model.encode(encoder_inputs, encoder_padding_mask)
            encoder_lengths = torch.tensor([len(t) for t in batch.data], dtype=torch.long)

            detections = kwargs.pop("detections")

            kwargs.update({
                "stop_fn": stop_on_ws(model.output_tokenizer)
            })

            output_words = [ipt.split() for ipt in inputs]
            assert len(detections) == len(output_words) and \
                   all(len(words) == len(det) for words, det in zip(output_words, detections))

            batch_current_indices = [0] * len(detections)
            batch_indices_to_decode = []
            for detection in detections:
                indices_to_decode = [i for i, det in enumerate(detection) if det]
                batch_indices_to_decode.append(indices_to_decode)

            while True:
                inputs_to_decode = [i for i, idx in enumerate(batch_current_indices)
                                    if idx < len(batch_indices_to_decode[i])]
                if not len(inputs_to_decode):
                    break

                kwargs.update({
                    "input_strings": [inputs[i] for i in inputs_to_decode],
                    "output_strings": [
                        " ".join(output_words[i][:batch_indices_to_decode[i][batch_current_indices[i]]])
                        for i in inputs_to_decode
                    ]
                })

                print(kwargs, inputs_to_decode, batch_current_indices)

                inputs_to_decode = torch.tensor(inputs_to_decode, dtype=torch.long)
                intermediate_results = inference.run_inference(
                    model=model.head,
                    output_tokenizer=model.output_tokenizer,
                    encoder_outputs={"encoder_outputs": inference.sub_select(encoder_outputs, inputs_to_decode)},
                    encoder_lengths={"encoder_outputs": inference.sub_select(encoder_lengths, inputs_to_decode)},
                    max_length=model_cfg.max_output_length,
                    **kwargs
                )

                for i, results in zip(inputs_to_decode, intermediate_results):
                    result_words = results[0].split()
                    word_idx = batch_indices_to_decode[i][batch_current_indices[i]]
                    output_words[i][word_idx] = result_words[word_idx]
                    batch_current_indices[i] += 1

                print(intermediate_results)
            return [[" ".join(words)] for words in output_words]

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
