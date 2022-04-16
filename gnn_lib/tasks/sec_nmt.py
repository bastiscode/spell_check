import re
from typing import List, Any, Union, Tuple, Optional

import torch

from gnn_lib import models
from gnn_lib.data import tokenization
from gnn_lib.data.utils import Sample, InferenceInfo
from gnn_lib.modules import inference
from gnn_lib.tasks import utils as task_utils
from gnn_lib.tasks.seq2seq import Seq2Seq
from gnn_lib.utils import Batch


def _match_input_output(input_string: str, output_string: str) -> str:
    pattern = re.compile(f"^{re.escape(input_string)}(\s?\S+\s?).*")
    match = pattern.fullmatch(output_string)
    if match is None:
        return ""
    else:
        return output_string[match.start(1):match.end(1)]


def stop_on_eos_or_ws(tokenizer: tokenization.Tokenizer) -> inference.StopFn:
    bos_token_id = tokenizer.token_to_id(tokenization.BOS)
    eos_token_id = tokenizer.token_to_id(tokenization.EOS)
    de_tok_fn = inference.get_de_tok_fn(tokenizer, bos_token_id, eos_token_id)

    def _stop(token_ids: List[int], output_string: Optional[str] = None) -> bool:
        assert output_string is not None
        if token_ids[-1] == eos_token_id:
            return True
        else:
            new_output_str = de_tok_fn(token_ids)
            matched = _match_input_output(tokenizer.normalize(output_string), new_output_str)
            return matched.endswith(" ")

    return _stop


class SECNMT(Seq2Seq):
    @torch.inference_mode()
    def inference(
            self,
            model: models.ModelForSeq2Seq,
            inputs: Union[Batch, List[Union[str, Sample]]],
            input_strings: Optional[List[int]] = None,
            **kwargs: Any
    ) -> List[List[str]]:
        self._check_model(model)
        model = model.eval()
        model_cfg: models.ModelForSeq2SeqConfig = model.cfg

        if isinstance(inputs, Batch):
            assert input_strings is not None
            batch = inputs
            input_strings = [model.input_tokenizer.normalize(ipt) for ipt in input_strings]
        else:
            batch = self._batch_sequences_for_inference(inputs)
            input_strings = [model.input_tokenizer.normalize(str(ipt)) for ipt in inputs]

        if "detections" not in kwargs:
            return super().inference(model, batch, input_strings=input_strings, **kwargs)

        else:
            input_words = [ipt.split() for ipt in input_strings]

            encoder_inputs, encoder_padding_mask = model.pad_inputs(batch.data)
            encoder_outputs = model.encode(encoder_inputs, encoder_padding_mask)
            encoder_lengths = torch.tensor([len(t) for t in batch.data], dtype=torch.long)

            detections = kwargs.pop("detections")
            kwargs.update({
                "stop_fn": stop_on_eos_or_ws(model.output_tokenizer)
            })

            assert (
                    len(detections) == len(input_words)
                    and all(len(words) == len(det) for words, det in zip(input_words, detections))
            )

            batch_current_indices = [0] * len(detections)
            batch_indices_to_decode = []
            batch_finished = []
            output_words = []
            for i, detection in enumerate(detections):
                indices_to_decode = [i for i, det in enumerate(detection) if det]
                batch_indices_to_decode.append(indices_to_decode)
                if len(indices_to_decode):
                    batch_finished.append(False)
                    output_words.append(input_words[i][:indices_to_decode[0]])
                else:
                    batch_finished.append(True)
                    output_words.append(input_words[i])

            while True:
                inputs_to_decode = [i for i, finished in enumerate(batch_finished) if not finished]
                if not len(inputs_to_decode):
                    break

                kwargs.update({
                    "input_strings": [input_strings[i] for i in inputs_to_decode],
                    "output_strings": [" ".join(output_words[i]) for i in inputs_to_decode]
                })

                inputs_to_decode = torch.tensor(inputs_to_decode, dtype=torch.long)
                intermediate_results = inference.run_inference(
                    model=model.head,
                    output_tokenizer=model.output_tokenizer,
                    encoder_outputs={"encoder_outputs": inference.sub_select(encoder_outputs, inputs_to_decode)},
                    encoder_lengths={"encoder_outputs": inference.sub_select(encoder_lengths, inputs_to_decode)},
                    max_length=model_cfg.max_output_length,
                    **kwargs
                )

                for i, results, output_string in zip(
                        inputs_to_decode.tolist(), intermediate_results, kwargs["output_strings"]
                ):
                    result_string = results[0]

                    matched = _match_input_output(
                        model.output_tokenizer.normalize(output_string), result_string
                    )
                    assert len(matched.split()) <= 1
                    matched_word = matched.strip()

                    if matched.startswith(" "):
                        output_words[i].append(matched_word)
                    else:
                        if not len(output_words[i]):
                            output_words[i].append("")
                        output_words[i][-1] += matched_word

                    batch_current_indices[i] += 1
                    if batch_current_indices[i] < len(batch_indices_to_decode[i]):
                        from_idx, to_idx = (
                            batch_indices_to_decode[i][batch_current_indices[i] - 1],
                            batch_indices_to_decode[i][batch_current_indices[i]]
                        )
                        output_words[i].extend(
                            input_words[i][from_idx + 1: to_idx]
                        )
                    else:
                        batch_finished[i] = True
                        from_idx = batch_indices_to_decode[i][-1]
                        output_words[i].extend(
                            input_words[i][from_idx + 1:]
                        )

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
