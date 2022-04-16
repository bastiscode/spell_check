from typing import List, Any, Union, Tuple, Optional

import torch

from gnn_lib import models
from gnn_lib.tasks import utils as task_utils
from gnn_lib.data.utils import Sample, flatten, InferenceInfo
from gnn_lib.modules import inference
from gnn_lib.modules import utils as mod_utils
from gnn_lib.tasks.token2seq import Token2Seq
from gnn_lib.utils import Batch


class SECWordsNMT(Token2Seq):
    @torch.inference_mode()
    def inference(
            self,
            model: models.ModelForToken2Seq,
            inputs: Union[Batch, List[Union[str, Sample]]],
            input_strings: Optional[List[str]] = None,
            **kwargs: Any
    ) -> List[List[str]]:
        self._check_model(model)
        model = model.eval()
        model_cfg: models.ModelForToken2SeqConfig = model.cfg

        if isinstance(inputs, Batch):
            assert input_strings is not None
            batch = inputs
            inputs = [model.input_tokenizer.normalize(ipt) for ipt in input_strings]
        else:
            batch = self._batch_sequences_for_inference(inputs)
            inputs = [model.input_tokenizer.normalize(str(ipt)) for ipt in inputs]

        encoder_group_lengths: List[torch.Tensor] = batch.info["encoder_group_lengths"]
        detections = kwargs.get("detections", [[1] * len(group_lengths) for group_lengths in encoder_group_lengths])

        detections_flattened = flatten(detections)
        assert all(det in {0, 1} for det in detections_flattened)
        if sum(detections_flattened) == 0:
            return [[ipt] for ipt in inputs]

        detection_mask = torch.tensor(detections_flattened, dtype=torch.bool)

        input_words = flatten([ipt.split() for ipt in inputs])
        assert len(detections_flattened) == len(input_words), (len(detections_flattened), len(input_words))

        encoder_lengths = [len(t) for t in batch.data]
        encoder_inputs, encoder_padding_mask = model.pad_inputs(batch.data)
        encoder_outputs = model.encode(encoder_inputs, encoder_padding_mask)
        encoder_outputs = [encoder_outputs[i, :l, :] for i, l in enumerate(encoder_lengths)]

        decoder_positions = torch.cat([
            torch.cat([torch.tensor([0]), torch.cumsum(group_lengths, dim=0)[:-1]])
            for group_lengths in encoder_group_lengths
        ])

        encoder_outputs = mod_utils.pad(
            [
                enc_output
                for enc_outputs, group_lengths in zip(encoder_outputs, encoder_group_lengths)
                for enc_output in torch.split(enc_outputs, mod_utils.tensor_to_python(group_lengths, force_list=True))
            ]
        )

        encoder_lengths = torch.cat(batch.info["encoder_group_lengths"])

        word_results = inference.run_inference(
            model=model.head,
            output_tokenizer=model.output_tokenizer,
            encoder_outputs={"encoder_outputs": encoder_outputs[detection_mask]},
            encoder_lengths={"encoder_outputs": encoder_lengths[detection_mask]},
            max_length=model_cfg.max_output_length,
            input_strings=[w for w, det in zip(input_words, detections_flattened) if det],
            decoder_positions=decoder_positions[detection_mask],
            **kwargs
        )

        words_per_input = [sum(detection) for detection in detections]
        word_results_per_input = mod_utils.split(word_results, words_per_input)

        all_results = []
        for input_str, word_results, word_detections, num_words in zip(
                inputs,
                word_results_per_input,
                detections,
                words_per_input
        ):
            assert len(word_results) == num_words

            input_words = input_str.split()

            batch_result_str = []
            if len(word_results) == 0:
                # greedy or sample inference give exactly one output per word
                min_num_word_results = 1
            else:
                # best first or beam search inference can give more than 1 output per word
                min_num_word_results = min(len(num_outputs_per_word) for num_outputs_per_word in word_results)
            for i in range(min_num_word_results):
                result_words = []
                result_idx = 0
                for input_word, detection in zip(input_words, word_detections):
                    if detection:
                        result_words.append(word_results[result_idx][i])
                        result_idx += 1
                    else:
                        result_words.append(input_word)
                assert result_idx == num_words
                batch_result_str.append(" ".join(result_words))
            all_results.append(batch_result_str)
        return all_results

    def _split_sample_for_inference(
            self,
            sample: Sample,
            max_length: int,
            context_length: int,
            **kwargs: Any
    ) -> List[Tuple[int, int, int, int]]:
        return task_utils.get_word_windows(sample, max_length, context_length)

    def _merge_inference_outputs(
            self,
            sequence: str,
            infos: List[InferenceInfo],
            predictions: List[List[str]],
            **kwargs: Any
    ) -> List[str]:
        return task_utils.merge_sec_words_nmt_outputs(
            sequence,
            infos,
            [[p.split() for p in prediction] for prediction in predictions]
        )
