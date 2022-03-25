from typing import List, Any

import torch
from gnn_lib import models
from gnn_lib.modules import inference
from gnn_lib.tasks.token2seq import Token2Seq
from gnn_lib.data import utils
from gnn_lib.tasks import utils as task_utils
from gnn_lib.modules import utils as mod_utils


class SECWordsNMT(Token2Seq):
    @torch.inference_mode()
    def inference(self,
                  model: models.ModelForToken2Seq,
                  inputs: List[str],
                  **kwargs: Any) -> List[List[str]]:
        self._check_model(model)
        model = model.eval()

        assert task_utils.is_string_input(inputs), "sec words nmt input must be a list of strings"
        batch = self.variant.prepare_sequences_for_inference(inputs)

        encoder_lengths = [len(t) for t in batch.data]
        encoder_inputs, encoder_padding_mask = model.pad_inputs(batch.data)
        encoder_outputs = model.encode(encoder_inputs, encoder_padding_mask)
        encoder_outputs = [encoder_outputs[i, :l, :] for i, l in enumerate(encoder_lengths)]

        encoder_group_lengths: List[torch.Tensor] = batch.info["encoder_group_lengths"]
        detections = kwargs.get("detections", [[1] * len(group_lengths) for group_lengths in encoder_group_lengths])

        decoder_positions = torch.cat([
            torch.cat([torch.tensor([0]), torch.cumsum(group_lengths, dim=0)[:-1]])
            for group_lengths in encoder_group_lengths
        ])

        input_words = utils.flatten([ipt.split() for ipt in inputs])
        detections_flattened = utils.flatten(detections)
        assert len(detections_flattened) == len(input_words)
        assert all(det in {0, 1} for det in detections_flattened)

        detection_mask = torch.tensor(detections_flattened, dtype=torch.bool)

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
            max_length=model.cfg.max_length,
            input_strings=[w for w, det in zip(input_words, detections_flattened) if det],
            decoder_positions=decoder_positions[detection_mask],
            **kwargs
        )

        words_per_input = [sum(detection) for detection in detections]
        word_results_per_input = mod_utils.split(word_results, words_per_input)

        all_results = []
        for input_str, word_results, word_detections, num_words, group_lengths in zip(
                inputs,
                word_results_per_input,
                detections,
                words_per_input,
                encoder_group_lengths
        ):
            assert len(word_results) == num_words

            input_words = input_str.split()
            assert len(input_words) == len(group_lengths)

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
