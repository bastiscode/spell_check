import copy
from typing import List, Any, Union

import torch
from gnn_lib import models
from gnn_lib.data.utils import Sample
from gnn_lib.modules import inference
from gnn_lib.tasks.token2seq import Token2Seq
from gnn_lib.data import utils
from gnn_lib.tasks import utils as task_utils
from gnn_lib.modules import utils as mod_utils
from gnn_lib.utils import Batch


class SECWordsNMT(Token2Seq):
    @torch.inference_mode()
    def inference(
            self,
            model: models.ModelForToken2Seq,
            inputs: List[Union[str, Sample]],
            **kwargs: Any
    ) -> List[List[str]]:
        self._check_model(model)
        model = model.eval()
        model_cfg: models.ModelForToken2SeqConfig = model.cfg

        org_inputs = copy.deepcopy(inputs)

        batch = self.variant.batch_sequences_for_inference(inputs)

        oversized = [len(t) > model_cfg.max_length for t in batch.data]
        if sum(oversized):
            # print(f"found {sum(oversized)} sequences that are too long: {[len(t) for t in batch.data]}")
            data = [t for i, t in enumerate(batch.data) if not oversized[i]]
            info = {k: [v_ for i, v_ in enumerate(v) if not oversized[i]] for k, v in batch.info.items()}
            batch = Batch(data, info)
            inputs = [ipt for i, ipt in enumerate(inputs) if not oversized[i]]

        encoder_lengths = [len(t) for t in batch.data]
        encoder_inputs, encoder_padding_mask = model.pad_inputs(batch.data)
        encoder_outputs = model.encode(encoder_inputs, encoder_padding_mask)
        encoder_outputs = [encoder_outputs[i, :l, :] for i, l in enumerate(encoder_lengths)]

        encoder_group_lengths: List[torch.Tensor] = batch.info["encoder_group_lengths"]
        if "detections" in kwargs:
            detections = [detection for i, detection in enumerate(kwargs["detections"]) if not oversized[i]]
        else:
            detections = [[1] * len(group_lengths) for group_lengths in encoder_group_lengths]

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
            max_length=model_cfg.max_output_length,
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

        prediction_idx = 0
        outputs = []
        for ipt, is_oversized in zip(org_inputs, oversized):
            if not is_oversized:
                outputs.append(all_results[prediction_idx])
                prediction_idx += 1
            else:
                outputs.append([ipt])
        assert prediction_idx == len(all_results)
        return outputs
