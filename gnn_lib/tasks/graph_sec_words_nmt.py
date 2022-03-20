from typing import List, Any

import torch

from gnn_lib import models
from gnn_lib.modules import inference, utils as mod_utils
from gnn_lib.tasks.multi_node2seq import MultiNode2Seq


class GraphSECWordsNMT(MultiNode2Seq):
    @torch.inference_mode()
    def inference(
            self,
            model: models.ModelForMultiNode2Seq,
            inputs: List[str],
            **kwargs: Any
    ) -> List[List[str]]:
        self._check_model(model)
        model = model.eval()

        assert len(model.cfg.decoder_node_types) == 1
        decoder_node_type = model.cfg.decoder_node_types[0]

        assert isinstance(inputs, list) and isinstance(inputs[0], str)
        g, infos = self.variant.prepare_sequences_for_inference(inputs)

        model_cfg: models.ModelForMultiNode2SeqConfig = model.cfg
        g = model.encode(g)

        (
            encoder_outputs, encoder_lengths, aligned_encoder_positions
        ) = mod_utils.multi_node2seq_encoder_outputs_from_graph(
            g,
            decoder_node_type,
            model_cfg.context_node_types[decoder_node_type],
            model_cfg.hidden_feature,
            model_cfg.align_positions_with[decoder_node_type] if model_cfg.align_positions_with is not None else None
        )

        org_num_nodes_to_decode = [len(enc_output) for enc_output in encoder_outputs[decoder_node_type]]
        detections = kwargs.get("detections", [[1] * to_decode for to_decode in org_num_nodes_to_decode])

        assert len(detections) == len(org_num_nodes_to_decode)
        assert all(len(det) == to_decode for det, to_decode in zip(detections, org_num_nodes_to_decode))
        assert all(all(d in {0, 1} for d in det) for det in detections)

        num_nodes_to_decode = [sum(det) for det in detections]
        if sum(num_nodes_to_decode) == 0:
            return inputs

        # filter the input words based on detections
        input_strings = []
        for detection, input_str in zip(detections, inputs):
            input_words = input_str.split()
            assert len(input_words) == len(detection)
            for det, word in zip(detection, input_words):
                if det:
                    input_strings.append(word)

        detections_mask = torch.cat(
            [torch.tensor(det, dtype=torch.bool) for det in detections]
        ).to(g.device)

        # bring encoder outputs and lengths into correct format
        encoder_outputs = {
            enc_node_type: mod_utils.pad([
                output for enc_output in enc_outputs for output in enc_output
            ])[detections_mask]
            for enc_node_type, enc_outputs in encoder_outputs.items()
        }

        encoder_lengths = {
            enc_node_type: torch.cat(enc_lengths)[detections_mask]
            for enc_node_type, enc_lengths in encoder_lengths.items()
        }

        if aligned_encoder_positions is not None:
            decoder_positions = torch.cat(aligned_encoder_positions)[detections_mask]
        else:
            decoder_positions = torch.zeros(detections_mask.sum(), dtype=torch.long, device=g.device)

        output_tokenizer = model.tokenizers[f"{decoder_node_type}_output_tokenizer"]

        decoder: mod_utils.DecoderMixin = model.head[decoder_node_type]

        results = inference.run_inference(
            model=decoder,
            output_tokenizer=output_tokenizer,
            encoder_outputs=encoder_outputs,
            encoder_lengths=encoder_lengths,
            max_length=512,
            decoder_positions=decoder_positions,
            input_strings=input_strings,
            **kwargs
        )

        results = mod_utils.split(results, num_nodes_to_decode)

        all_results = []
        for input_str, batch_word_results, word_detections, num_nodes, org_num_nodes in zip(
                inputs,
                results,
                detections,
                num_nodes_to_decode,
                org_num_nodes_to_decode
        ):
            assert len(batch_word_results) == num_nodes

            input_words = input_str.split()
            assert len(input_words) == org_num_nodes

            batch_result_str = []
            if len(batch_word_results) == 0:
                # greedy or sample inference give exactly one output per word
                min_num_word_results = 1
            else:
                # best first or beam search inference can give more than 1 output per word
                min_num_word_results = min(len(word_results) for word_results in batch_word_results)
            for i in range(min_num_word_results):
                result_words = []
                result_idx = 0
                for input_word, detection in zip(input_words, word_detections):
                    if detection:
                        result_words.append(batch_word_results[result_idx][i])
                        result_idx += 1
                    else:
                        result_words.append(input_word)
                assert result_idx == num_nodes
                batch_result_str.append(" ".join(result_words))
            all_results.append(batch_result_str)
        return all_results
