from typing import List, Union, Any, Optional, Callable

import dgl
import torch

from gnn_lib import models
from gnn_lib.data import index, utils
from gnn_lib.modules import inference, utils as mod_utils
from gnn_lib.tasks.multi_node2seq import MultiNode2Seq


def beam_score(
        normalize_by_length: bool = True,
        alpha: float = 1.0,
        restrict_to_input_or_prefix: bool = True
) -> inference.SCORE_FN:
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

        if restrict_to_input_or_prefix and len(pred_str_split) > 0:
            input_words, input_ws = utils.tokenize_words_regex(input_str)
            pred_words, pred_ws = utils.tokenize_words_regex(pred_str_split[-1])

            assert prefix_index is not None
            for pred_w in pred_words:
                pred_is_valid_prefix = (
                        len(prefix_index.retrieve(pred_w)) > 0 or
                        len(prefix_index.retrieve(pred_w.lower())) > 0
                )
                pred_is_prefix_of_input = (
                        any(ipt_w.startswith(pred_w) for ipt_w in input_words) or
                        any(ipt_w.lower().startswith(pred_w) for ipt_w in input_words)
                )
                if pred_is_valid_prefix or pred_is_prefix_of_input:
                    continue

                return -1_000_000.

        s = sum(beam.log_prob)
        if normalize_by_length:
            s = s / (len(beam.log_prob) ** alpha)

        # end = time.perf_counter()
        # print(f"scoring beam took {(end - start) * 1e6:.2f}us: {input_words} -> {pred_words} (score={s:.4f})")
        return s

    return score


class SECWordsNMT(MultiNode2Seq):
    expected_model = models.ModelForMultiNode2Seq  # type: ignore

    @torch.inference_mode()
    def inference(
            self,
            model: models.ModelForMultiNode2Seq,
            inputs: Union[List[str], dgl.DGLHeteroGraph],
            **kwargs: Any
    ) -> List[List[str]]:
        self._check_model(model)
        model = model.eval()

        assert len(model.cfg.decoder_node_types) == 1
        decoder_node_type = model.cfg.decoder_node_types[0]

        assert isinstance(inputs, list) and isinstance(inputs[0], str)
        g = self.variant.prepare_sequences_for_inference(inputs)

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
        if model_cfg.align_positions_with is not None:
            assert aligned_encoder_positions is not None

        org_num_nodes_to_decode = [len(enc_output) for enc_output in encoder_outputs[decoder_node_type]]
        detections = kwargs.get("detections", [[1] * to_decode for to_decode in org_num_nodes_to_decode])

        assert len(detections) == len(org_num_nodes_to_decode)
        assert all(len(det) == to_decode for det, to_decode in zip(detections, org_num_nodes_to_decode))
        assert all(all(d in {0, 1} for d in det) for det in detections)

        num_nodes_to_decode = [sum(det) for det in detections]

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
            for i in range(1 if len(batch_word_results) == 0 else len(batch_word_results[0])):
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
