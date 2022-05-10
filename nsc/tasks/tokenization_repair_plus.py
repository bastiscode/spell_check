from typing import List, Any, Dict, Tuple, Union, Optional

import torch
from torch.nn import functional as F

from nsc import models, tasks
from nsc.data import utils
from nsc.data.utils import Sample
from nsc.modules import utils as mod_utils, inference
from nsc.tasks import utils as task_utils
from nsc.utils import data_containers, Batch, to, tokenization_repair


class TokenizationRepairPlus(tasks.Task):
    expected_models = models.ModelForTokenizationRepairPlus  # type: ignore

    def _get_additional_stats(self, model: models.ModelForTokenizationRepairPlus) \
            -> Dict[str, data_containers.DataContainer]:
        return {
            "seq_length": data_containers.HistogramContainer(
                name="input_sequence_length"
            )
        }

    def _prepare_inputs_and_labels(
            self,
            batch: Batch,
            device: torch.device
    ) -> Tuple[Dict[str, Any], Any]:
        data_dict = {}
        label_dict = {}

        sec_invalid_indices = set()
        if "sec_label" in batch.info:
            decoder_inputs = []
            decoder_labels = []
            decoder_group_lengths = []
            for i, labels in enumerate(batch.info.pop("sec_label")):
                word_decoder_inputs = []
                word_decoder_labels = []
                word_decoder_lengths = []
                for word_labels in labels:
                    word_decoder_inputs.extend(word_labels[:-1])
                    word_decoder_labels.extend(word_labels[1:])
                    word_decoder_lengths.append(len(word_labels) - 1)
                if sum(word_decoder_lengths) > 1024:
                    # this is kind of a temporary hack to skip too long decoding sequences that cause OOM
                    self.logger.warning(f"skipping sample with decoder length {sum(word_decoder_lengths)}")
                    sec_invalid_indices.add(i)
                    continue
                decoder_inputs.append(torch.tensor(word_decoder_inputs, dtype=torch.long))
                decoder_labels.append(torch.tensor(word_decoder_labels, dtype=torch.long))
                decoder_group_lengths.append(torch.tensor(word_decoder_lengths, dtype=torch.long))

            label_dict["sec_labels"] = to(
                mod_utils.pad(decoder_labels, val=batch.info["sec_pad_token_id"][0]).long(), device
            )
            label_dict["sec_pad_token_id"] = batch.info["sec_pad_token_id"][0]
            data_dict["sec_decoder_inputs"] = decoder_inputs
            data_dict["sec_decoder_group_lengths"] = decoder_group_lengths

        data_dict["x"] = task_utils.exclude_indices(batch.data, sec_invalid_indices)

        if "tokenization_repair_label" in batch.info:
            label_dict["tokenization_repair_labels"] = to(
                torch.cat(task_utils.exclude_indices(batch.info.pop("tokenization_repair_label"), sec_invalid_indices)),
                device
            )

        label_dict["sed_labels"] = to(
            torch.cat(task_utils.exclude_indices(batch.info.pop("sed_label"), sec_invalid_indices)),
            device
        )

        batch_info = {
            "word_groups": task_utils.exclude_indices(batch.info["word_groups"], sec_invalid_indices),
            "word_ws_groups": task_utils.exclude_indices(batch.info["word_ws_groups"], sec_invalid_indices),
            "input_group_lengths": task_utils.exclude_indices(batch.info["input_group_lengths"], sec_invalid_indices),
            "word_group_lengths": task_utils.exclude_indices(batch.info["word_group_lengths"], sec_invalid_indices)
        }
        if "word_features" in batch.info:
            batch_info["word_features"] = task_utils.exclude_indices(batch.info["word_features"], sec_invalid_indices)
        if "char_groups" in batch.info:
            batch_info["char_groups"] = task_utils.exclude_indices(batch.info["char_groups"], sec_invalid_indices)

        return {**data_dict, **batch_info}, label_dict

    def _calc_loss(
            self,
            labels: Dict[str, Any],
            model_output: Dict[str, Any],
            additional_losses: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        loss = F.cross_entropy(
            input=torch.cat(model_output["sed"], dim=0),
            target=labels["sed_labels"]
        )
        if "tokenization_repair_labels" in labels:
            tokenization_repair_loss = F.cross_entropy(
                input=torch.cat(model_output["tokenization_repair"], dim=0),
                target=labels["tokenization_repair_labels"],
                weight=torch.tensor([1, 5, 5], dtype=torch.float, device=labels["tokenization_repair_labels"].device)
            )
            loss = loss + tokenization_repair_loss
        if "sec_labels" in labels:
            sec_loss = F.cross_entropy(
                input=model_output["sec"].view(-1, model_output["sec"].shape[-1]),
                target=labels["sec_labels"].view(-1),
                ignore_index=labels["sec_pad_token_id"]
            )
            loss = loss + sec_loss
        return loss + sum(additional_losses.values())

    def _update_stats(
            self,
            model: models.ModelForGraph2Seq,
            inputs: Dict[str, Any],
            labels: Any,
            model_output: Any,
            stats: Dict[str, data_containers.DataContainer],
            step: int,
            total_steps: int
    ) -> None:
        sequence_length_container = stats["seq_length"]
        sequence_length_container.add([len(t) for t in inputs["x"]])

    @torch.inference_mode()
    def inference(
            self,
            model: models.ModelForTokenizationRepairPlus,
            inputs: Union[Batch, List[Union[str, Sample]]],
            input_strings: Optional[List[str]] = None,
            output_type: str = "all",
            no_repair: bool = False,
            no_detect: bool = False,
            **kwargs: Any
    ) -> List[Dict[str, Any]]:
        self._check_model(model)
        model = model.eval()
        model_cfg: models.ModelForTokenizationRepairPlusConfig = model.cfg
        assert output_type in {"tokenization_repair", "sed", "sec", "all"}, \
            f"unknown tokenization repair plus output type {output_type}, must be one of " \
            f"{{tokenization_repair, sed, sec, all}}"
        # the no_repair flag ensures that the input tokenization is not repaired, useful when you are sure that
        # the input has no tokenization errors or when you want to evaluate on sed benchmarks

        if isinstance(inputs, Batch):
            assert input_strings is not None
            batch = inputs
        else:
            batch = self._batch_sequences_for_inference(inputs)
            input_strings = [str(ipt) for ipt in inputs]

        x, padding_mask, input_lengths = model.pad_inputs(batch.data, pad_val=model.input_pad_token_id)

        # embed tokens and encode input representations
        x = model.encode(x.long(), padding_mask=padding_mask)
        x = [x[i, :l] for i, l in enumerate(input_lengths)]

        if model_cfg.input_type == "byte":
            assert batch.info["char_groups"] is not None
            # if we have bytes as inputs group bytes into characters, since tokenization repair output is on
            # character level
            x = mod_utils.group_features(
                x, batch.info["char_groups"], aggregation="mean"
            )

        outputs: List[Dict[str, Any]] = []

        # tokenization repair output
        repaired_strings = []
        tok_rep_logits = model.head["tokenization_repair"](x)
        for repair_logits, input_string in zip(tok_rep_logits, input_strings):
            if no_repair:
                repaired_string = input_string
            else:
                repair_tokens = task_utils.class_predictions(repair_logits).tolist()
                repaired_string = tokenization_repair.repair_whitespace(input_string, repair_tokens)

            repaired_strings.append(repaired_string)
            outputs.append({
                "tokenization_repair": repaired_string
            })

        if output_type == "tokenization_repair":
            return outputs

        word_groups = []
        word_ws_groups = []
        word_features = []
        word_ws_group_lengths = []

        for i, (input_string, repaired_string) in enumerate(zip(input_strings, repaired_strings)):
            repaired_words, repaired_doc = utils.tokenize_words(repaired_string, return_doc=True)

            word_groups.append({
                "stage": "char_to_word",
                "groups": utils.get_character_groups_from_repaired_doc(list(input_string), repaired_doc)
            })

            word_ws_lengths = [0] * len(repaired_string.split())
            word_ws = []
            word_ws_idx = 0
            for word in repaired_doc:
                word_ws.append(word_ws_idx)
                word_ws_lengths[word_ws_idx] += 1
                if word.whitespace_ == " ":
                    word_ws_idx += 1

            word_ws_groups.append([{
                "stage": "word_to_word_ws",
                "groups": torch.tensor(word_ws, dtype=torch.long)
            }])
            word_ws_group_lengths.append(torch.tensor(word_ws_lengths, dtype=torch.long))

            if self.variant.cfg.add_word_features:
                word_features.append(
                    utils.get_word_features(repaired_doc, self.variant.dictionary)
                )

        # group characters by word (leaving out whitespaces)
        x = mod_utils.group_features(
            x, word_groups, aggregation="stack"
        )

        # average character representations per word to get word representations
        word_feat = [
            torch.cat([torch.mean(t, dim=0, keepdim=True) for t in stacked_feat], dim=0)
            for stacked_feat in x
        ]

        # add additional word features to word representations
        if self.variant.cfg.add_word_features:
            additional_word_features = to(word_features, model.device)
            word_feat = [
                torch.cat([w_feat, add_w_feat], dim=1)
                for w_feat, add_w_feat in zip(word_feat, additional_word_features)
            ]

        # encode word representations
        word_lengths = [len(t) for t in word_feat]
        word_feat = to(mod_utils.pad(word_feat), model.device)
        word_padding_mask = mod_utils.padding_mask(word_feat, word_lengths)
        word_feat = model.word_encoder(word_feat, padding_mask=word_padding_mask)

        if no_detect:
            for i, repaired_string in enumerate(repaired_strings):
                outputs[i]["sed"] = [1] * len(repaired_string.split())
        else:
            threshold = kwargs.get("threshold", 0.5)
            temperature = kwargs.get("temperature", 1.0)
            sed_logits = model.head["sed"](
                [word_feat[i, :l, :] for i, l in enumerate(word_lengths)],
                groups=word_ws_groups
            )
            for i, logits in enumerate(sed_logits):
                predictions = task_utils.class_predictions(logits, threshold, temperature).tolist()
                outputs[i]["sed"] = predictions

        if output_type in {"all", "sec"} and model_cfg.output_type.endswith("plus_sec"):
            repaired_words = [repaired_string.split() for repaired_string in repaired_strings]
            detections = [output["sed"] for output in outputs]
            detections_flattened = utils.flatten(detections)
            if sum(detections_flattened) == 0:
                for i in range(len(outputs)):
                    outputs[i]["sec"] = [repaired_strings[i]]
            else:
                repaired_words_flattened = utils.flatten(repaired_words)
                detection_mask = torch.tensor(detections_flattened, dtype=torch.bool)

                # flatten the stacked character representations per word into a single tensor
                x = utils.flatten([
                    [torch.cat(tensors) for tensors in mod_utils.split(stacked_feat, group_lengths)]
                    for stacked_feat, group_lengths in zip(x, word_ws_group_lengths)
                ])
                input_lengths = torch.tensor([len(t) for t in x], dtype=torch.long)
                x = mod_utils.pad(x)

                word_feat = [
                    w_feat
                    for i, l in enumerate(word_lengths)
                    for w_feat in torch.split(
                        word_feat[i, :l, :],
                        mod_utils.tensor_to_python(word_ws_group_lengths[i], force_list=True)
                    )
                ]
                word_lengths = torch.tensor([len(t) for t in word_feat], dtype=torch.long)
                word_feat = mod_utils.pad(word_feat)

                decoder_positions = torch.cat([
                    torch.cat([torch.tensor([0]), torch.cumsum(group_lengths, dim=0)[:-1]])
                    for group_lengths in word_ws_group_lengths
                ])

                word_results = inference.run_inference(
                    model=model.head["sec"],
                    output_tokenizer=model.sec_tokenizer,
                    encoder_outputs={
                        model_cfg.input_type: x[detection_mask],
                        "word": word_feat[detection_mask]
                    },
                    encoder_lengths={
                        model_cfg.input_type: input_lengths[detection_mask],
                        "word": word_lengths[detection_mask]
                    },
                    max_length=model_cfg.sec_max_output_length,
                    input_strings=[w for w, det in zip(repaired_words_flattened, detections_flattened) if det],
                    decoder_positions=decoder_positions[detection_mask],
                    **kwargs
                )

                words_per_input = [sum(detection) for detection in detections]
                word_results_per_input = mod_utils.split(word_results, words_per_input)

                for output_idx, (words, word_results, word_detections, num_words) in enumerate(zip(
                        repaired_words,
                        word_results_per_input,
                        detections,
                        words_per_input
                )):
                    assert len(word_results) == num_words

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
                        for repaired_word, detection in zip(words, word_detections):
                            if detection:
                                result_words.append(word_results[result_idx][i])
                                result_idx += 1
                            else:
                                result_words.append(repaired_word)
                        assert result_idx == num_words
                        batch_result_str.append(" ".join(result_words))
                    outputs[output_idx]["sec"] = batch_result_str

        return outputs

    def _split_sample_for_inference(
            self,
            sample: utils.Sample,
            max_length: int,
            context_length: int,
            **kwargs: Any
    ) -> List[Tuple[int, int, int, int]]:
        if self.variant.cfg.tokenization_level == "char":
            return task_utils.get_character_windows(sample, max_length, context_length)
        elif self.variant.cfg.tokenization_level == "byte":
            return task_utils.get_byte_windows(sample, max_length, context_length)
        else:
            raise RuntimeError("should not happen")

    def _merge_inference_outputs(
            self,
            sequence: str,
            infos: List[utils.InferenceInfo],
            predictions: List[Dict[str, Any]],
            output_type: str = "all",
            **kwargs: Any
    ) -> Dict[str, Any]:
        merged_repaired_string, matched_indices = task_utils.merge_tokenization_repair_outputs(
            sequence,
            infos,
            [p["tokenization_repair"] for p in predictions],
            return_match_indices=True
        )
        outputs = {
            "tokenization_repair": merged_repaired_string
        }
        if output_type != "tokenization_repair":
            # filter sed and sec outputs based on matched indices
            word_indices = []
            word_inference_infos = []
            window_start = 0
            for i, (prediction, window) in enumerate(zip(predictions, matched_indices)):
                m_start, m_end, p_start, p_end = task_utils.align_word_prediction_with_sequence(
                    prediction["tokenization_repair"],
                    (0, len(prediction["tokenization_repair"])),
                    window,
                    prediction["sed"]
                )
                assert m_start == p_start and m_end == p_end
                word_indices.append((p_start, p_end))
                window_length = window[1] - window[0]
                window_end = window_start + window_length
                word_inference_info = utils.InferenceInfo(
                    ctx_start=window_start,
                    ctx_end=window_end,
                    window_start=window_start,
                    window_end=window_end,
                    window_idx=i,
                    length=-1  # not used anymore
                )
                word_inference_infos.append(word_inference_info)
                window_start += window_length

            assert window_start == len(merged_repaired_string)

            if output_type in {"all", "sed"}:
                sed_predictions = [p["sed"][s:e] for p, (s, e) in zip(predictions, word_indices)]
                outputs["sed"] = task_utils.merge_sed_words_outputs(
                    merged_repaired_string, word_inference_infos, sed_predictions
                )
            if output_type in {"all", "sec"}:
                sec_predictions = [
                    [p.split()[s:e] for p in prediction["sec"]]
                    for prediction, (s, e) in zip(predictions, word_indices)
                ]
                outputs["sec"] = task_utils.merge_sec_words_nmt_outputs(
                    merged_repaired_string, word_inference_infos, sec_predictions, ensure_unique=False
                )

        return outputs
