import collections
import copy
from typing import List, Any, Dict, Tuple, Union

import torch
from torch.nn import functional as F

from gnn_lib import models, tasks
from gnn_lib.data import utils
from gnn_lib.data.utils import Sample
from gnn_lib.modules import utils as mod_utils
from gnn_lib.utils import data_containers, Batch, to, tokenization_repair
from gnn_lib.tasks import utils as task_utils


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
        label_dict = {
            "tokenization_repair_labels": to(torch.cat(batch.info.pop("tokenization_repair_label")), device),
            "sed_labels": to(torch.cat(batch.info.pop("sed_label")), device)
        }

        data_dict = {
            "x": batch.data
        }

        if "sec_label" in batch.info:
            decoder_inputs = []
            decoder_labels = []
            decoder_group_lengths = []
            for labels, splits in zip(batch.info.pop("sec_label"), batch.info.pop("sec_label_splits")):
                word_decoder_inputs = []
                word_decoder_labels = []
                word_decoder_lengths = []
                for word_labels in torch.split(labels, splits):
                    word_decoder_inputs.append(word_labels[:-1])
                    word_decoder_labels.append(word_labels[1:])
                    word_decoder_lengths.append(len(word_labels) - 1)
                decoder_inputs.append(torch.cat(word_decoder_inputs))
                decoder_labels.append(torch.cat(word_decoder_labels))
                decoder_group_lengths.append(torch.tensor(word_decoder_lengths, dtype=torch.long))

            label_dict["sec_labels"] = to(
                mod_utils.pad(decoder_labels, val=batch.info["sec_pad_token_id"][0]).long(), device
            )
            label_dict["sec_pad_token_id"] = batch.info["sec_pad_token_id"][0]
            data_dict["sec_decoder_inputs"] = decoder_inputs
            data_dict["sec_decoder_group_lengths"] = decoder_group_lengths

        return {**data_dict, **batch.info}, label_dict

    def _calc_loss(
            self,
            labels: Dict[str, Any],
            model_output: Dict[str, Any],
            additional_losses: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        tokenization_repair_loss = F.cross_entropy(
            input=torch.cat(model_output["tokenization_repair"], dim=0),
            target=labels["tokenization_repair_labels"],
            weight=torch.tensor([1, 5, 5], dtype=torch.float, device=labels["tokenization_repair_labels"].device)
        )
        sed_loss = F.cross_entropy(
            input=torch.cat(model_output["sed"], dim=0),
            target=labels["sed_labels"]
        )
        loss = tokenization_repair_loss + sed_loss
        if "sec_labels" in labels:
            sec_loss = F.cross_entropy(
                input=model_output["sec"].reshape(-1, model_output["sec"].shape[-1]),
                target=labels["sec_labels"].reshape(-1),
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
            inputs: List[Union[str, Sample]],
            output_type: str = "all",
            no_repair: bool = False,
            **kwargs: Any
    ) -> Union[Dict[str, Any], List[List[int]]]:
        self._check_model(model)
        model = model.eval()
        model_cfg: models.ModelForTokenizationRepairPlusConfig = model.cfg
        assert output_type in {"tokenization_repair", "sed", "sec", "all"}, \
            f"unknown tokenization repair plus output type {output_type}, must be one of " \
            f"{{tokenization_repair, sed, sec, all}}"
        # the no_repair flag ensures that the input tokenization is not repaired, useful when you are sure that
        # the input has no tokenization errors or when you want to evaluate on sed benchmarks

        oversized = [len(str(ipt)) > model_cfg.max_length for ipt in inputs]
        if sum(oversized) == len(inputs):
            return [
                [0] * len(str(ipt).split()) for ipt in inputs
            ]
        org_inputs = [str(ipt) for ipt in inputs]
        inputs = [ipt for i, ipt in enumerate(inputs) if not oversized[i]]

        batch = self.variant.batch_sequences_for_inference(inputs)

        inputs = [str(ipt) for ipt in inputs]

        x, padding_mask, lengths = model.pad_inputs(batch.data, pad_val=model.input_pad_token_id)

        # embed tokens and encode input representations
        x = model.encode(x.long(), padding_mask=padding_mask)
        x = [x[i, :l] for i, l in enumerate(lengths)]

        if model_cfg.input_type == "byte":
            assert batch.info["char_groups"] is not None
            # if we have bytes as inputs group bytes into characters, since tokenization repair output is on
            # character level
            x = mod_utils.group_features(
                x, batch.info["char_groups"], aggregation="mean"
            )

        outputs = {
            "tokenization_repair": {
                "repair_tokens": [],
                "repaired_strings": []
            },
            "sed": [],
            "sec": []
        }

        # tokenization repair output
        tok_rep_logits = model.head["tokenization_repair"](x)
        for repair_logits, input_str in zip(tok_rep_logits, inputs):
            repair_tokens = task_utils.class_predictions(repair_logits).tolist()
            if no_repair:
                repaired_string = input_str
            else:
                repaired_string = tokenization_repair.repair_whitespace(input_str, repair_tokens)

            outputs["tokenization_repair"]["repair_tokens"].append(repair_tokens)
            outputs["tokenization_repair"]["repaired_strings"].append(repaired_string)

        if output_type != "tokenization_repair":
            word_groups = []
            word_ws_groups = []
            word_features = []

            for input_string, repaired_string in zip(inputs, outputs["tokenization_repair"]["repaired_strings"]):
                repaired_words, repaired_doc = utils.tokenize_words(repaired_string, return_doc=True)

                word_groups.append({
                    "stage": "char_to_word",
                    "groups": utils.get_character_groups_from_repaired_doc(list(input_string), repaired_doc)
                })

                word_ws = []
                word_ws_idx = 0
                for word in repaired_doc:
                    word_ws.append(word_ws_idx)
                    if word.whitespace_ == " ":
                        word_ws_idx += 1

                word_ws_groups.append([{
                    "stage": "word_to_word_ws",
                    "groups": torch.tensor(word_ws, dtype=torch.long)
                }])

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
            if word_features is not None:
                additional_word_features = to(word_features, model.device)
                word_feat = [
                    torch.cat([w_feat, add_w_feat], dim=1)
                    for w_feat, add_w_feat in zip(word_feat, additional_word_features)
                ]

            # encode word representations
            lengths = [len(t) for t in word_feat]
            word_feat = to(mod_utils.pad(word_feat), model.device)
            word_padding_mask = mod_utils.padding_mask(word_feat, lengths)
            word_feat = model.word_encoder(word_feat, padding_mask=word_padding_mask)

            if output_type in {"all", "sed"}:
                sed_logits = model.head["sed"](
                    [word_feat[i, :l, :] for i, l in enumerate(lengths)],
                    groups=word_ws_groups
                )
                for logits in sed_logits:
                    predictions = task_utils.class_predictions(logits).tolist()
                    outputs["sed"].append(predictions)

            if output_type in {"all", "sec"} and model_cfg.output_type.endswith("plus_sec"):
                raise NotImplementedError("plus sec inference not yet implemented")

        if output_type == "tokenization_repair":
            return outputs["tokenization_repair"]["repair_tokens"]
        elif output_type == "sed":
            sed_outputs = []
            prediction_idx = 0
            for ipt, is_oversized in zip(org_inputs, oversized):
                if is_oversized:
                    sed_outputs.append([0] * len(ipt.split()))
                else:
                    sed_outputs.append(outputs["sed"][prediction_idx])
                    prediction_idx += 1
            assert prediction_idx == len(outputs["sed"])
            return sed_outputs
        elif output_type == "sec":
            return outputs["sec"]
        else:
            return outputs
