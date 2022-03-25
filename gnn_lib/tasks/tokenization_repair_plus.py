from typing import List, Any, Dict, Tuple

import torch
from torch.nn import functional as F

from gnn_lib import models, tasks
from gnn_lib.modules import utils
from gnn_lib.utils import data_containers, BATCH, to
from gnn_lib.utils.distributed import DistributedDevice


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
            batch: BATCH,
            device: DistributedDevice
    ) -> Tuple[Dict[str, Any], Any]:
        label_dict = {
            "tokenization_repair_labels": to(torch.cat(batch.info.pop("tokenization_repair_label")), device.device),
            "sed_labels": to(torch.cat(batch.info.pop("sed_label")), device.device)
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
                utils.pad(decoder_labels, val=batch.info["sec_pad_token_id"][0]).long(), device.device
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
            inputs: List[str],
            **kwargs: Any
    ) -> List[List[str]]:
        self._check_model(model)
        model = model.eval()

        assert all(isinstance(ipt, str) for ipt in inputs)
        batch = self.variant.prepare_sequences_for_inference(inputs)

        raise NotImplementedError
