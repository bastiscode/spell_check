from typing import List, Any, Dict, Tuple, Union

import torch
from torch.nn import functional as F

from nsc import models, tasks
from nsc.data.utils import Sample
from nsc.modules import utils
from nsc.tasks import utils as task_utils
from nsc.utils import data_containers, Batch, to


class Token2Seq(tasks.Task):
    expected_models = models.ModelForToken2Seq  # type: ignore

    def _get_additional_stats(self, model: models.ModelForToken2Seq) -> Dict[str, data_containers.DataContainer]:
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
        decoder_inputs = []
        decoder_labels = []
        decoder_group_lengths = []

        invalid_indices = set()
        for i, labels in enumerate(batch.info.pop("label")):
            sample_decoder_inputs = []
            sample_decoder_labels = []
            sample_decoder_lengths = []
            for label in labels:
                sample_decoder_inputs.extend(label[:-1])
                sample_decoder_labels.extend(label[1:])
                sample_decoder_lengths.append(len(label) - 1)
            if sum(sample_decoder_lengths) > 1024:
                # this is kind of a temporary hack to skip too long decoding sequences that cause OOM
                self.logger.warning(f"skipping sample with decoder length {sum(sample_decoder_lengths)}")
                invalid_indices.add(i)
                continue
            decoder_inputs.append(torch.tensor(sample_decoder_inputs, dtype=torch.long))
            decoder_labels.append(torch.tensor(sample_decoder_labels, dtype=torch.long))
            decoder_group_lengths.append(torch.tensor(sample_decoder_lengths, dtype=torch.long))

        decoder_labels = to(utils.pad(decoder_labels, val=batch.info["pad_token_id"][0]).long(), device)

        return (
            {
                "x": task_utils.exclude_indices(batch.data, invalid_indices),
                "decoder_inputs": decoder_inputs,
                "decoder_group_lengths": decoder_group_lengths,
                "encoder_group_lengths": task_utils.exclude_indices(batch.info["encoder_group_lengths"],
                                                                    invalid_indices)
            },
            {"labels": decoder_labels, "pad_token_id": batch.info["pad_token_id"][0]}
        )

    def _calc_loss(self,
                   labels: Dict[str, Any],
                   model_output: torch.Tensor,
                   additional_losses: Dict[str, torch.Tensor]) -> torch.Tensor:
        return F.cross_entropy(
            input=model_output.view(-1, model_output.shape[-1]),
            target=labels["labels"].view(-1),
            ignore_index=labels["pad_token_id"]
        ) + sum(additional_losses.values())

    def _update_stats(self,
                      model: models.ModelForGraph2Seq,
                      inputs: Dict[str, Any],
                      labels: Any,
                      model_output: Any,
                      stats: Dict[str, data_containers.DataContainer],
                      step: int,
                      total_steps: int) -> None:
        sequence_length_container = stats["seq_length"]
        sequence_length_container.add([len(t) for t in inputs["x"]])

    @torch.inference_mode()
    def inference(self,
                  model: models.ModelForToken2Seq,
                  inputs: Union[Batch, List[Union[str, Sample]]],
                  **kwargs: Any) -> List[List[str]]:
        raise NotImplementedError
