from typing import List, Any, Dict, Tuple, Union

import torch
from torch.nn import functional as F

from gnn_lib import models, tasks
from gnn_lib.modules import inference, utils
from gnn_lib.utils import data_containers, BATCH, to
from gnn_lib.utils.distributed import DistributedDevice
from gnn_lib.tasks import utils as task_utils


class Seq2Seq(tasks.Task):
    expected_models = models.ModelForSeq2Seq  # type: ignore

    def _get_additional_stats(self, model: models.ModelForSeq2Seq) -> Dict[str, data_containers.DataContainer]:
        return {
            "text": data_containers.MultiTextContainer(
                name="text_samples",
                max_samples=4
            ),
            "seq_length": data_containers.HistogramContainer(
                name="input_sequence_length"
            )
        }

    def _prepare_inputs_and_labels(self,
                                   batch: BATCH,
                                   device: DistributedDevice) -> Tuple[Dict[str, Any], Any]:
        decoder_inputs = []
        decoder_labels = []
        for labels in batch.info.pop("label"):
            decoder_inputs.append(labels[:-1])
            decoder_labels.append(labels[1:])

        decoder_labels = to(utils.pad(decoder_labels, val=batch.info["pad_token_id"][0]).long(), device.device)

        return (
            {
                "x": batch.data,
                "decoder_inputs": decoder_inputs,
                **batch.info
            },
            {"labels": decoder_labels, "pad_token_id": batch.info["pad_token_id"][0]}
        )

    def _calc_loss(self,
                   labels: Dict[str, Any],
                   model_output: torch.Tensor,
                   additional_losses: Dict[str, torch.Tensor]) -> torch.Tensor:
        return F.cross_entropy(
            input=model_output.reshape(-1, model_output.shape[-1]),
            target=labels["labels"].reshape(-1),
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

        text_container: data_containers.MultiTextContainer = stats["text"]  # type: ignore
        if (
                step % max(total_steps // text_container.max_samples, 1) != 0
                or len(text_container.samples) >= text_container.max_samples
        ):
            return

        input_str = model.input_tokenizer.de_tokenize(inputs["x"][0].tolist())

        length = len(inputs["decoder_inputs"][0])
        label_ids = labels["labels"][0, :length].tolist()
        output_ids = torch.argmax(model_output[0, :length], dim=1).tolist()

        labels_str = model.output_tokenizer.de_tokenize(label_ids)
        pred_str = model.output_tokenizer.de_tokenize(output_ids)

        # this multiline string looks silly but has to be that way, since tensorboard formats text as markdown
        text_container.add(
            f"""
    input:\t{input_str}
    label:\t{labels_str}
    pred:\t{pred_str}
---
            """
        )

    @torch.inference_mode()
    def inference(self,
                  model: models.ModelForSeq2Seq,
                  inputs: Union[BATCH, List[str]],
                  **kwargs: Any) -> List[List[str]]:
        self._check_model(model)
        model = model.eval()

        got_str_input = task_utils.is_string_input(inputs)
        if got_str_input:
            batch = self.variant.prepare_sequences_for_inference(inputs)
        else:
            batch = inputs

        encoder_inputs, encoder_padding_mask = model.pad_inputs(batch.data)
        encoder_outputs = model.encode(encoder_inputs, encoder_padding_mask)

        return inference.run_inference(
            model=model.head,
            output_tokenizer=model.output_tokenizer,
            encoder_outputs={"encoder_outputs": encoder_outputs},
            encoder_lengths={"encoder_outputs": torch.tensor([len(t) for t in batch.data], dtype=torch.long)},
            max_length=model.cfg.max_length,
            **kwargs
        )
