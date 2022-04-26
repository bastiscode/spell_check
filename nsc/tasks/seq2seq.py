from typing import List, Any, Dict, Tuple, Union

import torch
from torch.nn import functional as F

from nsc import models, tasks
from nsc.data.utils import Sample
from nsc.modules import inference, utils
from nsc.utils import data_containers, Batch, to
from nsc.tasks import utils as task_utils


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

    def _prepare_inputs_and_labels(
            self,
            batch: Batch,
            device: torch.device
    ) -> Tuple[Dict[str, Any], Any]:
        decoder_inputs = []
        decoder_labels = []

        invalid_indices = set()
        for i, labels in enumerate(batch.info.pop("label")):
            if len(labels) - 1 > 1024:
                # this is kind of a temporary hack to skip too long decoding sequences that cause OOM
                self.logger.warning(f"skipping sample with decoder length {len(labels) - 1}")
                invalid_indices.add(i)
                continue
            decoder_inputs.append(torch.tensor(labels[:-1], dtype=torch.long))
            decoder_labels.append(torch.tensor(labels[1:], dtype=torch.long))

        decoder_labels = to(utils.pad(decoder_labels, val=batch.info["pad_token_id"][0]).long(), device)

        return (
            {
                "x": task_utils.exclude_indices(batch.data, invalid_indices),
                "decoder_inputs": decoder_inputs
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
    def inference(
            self,
            model: models.ModelForSeq2Seq,
            inputs: Union[Batch, List[Union[str, Sample]]],
            **kwargs: Any
    ) -> List[List[str]]:
        self._check_model(model)
        model = model.eval()
        model_cfg: models.ModelForSeq2SeqConfig = model.cfg

        if isinstance(inputs, Batch):
            batch = inputs
        else:
            batch = self._batch_sequences_for_inference(inputs)

        encoder_inputs, encoder_padding_mask = model.pad_inputs(batch.data)
        encoder_outputs = model.encode(encoder_inputs, encoder_padding_mask)

        return inference.run_inference(
            model=model.head,
            output_tokenizer=model.output_tokenizer,
            encoder_outputs={"encoder_outputs": encoder_outputs},
            encoder_lengths={"encoder_outputs": torch.tensor([len(t) for t in batch.data], dtype=torch.long)},
            max_length=model_cfg.max_output_length,
            **kwargs
        )
