from typing import Union, List, Any, Dict, Tuple

import dgl
import torch
from torch.nn import functional as F

from gnn_lib import models, tasks
from gnn_lib.data import tokenization
from gnn_lib.data.utils import Sample
from gnn_lib.modules import inference, utils
from gnn_lib.tasks import utils as task_utils
from gnn_lib.utils import data_containers, Batch


class Graph2Seq(tasks.Task):
    expected_model = models.ModelForGraph2Seq  # type: ignore

    def _get_additional_stats(self, model: models.ModelForGraph2Seq) -> Dict[str, data_containers.DataContainer]:
        return {
            "text": data_containers.MultiTextContainer(
                name="text_samples",
                max_samples=4
            )
        }

    def _prepare_inputs_and_labels(
            self,
            batch: Batch,
            device: torch.device
    ) -> Tuple[Dict[str, Any], Any]:
        decoder_inputs = []
        decoder_labels = []
        decoder_lengths = []
        for labels in batch.info.pop("label"):
            decoder_inputs.append(torch.tensor(labels[:-1], device=device, dtype=torch.long))
            decoder_labels.append(torch.tensor(labels[1:], device=device, dtype=torch.long))
            decoder_lengths.append(len(labels) - 1)

        pad_token_id = float(batch.info["pad_token_id"][0])
        decoder_inputs = torch.nn.utils.rnn.pad_sequence(
            decoder_inputs,
            padding_value=pad_token_id,
            batch_first=True
        ).long()
        decoder_labels = torch.nn.utils.rnn.pad_sequence(
            decoder_labels,
            padding_value=pad_token_id,
            batch_first=True
        ).long()
        decoder_lengths = torch.tensor(decoder_lengths, device=device, dtype=torch.long)

        return (
            {
                "g": batch.data,
                "decoder_inputs": decoder_inputs,
                "decoder_lengths": decoder_lengths,
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
        text_container: data_containers.MultiTextContainer = stats["text"]  # type: ignore
        if (
                step % max(total_steps // text_container.max_samples, 1) != 0
                or len(text_container.samples) >= text_container.max_samples
        ):
            return

        token_tokenizer: tokenization.Tokenizer = model.input_tokenizers["token"]

        token_ids = task_utils.get_token_ids_from_graphs(inputs["g"])
        input_str = token_tokenizer.de_tokenize(token_ids[0])

        length = inputs["decoder_lengths"][0]
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
            model: models.ModelForGraph2Seq,
            inputs: List[Union[str, Sample]],
            **kwargs: Any
    ) -> List[List[str]]:
        self._check_model(model)
        model = model.eval()
        model_cfg: models.ModelForGraph2SeqConfig = model.cfg

        batch = self._batch_sequences_for_inference(inputs)

        g = model.encode(batch.data)
        encoder_outputs, encoder_lengths = utils.graph2seq_encoder_outputs_from_graph(
            g, model_cfg.context_node_types, model_cfg.hidden_feature
        )

        return inference.run_inference(
            model=model,
            output_tokenizer=model.output_tokenizer,
            encoder_outputs=encoder_outputs,
            encoder_lengths=encoder_lengths,
            max_length=model_cfg.max_output_length,
            **kwargs
        )
