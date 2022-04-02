import collections
from typing import List, Union, Tuple, Dict, Any

import dgl
import torch
from torch.nn import functional as F

from gnn_lib import models, tasks
from gnn_lib.modules import utils, inference
from gnn_lib.modules.utils import pad
from gnn_lib.utils import Batch


class MultiNode2Seq(tasks.Task):
    expected_models = models.ModelForMultiNode2Seq  # type: ignore

    def _prepare_inputs_and_labels(
            self,
            batch: Batch,
            device: torch.device
    ) -> Tuple[Dict[str, Any], Any]:
        decoder_inputs = collections.defaultdict(list)
        decoder_labels = collections.defaultdict(list)
        decoder_lengths = collections.defaultdict(list)

        for label_dict in batch.info.pop("label"):
            for node_type, labels in label_dict.items():
                graph_decoder_labels = []
                graph_decoder_inputs = []
                graph_decoder_lengths = []
                for label in labels:
                    graph_decoder_inputs.extend(label[:-1])
                    graph_decoder_labels.extend(label[1:])
                    graph_decoder_lengths.append(len(label) - 1)
                decoder_inputs[node_type].append(
                    torch.tensor(graph_decoder_inputs, device=device, dtype=torch.long)
                )
                decoder_labels[node_type].append(
                    torch.tensor(graph_decoder_labels, device=device, dtype=torch.long)
                )
                decoder_lengths[node_type].append(
                    torch.tensor(graph_decoder_lengths, device=device, dtype=torch.long)
                )

        pad_token_ids = {
            node_type: pad_token_id for node_type, pad_token_id in batch.info["pad_token_id"][0].items()
        }

        decoder_inputs = {
            node_type: pad(ipt, float(pad_token_ids[node_type])).long()
            for node_type, ipt in decoder_inputs.items()
        }
        decoder_labels = {
            node_type: pad(lab, float(pad_token_ids[node_type])).long()
            for node_type, lab in decoder_labels.items()
        }

        return (
            {
                "g": batch.data,
                "decoder_inputs": decoder_inputs,
                "decoder_lengths": dict(decoder_lengths),
                **batch.info
            },
            {"labels": decoder_labels, "pad_token_id": pad_token_ids}
        )

    def _calc_loss(self,
                   labels: Dict[str, Any],
                   model_output: Dict[str, torch.Tensor],
                   additional_losses: Dict[str, torch.Tensor]) -> torch.Tensor:
        return sum(
            F.cross_entropy(
                input=output.reshape(-1, output.shape[-1]),
                target=labels["labels"][node_type].reshape(-1),
                ignore_index=labels["pad_token_id"][node_type]
            ) for node_type, output in model_output.items()
        ) + sum(additional_losses.values())

    @torch.inference_mode()
    def inference(
            self,
            model: models.ModelForMultiNode2Seq,
            inputs: Union[List[str], Tuple[dgl.DGLHeteroGraph, List[Dict[str, Any]]]],
            **kwargs: Any
    ) -> Dict[str, List[List[List[str]]]]:
        raise NotImplementedError
