from typing import Dict, List, Union, Any

import dgl
import torch

from gnn_lib import models
from gnn_lib.data import variants, tokenization
from gnn_lib.tasks.multi_node_classification import MultiNodeClassification
from gnn_lib.utils import data_containers
from gnn_lib.tasks import utils as task_utils


class SEDWords(MultiNodeClassification):
    def __init__(self, variant_cfg: variants.DatasetVariantConfig, checkpoint_dir: str, seed: int):
        super().__init__(variant_cfg, checkpoint_dir, seed)
        self.variant_cfg: variants.SEDWordsConfig
        if self.variant_cfg.encoding_scheme == "whitespace_words":
            self.word_node_type = "word"
        elif self.variant_cfg.encoding_scheme == "tokens":
            self.word_node_type = "token"
        else:
            raise ValueError

    def _get_additional_stats(self, model: models.ModelForMultiNodeClassification) \
            -> Dict[str, data_containers.DataContainer]:
        stats = {
            f"{self.word_node_type}_accuracy": data_containers.AverageScalarContainer(
                name="word_accuracy"),
            f"{self.word_node_type}_f1_prec_rec": data_containers.F1PrecRecContainer(
                name="fpr",
                class_names={1: "word"}
            ),
            "text": data_containers.MultiTextContainer(
                name="text_samples",
                max_samples=4
            )
        }
        return stats

    def _update_stats(self,
                      model: models.ModelForMultiNodeClassification,
                      inputs: Dict[str, Any],
                      labels: Dict[str, torch.Tensor],
                      model_output: Dict[str, torch.Tensor],
                      stats: Dict[str, data_containers.DataContainer],
                      step: int,
                      log_every: int) -> None:
        super()._update_stats(model, inputs, labels, model_output, stats, step, log_every)
        text_container = stats["text"]
        if step % max(log_every // text_container.max_samples, 1) != 0 or \
                len(text_container.samples) >= text_container.max_samples:
            return

        token_ids = task_utils.get_token_ids_from_graphs(inputs["g"])
        tokenizer: tokenization.Tokenizer = model.tokenizers["token"]
        input_text = tokenizer.de_tokenize(token_ids[0])

        batch_num_words = inputs["g"].batch_num_nodes(self.word_node_type)
        word_lengths = [len(w) for w in input_text.split()]
        labels = labels[self.word_node_type][:batch_num_words[0]].tolist()
        labels_str = " ".join(str(l).ljust(pad) for l, pad in zip(labels, word_lengths))
        pred = torch.argmax(model_output[self.word_node_type][:batch_num_words[0]], 1).tolist()
        pred_str = " ".join(str(p).ljust(pad) for p, pad in zip(pred, word_lengths))

        text_container.add(
            f"""
    input:\t{input_text}
    label:\t{labels_str}
    pred:\t{pred_str}
---
            """
        )

    @torch.inference_mode()
    def inference(
            self,
            model: models.ModelForMultiNodeClassification,
            inputs: Union[List[str], dgl.DGLHeteroGraph],
            **kwargs: Any
    ) -> List[List[int]]:
        detections_dicts = super().inference(model, inputs, **kwargs)
        return [detections_dict[self.word_node_type] for detections_dict in detections_dicts]
