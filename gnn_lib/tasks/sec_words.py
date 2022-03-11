from typing import Dict, List, Union, Any

import dgl
import torch
from torch.nn import functional as F

from gnn_lib import models
from gnn_lib.data import variants, utils, tokenization
from gnn_lib.tasks.multi_node_classification import MultiNodeClassification
from gnn_lib.utils import data_containers


class SECWords(MultiNodeClassification):
    def __init__(self, variant_cfg: variants.DatasetVariantConfig, checkpoint_dir: str, seed: int):
        super().__init__(variant_cfg, checkpoint_dir, seed)
        self.variant_cfg: variants.SECWordsConfig
        self.variant: variants.SECWords
        if self.variant_cfg.encoding_scheme == "words":
            self.word_node_type = "word"
        elif self.variant_cfg.encoding_scheme == "tokens":
            self.word_node_type = "token"
        else:
            raise ValueError
        self.unk_token_id = self.variant.word_tokenizer.token_to_id(tokenization.UNK)

    def _get_additional_stats(self,
                              cfg: models.ModelForMultiNodeClassificationConfig
                              ) -> Dict[str, data_containers.DataContainer]:
        stats = {}
        for node_type, num_classes in cfg.num_classes.items():
            stats[f"{node_type}_accuracy"] = data_containers.AverageScalarContainer(name=f"{node_type}_accuracy")
        return stats

    def _calc_loss(self,
                   labels: Dict[str, torch.Tensor],
                   model_output: Dict[str, torch.Tensor],
                   additional_losses: Dict[str, torch.Tensor],
                   stats: Dict[str, data_containers.DataContainer]) -> torch.Tensor:
        loss = None
        for node_type, pred in model_output.items():
            node_type_loss = F.cross_entropy(pred, labels[node_type])
            if loss is None:
                loss = node_type_loss
            else:
                loss += node_type_loss

            predictions = torch.argmax(pred, dim=1)
            stats[f"{node_type}_accuracy"].add((labels[node_type] == predictions).cpu())

        loss += sum(additional_losses.values())

        return loss

    @torch.inference_mode()
    def inference(
            self,
            model: models.ModelForMultiNodeClassification,
            inputs: List[str],
            **kwargs: Any
    ) -> List[str]:
        self._check_model(model)
        model.eval()

        assert isinstance(inputs, list) and isinstance(inputs[0], str)
        self.variant: variants.SECWords
        g, infos = self.variant.prepare_sequences_for_inference(inputs)

        tokenized = utils.tokenize_words_batch(inputs, return_docs=True)
        predictions_dicts = super().inference(model, g, **kwargs)

        outputs = []
        for (input_words, doc), predictions_dict in zip(tokenized, predictions_dicts):
            predictions = predictions_dict[self.word_node_type]
            assert len(doc) == len(predictions)
            # if model predicts [UNK], just keep the input word at this position
            words = [
                self.variant.word_tokenizer.id_to_token(pred)
                if pred != self.unk_token_id else input_words[i]
                for i, pred in enumerate(predictions)
            ]
            output = utils.de_tokenize_words(words, doc)
            outputs.append(output)

        return outputs
