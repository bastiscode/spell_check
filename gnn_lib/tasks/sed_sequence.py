from typing import Dict, List, Union, Any

import torch
from gnn_lib import models
from gnn_lib.modules import utils
from gnn_lib.tasks import utils as task_utils
from gnn_lib.tasks.sequence_classification import SequenceClassification
from gnn_lib.utils import BATCH


class SEDSequence(SequenceClassification):
    @torch.inference_mode()
    def inference(
            self,
            model: models.ModelForSequenceClassification,
            inputs: Union[List[str], BATCH],
            **kwargs: Any
    ) -> List[Dict[str, List]]:
        self._check_model(model)
        model = model.eval()

        got_str_input = isinstance(inputs, list) and isinstance(inputs[0], str)
        if got_str_input:
            g, infos = self.variant.prepare_sequences_for_inference(inputs)
        else:
            g, infos = inputs

        outputs, _ = model(g, **infos)

        return_logits = kwargs.get("return_logits", False)

        batch_predictions_dict = {}
        for node_type in model.cfg.num_classes:
            num_nodes = g.batch_num_nodes(node_type)
            if "groups" in infos and node_type in infos["groups"][0]:
                num_nodes = [max(group[node_type][-1]["groups"]) + 1 for group in infos["groups"]]

            if return_logits:
                predictions = utils.tensor_to_python(outputs[node_type], force_list=True)
            else:
                threshold = kwargs.get(f"{node_type}_threshold", kwargs.get("threshold", 0.5))
                temperature = kwargs.get(f"{node_type}_temperature", kwargs.get("temperature", 1.0))
                predictions = utils.tensor_to_python(
                    task_utils.class_predictions(
                        outputs[node_type],
                        threshold,
                        temperature
                    ),
                    force_list=True
                )

            predictions = utils.split(
                predictions,
                num_nodes
            )
            batch_predictions_dict[node_type] = predictions

        return [
            {node_type: predictions[i] for node_type, predictions in batch_predictions_dict.items()}
            for i in range(g.batch_size)
        ]
