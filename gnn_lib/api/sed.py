import os
import pprint
from typing import List, Optional, Union, Any, Dict

import torch
from torch import autocast
from tqdm import tqdm

from gnn_lib.api.utils import (
    ModelInfo,
    StringInputOutput,
    load_experiment,
    get_string_dataset_and_loader,
    reorder_data, get_device_info, _APIBase, get_inference_dataset_and_loader, load_text_file
)
from gnn_lib.data import DatasetVariants
from gnn_lib.data.utils import clean_sequence
from gnn_lib.modules import inference
from gnn_lib.tasks import graph_sed_words, graph_sed_sequence, sed_words, sed_sequence, tokenization_repair_plus
from gnn_lib.utils import common

__all__ = ["get_available_spelling_error_detection_models", "SpellingErrorDetector"]

Detections = Union[str, List[int], List[List[int]]]


def get_available_spelling_error_detection_models() -> List[ModelInfo]:
    return [
        ModelInfo(
            task="sed_words",
            name="gnn_default",
            description="Graph Neural Network which extends the default Transformer fully connected graph "
                        "with word nodes and word features"
        ),
        ModelInfo(
            task="sed_words",
            name="gnn_cliques_wfc",
            description="Graph Neural Network which processes language graphs with fully connected word nodes "
                        "and fully connected sub-word cliques for each word"
        ),
        ModelInfo(
            task="sed_sequence",
            name="gnn_default",
            description="Graph Neural Network which extends the default Transformer fully connected graph "
                        "with word nodes, word features and a sequence node"
        ),
        ModelInfo(
            task="sed_sequence",
            name="gnn_cliques_wfc",
            description="Graph Neural Network which processes language graphs with fully connected word nodes, "
                        "fully connected sub-word cliques for each word and a sequence node"
        )
    ]


class SpellingErrorDetector(_APIBase):
    def __init__(
            self,
            model_dir: str,
            device: Union[str, int],
            **kwargs: Dict[str, Any]
    ) -> None:
        logger = common.get_logger("SPELLING_ERROR_DETECTION")

        if device != "cpu" and not torch.cuda.is_available():
            logger.info(f"could not find a GPU, using CPU as fallback option")
            device = "cpu"

        device = torch.device(device)
        logger.info(f"running spelling error detection on device {get_device_info(device)}")

        cfg, task, model = load_experiment(
            model_dir,
            device,
            kwargs.get("override_env_vars"),
            kwargs.get("keep_existing_env_vars", False)
        )

        assert (
                isinstance(task, graph_sed_words.GraphSEDWords)
                or isinstance(task, sed_words.SEDWords)
                or isinstance(task, graph_sed_sequence.GraphSEDSequence)
                or isinstance(task, sed_sequence.SEDSequence)
                or isinstance(task, tokenization_repair_plus.TokenizationRepairPlus)
        ), \
            f"expected experiment to be of type SEDWords, GraphSEDWords, " \
            f"SEDSequence, GraphSEDSequence or TokenizationRepairPlus, but got {type(task)}"

        self.max_length = model.cfg.max_length

        super().__init__(model, cfg, task, device, logger)

    @property
    def task_name(self) -> str:
        return (
            "sed_sequence" if self.task.variant_cfg.type == DatasetVariants.SED_SEQUENCE
            else "sed_words"
        )

    @staticmethod
    def from_pretrained(
            task: str = "sed_words",
            model: str = "gnn_default",
            device: Union[str, int] = "cuda",
            cache_dir: Optional[str] = None,
            force_download: bool = False
    ) -> "SpellingErrorDetector":
        assert any(model == m.name and task == m.task for m in get_available_spelling_error_detection_models()), \
            f"task {task} and model {model} do not match any of the available models:\n" \
            f"{pprint.pformat(get_available_spelling_error_detection_models())}"

        model_dir, data_dir, config_dir = super()._download(task, model, cache_dir, force_download)

        return SpellingErrorDetector(
            model_dir,
            device,
            **{
                "override_env_vars": {
                    "GNN_LIB_DATA_DIR": data_dir,
                    "GNN_LIB_CONFIG_DIR": config_dir
                }
            }
        )

    @staticmethod
    def from_experiment(
            experiment_dir: str,
            device: Union[str, int] = "cuda"
    ) -> "SpellingErrorDetector":
        return SpellingErrorDetector(
            experiment_dir,
            device,
            **{
                "keep_existing_env_vars": True
            }
        )

    @torch.inference_mode()
    def _detect_text_raw(
            self,
            inputs: Union[str, List[str]],
            threshold: float = 0.5,
            batch_size: int = 16,
            sort_by_length: bool = True,
            show_progress: bool = False
    ) -> List[List[int]]:
        if isinstance(inputs, str):
            inputs = load_text_file(inputs)

        inputs = [clean_sequence(ipt) for ipt in inputs]

        dataset, loader = get_inference_dataset_and_loader(
            inputs,
            variant=self.task.variant,
            max_length=self.max_length,
            sort_by_length=sort_by_length,
            batch_size=batch_size
        )

        pbar = tqdm(
            loader,
            total=dataset.char_length(),
            ascii=True,
            leave=False,
            disable=not show_progress,
            unit="char"
        )

        inference_kwargs = {
            "threshold": threshold
        }

        if isinstance(self.task, tokenization_repair_plus.TokenizationRepairPlus):
            inference_kwargs["output_type"] = "tokenization_repair"
            inference_kwargs["no_repair"] = True

        all_outputs = []
        for i, (batch, infos, _) in enumerate(pbar):
            batch_length = sum(info.ctx_end - info.ctx_start for info in infos)
            pbar.set_description(
                f"[Batch {i + 1}] Detecting spelling errors in {len(batch):,} sequences "
                f"with {batch_length:,} characters in total"
            )

            # this is a slight hack for now, because fp32 on cpu throws an error even when enabled=False
            if self.mixed_precision_enabled:
                with autocast(
                        device_type=self.device.type,
                        dtype=self._mixed_precision_dtype,
                        enabled=self.mixed_precision_enabled
                ):
                    outputs = self.task.inference(self.model, batch, **inference_kwargs)
            else:
                outputs = self.task.inference(self.model, batch, **inference_kwargs)

            all_outputs.extend(outputs)
            pbar.update(batch_length)

        pbar.close()
        all_outputs = reorder_data(all_outputs, dataset.indices)
        return self.task.variant.postprocess_inference_outputs(inputs, dataset.sample_infos, all_outputs)

    def detect_text(
            self,
            inputs: StringInputOutput,
            threshold: float = 0.5,
            batch_size: int = 16,
            sort_by_length: bool = True,
            show_progress: bool = False
    ) -> Union[List[int], List[List[int]]]:
        input_is_string = isinstance(inputs, str)
        assert (
                input_is_string
                or all(isinstance(ipt, str) for ipt in inputs)
        ), f"input needs to be a string or a list of strings"

        outputs = self._detect_text_raw(
            [inputs] if input_is_string else inputs,
            threshold,
            batch_size,
            sort_by_length,
            show_progress
        )
        return outputs[0] if input_is_string else outputs

    def detect_file(
            self,
            input_file_path: str,
            output_file_path: Optional[str] = None,
            threshold: float = 0.5,
            batch_size: int = 16,
            sort_by_length: bool = True,
            show_progress: bool = True
    ) -> Optional[Union[List[int], List[List[int]]]]:
        outputs = self._detect_text_raw(
            input_file_path, threshold, batch_size, sort_by_length, show_progress
        )
        if output_file_path is not None:
            with open(output_file_path, "w", encoding="utf8") as out_file:
                for output in outputs:
                    out_file.write(inference.inference_output_to_str(output))
                    out_file.write("\n")
            return None
        else:
            return outputs
