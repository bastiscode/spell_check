import pprint
from typing import List, Optional, Union, Any, Dict

import torch
from tqdm import tqdm

from gnn_lib.api.utils import (
    ModelInfo,
    StringInputOutput,
    download_data,
    download_configs,
    download_model,
    load_experiment,
    get_cpu_info,
    get_gpu_info,
    get_string_dataset_and_loader,
    reorder_data, get_device_info
)
from gnn_lib.modules import inference
from gnn_lib.tasks import graph_sed_words, graph_sed_sequence
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


class SpellingErrorDetector:
    def __init__(
            self,
            model_dir: str,
            device: Union[str, int],
            **kwargs: Dict[str, Any]
    ) -> None:
        self.logger = common.get_logger("SPELLING_ERROR_DETECTION")

        if device != "cpu" and not torch.cuda.is_available():
            self.logger.info(f"could not find a GPU, using CPU as fallback option")
            device = "cpu"

        self.device = torch.device(device)
        self.logger.info(f"running spelling error detection on device {get_device_info(self.device)}")

        self.cfg, self.task, self.model = load_experiment(
            model_dir,
            self.device,
            kwargs.get("override_env_vars"),
            kwargs.get("keep_existing_env_vars", False)
        )

        assert isinstance(self.task, sed_words.GraphSEDWords) or isinstance(self.task, sed_sequence.GraphSEDSequence), \
            f"expected experiment to be of type SEDWords or SEDSequence, but got {type(self.task)}"

        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

        self.max_length = self.model.embedding.node_embedding.pos_emb["token"].max_len or float("inf")

    @property
    def task_name(self) -> str:
        return "sed_words" if isinstance(self.task, sed_words.GraphSEDWords) else "sed_sequence"

    @property
    def model_name(self) -> str:
        return self.cfg.experiment_name

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

        logger = common.get_logger("DOWNLOAD")

        data_dir = download_data(force_download, logger, cache_dir)
        config_dir = download_configs(force_download, logger, cache_dir)
        model_dir = download_model(
            task=task,
            name=model,
            cache_dir=cache_dir,
            force_download=force_download,
            logger=logger
        )
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
        dataset, loader = get_string_dataset_and_loader(
            inputs,
            sort_by_length,
            batch_size
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

        all_outputs = []
        for i, (batch, info) in enumerate(pbar):
            batch_length = sum(info["lengths"])
            pbar.set_description(
                f"[Batch {i + 1}] Detecting spelling errors in {len(batch):,} sequences "
                f"with {batch_length:,} characters in total"
            )

            outputs = self.task.inference(self.model, batch, **inference_kwargs)
            all_outputs.extend(outputs)

            pbar.update(batch_length)

        pbar.close()
        return reorder_data(all_outputs, dataset.indices)

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
                or (isinstance(inputs, list) and len(inputs) > 0 and isinstance(inputs[0], str))
        ), f"input needs to be a string or a non empty list of strings"

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

    def to(self, device: Union[str, int]) -> "SpellingErrorDetector":
        self.device = torch.device(device)
        self.model.to(self.device)
        return self
