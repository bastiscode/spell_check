import dataclasses
import os
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
    reorder_data,
    load_text_file,
    get_device_info
)
from gnn_lib.data import index
from gnn_lib.modules import inference
from gnn_lib.tasks import sec_nmt, sec_words_nmt
from gnn_lib.utils import common

__all__ = ["get_available_spelling_error_correction_models", "SpellingErrorCorrector"]

Detections = Union[List[int], List[List[int]]]


def get_available_spelling_error_correction_models() -> List[ModelInfo]:
    return [
        ModelInfo(
            task="sec",
            name="transformer_words_nmt",
            description="Transformer model that corrects sequences by translating each word individually "
                        "from misspelled to correct"
        ),
        ModelInfo(
            task="sec",
            name="transformer_nmt",
            description="Transformer model that translates a sequence with spelling errors into a "
                        "sequence without spelling errors"
        )
    ]


@dataclasses.dataclass
class Search:
    pass


class GreedySearch(Search):
    pass


class SampleSearch(Search):
    top_k: int = 5


class BestFirstSearch(Search):
    pass


class BeamSearch(Search):
    beam_width: int = 5


@dataclasses.dataclass
class SpellingCorrectionScore:
    normalize_by_length: bool = True
    alpha: float = 1.0
    mode: str = "log_likelihood"
    prefix_index: Optional[index.PrefixIndex] = None


def inference_kwargs_from_search_and_score(search: Search, score: SpellingCorrectionScore) -> Dict[str, Any]:
    inference_kwargs = {}
    if isinstance(search, GreedySearch):
        inference_kwargs["search_mode"] = "greedy"
    elif isinstance(search, SampleSearch):
        inference_kwargs["search_mode"] = "sample"
        inference_kwargs["sample_top_k"] = search.top_k
    elif isinstance(search, BeamSearch):
        inference_kwargs["search_mode"] = "beam"
        inference_kwargs["beam_width"] = search.beam_width
    elif isinstance(search, BestFirstSearch):
        inference_kwargs["search_mode"] = "best_first"
    else:
        raise RuntimeError(f"unknown search specification {search.__class__.__name__}")

    if score.mode != "log_likelihood" and score.prefix_index is None:
        logger = common.get_logger("DOWNLOAD")
        logger.info(f"score mode is {score.mode}, but not prefix index is given, downloading data to use "
                    f"pretrained prefix index")
        data_dir = download_data(False, logger)
        score.prefix_index = index.PrefixIndex(
            os.path.join(data_dir, "prefix_index", "merged_train_100k_prefix_index.pkl")
        )

    inference_kwargs["score_fn"] = inference.spelling_correction_score(
        score.mode,
        score.prefix_index,
        score.normalize_by_length,
        score.alpha
    )

    return inference_kwargs


class SpellingErrorCorrector:
    def __init__(
            self,
            model_dir: str,
            device: Union[str, int],
            **kwargs: Dict[str, Any]
    ) -> None:
        self.logger = common.get_logger("SPELLING_ERROR_CORRECTION")

        if device != "cpu" and not torch.cuda.is_available():
            self.logger.info(f"could not find a GPU, using CPU as fallback option")
            device = "cpu"

        self.device = torch.device(device)
        self.logger.info(f"running spelling error correction on device {get_device_info(self.device)}")

        self.cfg, self.task, self.model = load_experiment(
            model_dir,
            self.device,
            kwargs.get("override_env_vars"),
            kwargs.get("keep_existing_env_vars", False)
        )

        assert isinstance(self.task, sec_nmt.SECNMT) or isinstance(self.task, sec_words_nmt.SECWordsNMT), \
            f"expected experiment to be of type SECNMT or SECWordsNMT, but got {type(self.task)}"

        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

        self.max_length = self.model.embedding.node_embedding.pos_emb["token"].max_len or float("inf")

    @property
    def task_name(self) -> str:
        return "sec"

    @property
    def model_name(self) -> str:
        return self.cfg.experiment_name

    @staticmethod
    def from_pretrained(
            model: str = "transformer_words_nmt",
            device: Union[str, int] = "cuda",
            cache_dir: Optional[str] = None,
            force_download: bool = False
    ) -> "SpellingErrorCorrector":
        assert any(model == m.name for m in get_available_spelling_error_correction_models()), \
            f"model {model} does not match any of the available models:\n" \
            f"{pprint.pformat(get_available_spelling_error_correction_models())}"

        logger = common.get_logger("DOWNLOAD")

        data_dir = download_data(force_download, logger, cache_dir)
        config_dir = download_configs(force_download, logger, cache_dir)
        model_dir = download_model(
            task="sec",
            name=model,
            cache_dir=cache_dir,
            force_download=force_download,
            logger=logger
        )
        return SpellingErrorCorrector(
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
    ) -> "SpellingErrorCorrector":
        return SpellingErrorCorrector(
            experiment_dir,
            device,
            **{
                "keep_existing_env_vars": True
            }
        )

    @torch.inference_mode()
    def _correct_text_raw(
            self,
            inputs: Union[str, List[str]],
            detections: Optional[List[List[int]]] = None,
            search: Search = GreedySearch(),
            score: SpellingCorrectionScore = SpellingCorrectionScore(),
            batch_size: int = 16,
            sort_by_length: bool = True,
            show_progress: bool = False
    ) -> List[str]:
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

        inference_kwargs = inference_kwargs_from_search_and_score(search, score)

        all_outputs = []
        for i, (batch, info) in enumerate(pbar):
            if detections is not None:
                batch_detections = [detections[idx] for idx in info["indices"]]
                assert all(len(det) == len(seq.split()) for det, seq in zip(batch_detections, batch)), \
                    f"expected to have a detection flag of 0 or 1 for every word in the batch " \
                    f"but got {pprint.pformat(batch_detections)} as detections and {pprint.pformat(batch)} as batch"
                inference_kwargs.update({
                    "detections": batch_detections
                })

            batch_length = sum(info["lengths"])
            pbar.set_description(
                f"[Batch {i + 1}] Correcting spelling errors of {len(batch):,} sequences "
                f"with {batch_length:,} characters in total"
            )

            outputs = self.task.inference(self.model, batch, **inference_kwargs)
            all_outputs.extend(outputs)

            pbar.update(batch_length)

        pbar.close()
        reordered_outputs = reorder_data(all_outputs, dataset.indices)
        return [inference.inference_output_to_str(output) for output in reordered_outputs]

    def correct_text(
            self,
            inputs: StringInputOutput,
            detections: Optional[Detections] = None,
            search: Search = GreedySearch(),
            score: SpellingCorrectionScore = SpellingCorrectionScore(),
            batch_size: int = 16,
            sort_by_length: bool = True,
            show_progress: bool = False
    ) -> StringInputOutput:
        input_is_string = isinstance(inputs, str)
        assert (
                input_is_string
                or (isinstance(inputs, list) and len(inputs) > 0 and isinstance(inputs[0], str))
        ), f"input needs to be a string or a non empty list of strings"

        outputs = self._correct_text_raw(
            [inputs] if input_is_string else inputs,
            [detections] if input_is_string and detections is not None else detections,
            search,
            score,
            batch_size,
            sort_by_length,
            show_progress
        )
        return outputs[0] if input_is_string else outputs

    def correct_file(
            self,
            input_file_path: str,
            output_file_path: Optional[str] = None,
            detections: Optional[Union[str, List[List[int]]]] = None,
            search: Search = GreedySearch(),
            score: SpellingCorrectionScore = SpellingCorrectionScore(),
            batch_size: int = 16,
            sort_by_length: bool = True,
            show_progress: bool = True
    ) -> Optional[List[str]]:
        if detections is not None and isinstance(detections, str):
            detections = load_text_file(detections)
            detections = [[int(det) for det in detection.split()] for detection in detections]

        outputs = self._correct_text_raw(
            input_file_path, detections, search, score, batch_size, sort_by_length, show_progress
        )

        if output_file_path is not None:
            with open(output_file_path, "w", encoding="utf8") as out_file:
                for output in outputs:
                    out_file.write(output)
                    out_file.write("\n")
            return None
        else:
            return outputs

    def to(self, device: Union[str, int]) -> "SpellingErrorCorrector":
        self.device = torch.device(device)
        self.model.to(self.device)
        return self
