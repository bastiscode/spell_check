import os
import pprint
from dataclasses import dataclass
from typing import List, Optional, Union, Any, Dict

import torch
from torch import autocast
from tqdm import tqdm

from gnn_lib.api.utils import (
    _APIBase,
    ModelInfo,
    StringInputOutput,
    download_data,
    load_experiment,
    reorder_data,
    load_text_file,
    get_device_info,
    get_inference_dataset_and_loader, save_text_file
)
from gnn_lib.data import index, tokenization
from gnn_lib.data.utils import clean_sequence
from gnn_lib.modules import inference
from gnn_lib.tasks import graph_sec_nmt, graph_sec_words_nmt, sec_nmt, sec_words_nmt, tokenization_repair_plus
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


@dataclass
class Search:
    pass


@dataclass
class GreedySearch(Search):
    pass


@dataclass
class SampleSearch(Search):
    top_k: int = 5


@dataclass
class BestFirstSearch(Search):
    pass


@dataclass
class BeamSearch(Search):
    beam_width: int = 5


@dataclass
class SpellingCorrectionScore:
    normalize_by_length: bool = True
    alpha: float = 1.0
    mode: str = "log_likelihood"
    prefix_index: Optional[index.PrefixIndex] = None


def inference_kwargs_from_search_and_score(
        search: Search,
        score: SpellingCorrectionScore,
        tokenizer: tokenization.Tokenizer
) -> Dict[str, Any]:
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
        logger.info(f"score mode is {score.mode}, but no prefix index is given, downloading data to use "
                    f"pretrained prefix index")
        data_dir = download_data(False, logger)
        score.prefix_index = index.PrefixIndex(
            os.path.join(data_dir, "prefix_index", "merged_train_100k_prefix_index.pkl")
        )

    inference_kwargs["score_fn"] = inference.spelling_correction_score(
        mode=score.mode,
        prefix_index=score.prefix_index,
        de_tok_fn=inference.get_de_tok_fn(
            tokenizer,
            tokenizer.token_to_id(tokenization.BOS),
            tokenizer.token_to_id(tokenization.EOS)
        ),
        normalize_by_length=score.normalize_by_length,
        alpha=score.alpha
    )

    return inference_kwargs


class SpellingErrorCorrector(_APIBase):
    def __init__(
            self,
            model_dir: str,
            device: Union[str, int],
            **kwargs: Dict[str, Any]
    ) -> None:
        logger = common.get_logger("SPELLING_ERROR_CORRECTION")

        if device != "cpu" and not torch.cuda.is_available():
            logger.info(f"could not find a GPU, using CPU as fallback option")
            device = "cpu"

        device = torch.device(device)
        logger.info(f"running spelling error correction on device {get_device_info(device)}")

        cfg, task, model = load_experiment(
            model_dir,
            device,
            kwargs.get("override_env_vars"),
            kwargs.get("keep_existing_env_vars", False)
        )

        assert (
                isinstance(task, graph_sec_nmt.GraphSECNMT)
                or isinstance(task, sec_nmt.SECNMT)
                or isinstance(task, graph_sec_words_nmt.GraphSECWordsNMT)
                or isinstance(task, sec_words_nmt.SECWordsNMT)
        ), f"expected experiment to be of type SECNMT or SECWordsNMT, but got {type(task)}"

        self.max_length = model.cfg.max_length

        super().__init__(model, cfg, task, device, logger)

    @property
    def task_name(self) -> str:
        return "sec"

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

        model_dir, data_dir, config_dir = super()._download("sec", model, cache_dir, force_download)

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
        if isinstance(inputs, str):
            inputs = load_text_file(inputs)

        inputs = [clean_sequence(ipt) for ipt in inputs]

        inference_kwargs = inference_kwargs_from_search_and_score(search, score, self.model.output_tokenizer)
        if isinstance(self.task, tokenization_repair_plus.TokenizationRepairPlus):
            inference_kwargs["output_type"] = "sec"
            inference_kwargs["no_repair"] = True

        dataset, loader = get_inference_dataset_and_loader(
            inputs,
            task=self.task,
            max_length=self.max_length,
            sort_by_length=sort_by_length,
            batch_size=batch_size,
            **inference_kwargs
        )

        if (detections is not None and (
                isinstance(self.task, sec_words_nmt.SECWordsNMT)
                or isinstance(self.task, graph_sec_words_nmt.GraphSECWordsNMT)
        )):
            # prepare detections based on inference infos from inference dataset
            assert (
                    len(detections) == len(inputs)
                    and all(len(det) == len(ipt.split()) for det, ipt in zip(detections, inputs))
            ), "expected one detection for every word in every input sequence"
            new_detections = []
            input_idx = -1
            for info in dataset.sample_infos:
                if info.window_idx == 0:
                    input_idx += 1
                sequence = inputs[input_idx]
                detection = detections[input_idx]
                num_words_before_context = len(sequence[:info.ctx_start].split())
                num_words_until_context_end = len(sequence[:info.ctx_end].split())
                assert (
                        num_words_before_context + len(sequence[info.ctx_start:info.ctx_end].split())
                        == num_words_until_context_end
                ), "when using detections for spelling error correction, too long sequences must be split between " \
                   "and not within whitespace separated words"
                new_detections.append(detection[num_words_before_context:num_words_until_context_end])

            assert input_idx == len(inputs) - 1
            detections = new_detections
        else:
            detections = None

        pbar = tqdm(
            loader,
            total=dataset.char_length(),
            ascii=True,
            leave=False,
            disable=not show_progress,
            unit="char"
        )

        all_outputs = []
        for i, (batch, infos, indices) in enumerate(pbar):
            if detections is not None:
                batch_detections = [detections[idx] for idx in indices]
                inference_kwargs.update({
                    "detections": batch_detections
                })

            batch_length = sum(info.ctx_end - info.ctx_start for info in infos)
            pbar.set_description(
                f"[Batch {i + 1}] Correcting spelling errors of {len(batch):,} sequences "
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
        all_outputs = self.task.postprocess_inference_outputs(
            inputs, dataset.sample_infos, all_outputs, **inference_kwargs
        )
        return [inference.inference_output_to_str(output) for output in all_outputs]

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
                or (isinstance(inputs, list) and all(isinstance(ipt, str) for ipt in inputs))
        ), f"input needs to be a string or a list of strings"

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
            save_text_file(output_file_path, outputs)
        else:
            return outputs
