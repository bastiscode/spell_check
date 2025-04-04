import os
import pprint
from dataclasses import dataclass
from typing import List, Optional, Union, Any, Dict

import torch
from torch import autocast
from tqdm import tqdm

from nsc import models
from nsc.api.utils import (
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
from nsc.data import index, tokenization
from nsc.data.utils import clean_sequence
from nsc.modules import inference
from nsc.tasks import graph_sec_nmt, graph_sec_words_nmt, sec_nmt, sec_words_nmt, tokenization_repair_plus
from nsc.utils import common

Detections = Union[List[int], List[List[int]]]


def get_available_spelling_error_correction_models() -> List[ModelInfo]:
    """
    Get available spelling error correction models

    Returns: list of spelling error correction model infos each containing the task name,
    model name and a short description

    """
    return [
        ModelInfo(
            task="sec",
            name="transformer words nmt",
            description="Transformer model that corrects sequences by translating each word individually "
                        "from misspelled to correct."
        ),
        ModelInfo(
            task="sec",
            name="transformer words nmt neuspell",
            description="Transformer model that corrects sequences by translating each word individually "
                        "from misspelled to correct. "
                        "(pretrained without Neuspell and BEA misspellings, finetuned on Neuspell training data)"
        ),
        ModelInfo(
            task="sec",
            name="whitespace correction++",
            description="Transformer based model that corrects sequences by first correcting the whitespaces, then "
                        "detecting spelling errors for each word in the repaired text and then translating "
                        "every detected misspelled word to its corrected version."
        ),
        ModelInfo(
            task="sec",
            name="transformer nmt",
            description="Transformer model that translates a sequence with spelling errors into a "
                        "sequence without spelling errors."
        ),
        ModelInfo(
            task="sec",
            name="transformer nmt neuspell",
            description="Transformer model that translates a sequence with spelling errors into a "
                        "sequence without spelling errors. "
                        "(pretrained without Neuspell and BEA misspellings, finetuned on Neuspell training data)"
        ),
        ModelInfo(
            task="sec",
            name="transformer with whitespace correction nmt",
            description="Transformer model that translates a sequence with spelling and whitespace errors "
                        "into a sequence without spelling and whitespace errors. Different from "
                        "transformer nmt because this model tokenizes into characters and "
                        "was trained on text with spelling and whitespaces errors, "
                        "whereas transformer nmt tokenizes into sub-words and "
                        "was trained only on text with spelling errors."
        )
    ]


@dataclass
class Search:
    """

    Base class for all search methods.

    """
    pass


@dataclass
class GreedySearch(Search):
    """

    Greedy search: Always go to the highest scoring next node.

    """
    pass


@dataclass
class SampleSearch(Search):
    """

    Sample search: Go to a random node of the top_k highest scoring next nodes.

    """
    top_k: int = 5


@dataclass
class BestFirstSearch(Search):
    """

    Best first search: Always expand the highest scoring path out of all paths encountered so far.

    """
    pass


@dataclass
class BeamSearch(Search):
    """

    Beam search: Maintain and expand the top beam_width paths.

    """
    beam_width: int = 5


@dataclass
class Score:
    """

    Determines how paths during/after decoding are scored/rescored.

    The default mode "log_likelihood" is to score a candidate using the
    token sequence log likelihood given as the sum of all token log probabilities.
    For some search methods we optionally support normalization by the token sequence length
    (so shorter paths are not preferred):

        score = sum(log_probabilities) / (len(log_probabilities)^alpha)

    Alpha here can be used to state a preference for shorter or longer sequences, if alpha > 1 longer sequences are
    preferred, if alpha < 1 shorter sequences are preferred, and if alpha = 0 the score equals the sum of the
    log probabilities.

    Other supported modes are "dictionary" and "diff_from_input". They only allow
    candidates that either contain dictionary words only or differ from the input text.
    For "dictionary" mode a prefix index must be specified, so that we are able to check if a decoded text is
    a prefix of or equal to a dictionary word.

    """
    normalize_by_length: bool = True
    alpha: float = 1.0
    mode: str = "log_likelihood"
    prefix_index: Optional[index.PrefixIndex] = None


def inference_kwargs_from_search_and_score(
        search: Search,
        score: Score,
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
        raise RuntimeError(
            f"unknown search specification {search.__class__.__name__}")

    if score.mode == "dictionary" and score.prefix_index is None:
        logger = common.get_logger("DOWNLOAD")
        logger.info(f"score mode is {score.mode}, but no prefix index is given, downloading data to use "
                    f"our precomputed prefix index")
        data_dir = download_data(False, logger)
        score.prefix_index = index.PrefixIndex(
            os.path.join(data_dir, "prefix_index",
                         "merged_train_100k_prefix_index.pkl")
        )

    inference_kwargs["normalize_by_length"] = score.normalize_by_length
    inference_kwargs["alpha"] = score.alpha
    if score.mode != "log_likelihood":
        inference_kwargs["re_score_fn"] = inference.spelling_correction_score(
            mode=score.mode,
            prefix_index=score.prefix_index,
            de_tok_fn=inference.get_de_tok_fn(
                tokenizer,
                tokenizer.token_to_id(tokenization.BOS),
                tokenizer.token_to_id(tokenization.EOS)
            ),
            eos_token_id=tokenizer.token_to_id(tokenization.EOS),
            unk_token_id=tokenizer.token_to_id(tokenization.UNK)
        )

    return inference_kwargs


class SpellingErrorCorrector(_APIBase):
    """Spelling error correction

    Class to run spelling error correction models.

    """

    def __init__(
            self,
            model_dir: str,
            device: Union[str, int],
            **kwargs: Dict[str, Any]
    ) -> None:
        """Spelling error correction constructor.

        Do not use this explicitly.
        Use the static SpellingErrorCorrector.from_pretrained() and SpellingErrorCorrector.from_experiment() methods
        instead.

        Args:
            model_dir: directory of the model to load
            device: device to load the model in
        """
        logger = common.get_logger("SPELLING_ERROR_CORRECTION")

        if device != "cpu" and not torch.cuda.is_available():
            logger.info(f"could not find a GPU, using CPU as fallback option")
            device = "cpu"
        device = torch.device(device)

        cfg, task, model = load_experiment(
            model_dir,
            device,
            kwargs.get("override_env_vars"),
            kwargs.get("keep_existing_env_vars")
        )

        assert (
            isinstance(task, graph_sec_nmt.GraphSECNMT)
            or isinstance(task, sec_nmt.SECNMT)
            or isinstance(task, graph_sec_words_nmt.GraphSECWordsNMT)
            or isinstance(task, sec_words_nmt.SECWordsNMT)
            or (isinstance(task, tokenization_repair_plus.TokenizationRepairPlus)
                and model.cfg.output_type == "tokenization_repair_plus_sed_plus_sec")
        ), f"expected experiment to be of type SECNMT, GraphSECNMT, SECWordsNMT, GraphSECWordsNMT or " \
           f"TokenizationRepairPlus (with sec output type), but got {task.__class__.__name__}"

        super().__init__(model, cfg, task, device, logger)

    @property
    def task_name(self) -> str:
        return "sec"

    @staticmethod
    def from_pretrained(
            task: str = "sec",
            model: str = "transformer words nmt",
            device: Union[str, int] = "cuda",
            download_dir: Optional[str] = None,
            cache_dir: Optional[str] = None,
            force_download: bool = False
    ) -> "SpellingErrorCorrector":
        assert any(model == m.name for m in get_available_spelling_error_correction_models()), \
            f"model {model} does not match any of the available models:\n" \
            f"{pprint.pformat(get_available_spelling_error_correction_models())}"

        model_dir, data_dir, config_dir = SpellingErrorCorrector._download(
            "sec",
            model,
            download_dir,
            cache_dir,
            force_download
        )

        return SpellingErrorCorrector(
            model_dir,
            device,
            **{
                "override_env_vars": {
                    "NSC_DATA_DIR": data_dir,
                    "NSC_CONFIG_DIR": config_dir
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
            **{"keep_existing_env_vars": {"NSC_DATA_DIR", "NSC_CONFIG_DIR"}}
        )

    def _get_output_tokenizer(self) -> tokenization.Tokenizer:
        if isinstance(self.model, models.ModelForTokenizationRepairPlus):
            return self.model.sec_tokenizer
        elif isinstance(self.model, models.ModelForMultiNode2Seq):
            return self.model.output_tokenizers["token"]
        else:
            return self.model.output_tokenizer

    @torch.inference_mode()
    def _correct_text_raw(
            self,
            inputs: Union[str, List[str]],
            detections: Optional[List[List[int]]] = None,
            search: Search = GreedySearch(),
            score: Score = Score(),
            batch_size: int = 16,
            batch_max_length_factor: Optional[float] = None,
            sort_by_length: bool = True,
            show_progress: bool = False,
            return_raw: bool = False,
            **kwargs: Any
    ) -> List[Any]:
        if isinstance(inputs, str):
            inputs = load_text_file(inputs)

        inputs = [clean_sequence(ipt) for ipt in inputs]
        num_inputs = len(inputs)
        invalid_inputs = set(i for i in range(num_inputs) if inputs[i] == "")
        inputs = [inputs[i]
                  for i in range(num_inputs) if i not in invalid_inputs]

        inference_kwargs = inference_kwargs_from_search_and_score(
            search, score, self._get_output_tokenizer())
        is_tokenization_repair_plus = isinstance(
            self.task, tokenization_repair_plus.TokenizationRepairPlus)
        if is_tokenization_repair_plus:
            # ignore detections with tr+, because tr+ has a detection capabilities in itself
            detections = None
            inference_kwargs["output_type"] = "sec"
            inference_kwargs["no_repair"] = kwargs.get(
                "tokenization_repair_plus_no_repair", False)
            inference_kwargs["no_detect"] = kwargs.get(
                "tokenization_repair_plus_no_detect", False)
            inference_kwargs["threshold"] = kwargs.get(
                "tokenization_repair_plus_threshold", 0.5)

        if detections is not None:
            detections = [detections[i]
                          for i in range(num_inputs) if i not in invalid_inputs]
            assert (
                len(detections) == len(inputs)
                and all(len(det) == len(ipt.split()) for det, ipt in zip(detections, inputs))
            ), "expected one detection for every word in every input sequence"

        num_workers = 0 if len(inputs) <= 16 else min(
            4, len(os.sched_getaffinity(0)))
        dataset, loader = get_inference_dataset_and_loader(
            sequences=inputs,
            task=self.task,
            max_length=self.max_length,
            sort_by_length=sort_by_length,
            batch_size=batch_size,
            batch_max_length_factor=batch_max_length_factor,
            num_workers=num_workers,
            **inference_kwargs
        )

        if detections is not None:
            # prepare detections based on inference infos from inference dataset
            new_detections = []
            input_idx = -1
            for info in dataset.sample_infos:
                if info.window_idx == 0:
                    input_idx += 1
                sequence = inputs[input_idx]
                detection = detections[input_idx]
                num_words_before_context = len(
                    sequence[:info.ctx_start].split())
                num_words_until_context_end = len(
                    sequence[:info.ctx_end].split())
                assert (
                    num_words_before_context +
                    len(sequence[info.ctx_start:info.ctx_end].split())
                    == num_words_until_context_end
                ), "when using detections for spelling error correction, too long sequences must be split between " \
                   "and not within whitespace separated words"
                new_detections.append(
                    detection[num_words_before_context:num_words_until_context_end])

            assert input_idx == len(inputs) - 1
            detections = new_detections

        pbar = tqdm(
            loader,
            total=dataset.byte_length(),
            ascii=True,
            leave=False,
            disable=not show_progress,
            unit="B",
            unit_scale=True,
            unit_divisor=1000
        )

        all_outputs = []
        for i, (batch, infos, indices) in enumerate(pbar):
            batch_strings = [str(dataset.samples[idx]) for idx in indices]
            batch_bytes = sum(len(s.encode("utf8")) for s in batch_strings)

            inference_kwargs.update({
                "input_strings": batch_strings
            })
            if detections is not None:
                batch_detections = [detections[idx] for idx in indices]
                inference_kwargs.update({
                    "detections": batch_detections
                })

            pbar.set_description(
                f"[Batch {i + 1}] Running {self.task_name} on {len(indices):,} sequences ({batch_bytes / 1000:,.1f}kB)"
            )

            # this is a slight hack for now, because fp32 on cpu throws an error even when enabled=False
            if self.mixed_precision_enabled:
                with autocast(
                        device_type=self.device.type,
                        dtype=self._mixed_precision_dtype,
                        enabled=self.mixed_precision_enabled
                ):
                    outputs = self.task.inference(
                        self.model, batch, **inference_kwargs)
            else:
                outputs = self.task.inference(
                    self.model, batch, **inference_kwargs)

            all_outputs.extend(outputs)
            pbar.update(batch_bytes)

        pbar.close()
        all_outputs = reorder_data(all_outputs, dataset.indices)
        all_outputs = self.task.postprocess_inference_outputs(
            inputs, dataset.sample_infos, all_outputs, **inference_kwargs
        )
        if is_tokenization_repair_plus:
            all_outputs = [output["sec"] for output in all_outputs]

        all_outputs_with_invalid = []
        output_idx = 0
        for i in range(num_inputs):
            if i in invalid_inputs:
                all_outputs_with_invalid.append(None if return_raw else "")
            else:
                all_outputs_with_invalid.append(
                    all_outputs[output_idx] if return_raw
                    else inference.inference_output_to_str(all_outputs[output_idx])
                )
                output_idx += 1
        assert output_idx == len(all_outputs)
        return all_outputs_with_invalid

    def correct_text(
            self,
            inputs: StringInputOutput,
            detections: Optional[Detections] = None,
            search: Search = GreedySearch(),
            score: Score = Score(),
            batch_size: int = 16,
            batch_max_length_factor: Optional[float] = None,
            sort_by_length: bool = True,
            show_progress: bool = False,
            **kwargs: Any
    ) -> StringInputOutput:
        """

        Correct spelling errors in text.

        Args:
            inputs: text to correct given as a single string or a list of strings
            detections: spelling error detections (from a SpellingErrorDetector) to guide the correction, if
                inputs is a single str, detections must be a list of integers, otherwise if inputs is a list of strings,
                detections should be a list of lists of integers
            search: Search instance to determine the search method to use for decoding
            score: Score instance to determine how to score search paths during decoding
            batch_size: how many sequences to process at once
            batch_max_length_factor: sets the maximum total length of a batch to be
                batch_max_length_factor * model_max_input_length, if a model e.g. has a max input length of 512 tokens
                and batch_max_length_factor is 4 then one batch will contain as many input sequences as fit within
                512 * 4 = 2048 tokens (takes precedence over batch_size if specified)
            sort_by_length: sort the inputs by length before processing them
            show_progress: display progress bar

        Returns: corrected text as string or list of strings

        """
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
            batch_max_length_factor,
            sort_by_length,
            show_progress,
            **kwargs
        )
        return outputs[0] if input_is_string else outputs

    def correct_file(
            self,
            input_file_path: str,
            output_file_path: Optional[str] = None,
            detections: Optional[Union[str, List[List[int]]]] = None,
            search: Search = GreedySearch(),
            score: Score = Score(),
            batch_size: int = 16,
            batch_max_length_factor: Optional[float] = None,
            sort_by_length: bool = True,
            show_progress: bool = True,
            **kwargs: Any
    ) -> Optional[List[str]]:
        """

        Correct spelling errors in a file.

        Args:
            input_file_path: path to an input file, which will be corrected line by line
            output_file_path: path to an output file, where corrected text will be saved line by line
            detections: spelling error detections (from a SpellingErrorDetector) to guide the correction, can either
                be a path to a file containing detections or a list of lists of integers
            search: Search instance to determine the search method to use for decoding
            score:  Score instance to determine how to score search paths during decoding
            batch_size: how many sequences to process at once
            batch_max_length_factor: sets the maximum total length of a batch to be
                batch_max_length_factor * model_max_input_length, if a model e.g. has a max input length of 512 tokens
                and batch_max_length_factor is 4 then one batch will contain as many input sequences as fit within
                512 * 4 = 2048 tokens (takes precedence over batch_size if specified)
            sort_by_length: sort the inputs by length before processing them
            show_progress: display progress bar

        Returns: corrected file as list of strings if output_file_path is not specified else None

        """
        if detections is not None and isinstance(detections, str):
            detections = load_text_file(detections)
            detections = [[int(det) for det in detection.split()]
                          for detection in detections]

        outputs = self._correct_text_raw(
            input_file_path,
            detections,
            search,
            score,
            batch_size,
            batch_max_length_factor,
            sort_by_length,
            show_progress,
            **kwargs
        )

        if output_file_path is not None:
            save_text_file(output_file_path, outputs)
        else:
            return outputs

    def suggest(
            self,
            text: str,
            n: int,
            detections: Optional[List[int]] = None,
            **kwargs: Any
    ) -> List[str]:
        """

        Generate suggestions for possible corrections of a text.

        Note that if you use this with a model that translates each word individually (e.g. transformer word),
        the i-th suggestion string is built by joining the i-th suggestions for all words (they do not necessarily form
        a useful sentence together). To obtain n suggestions for each word in the text you can simply split every
        suggestion string on whitespaces.

        Args:
            text: text to generate suggestions for
            n: number of suggestions to generate
            detections: optional detections to exclude some words

        Returns: suggestions as list of strings

        """
        suggestions = self._correct_text_raw(
            inputs=[text],
            detections=[detections] if detections is not None else None,
            search=BeamSearch(n),
            batch_size=1,
            return_raw=True,
            **kwargs
        )[0]
        if suggestions is None:
            return [""]
        else:
            return suggestions
