import pprint
from typing import List, Optional, Union, Any, Dict, Tuple

import torch

from nsc.api.utils import (
    ModelInfo,
    StringInputOutput,
    load_experiment,
    get_device_info,
    _APIBase,
    save_text_file, load_text_file
)
from nsc.data import DatasetVariants
from nsc.modules import inference
from nsc.tasks import graph_sed_words, graph_sed_sequence, sed_words, sed_sequence, tokenization_repair_plus
from nsc.utils import common

Detections = Union[str, List[int], List[List[int]]]


def get_available_spelling_error_detection_models() -> List[ModelInfo]:
    """
    Get available spelling error detection models

    Returns: list of spelling error detection model infos each containing the task name,
    model name and a short description

    """
    return [
        ModelInfo(
            task="sed words",
            name="gnn+",
            description="Attentional Graph Neural Network which processes language graphs with "
                        "fully connected word nodes, word features and fully connected sub-word cliques. "
                        "Predicts spelling errors on word level using the word node representations."
        ),
        ModelInfo(
            task="sed words",
            name="gnn+ neuspell",
            description="Attentional Graph Neural Network which processes language graphs with "
                        "fully connected word nodes, word features and fully connected sub-word cliques. "
                        "Predicts spelling errors on word level using the word node representations. "
                        "(pretrained without Neuspell and BEA misspellings, finetuned on Neuspell training data)"
        ),
        ModelInfo(
            task="sed words",
            name="transformer",
            description="Regular transformer processing a sequence of sub-word tokens. "
                        "Predicts spelling errors on word level "
                        "using the aggregated sub-word representations per word."
        ),
        ModelInfo(
            task="sed words",
            name="transformer neuspell",
            description="Regular transformer processing a sequence of sub-word tokens. "
                        "Predicts spelling errors on word level "
                        "using the aggregated sub-word representations per word. "
                        "(pretrained without Neuspell and BEA misspellings, finetuned on Neuspell training data)"
        ),
        ModelInfo(
            task="sed words",
            name="transformer+",
            description="Regular transformer processing a sequence of sub-word tokens. "
                        "Before predicting spelling errors, sub-word representations within a word are aggregated "
                        "and enriched with word features to obtain word representations. Predicts spelling errors on "
                        "word level using those word representations."
        ),
        ModelInfo(
            task="sed words",
            name="transformer+ neuspell",
            description="Regular transformer processing a sequence of sub-word tokens. "
                        "Before predicting spelling errors, sub-word representations within a word are aggregated "
                        "and enriched with word features to obtain word representations. Predicts spelling errors on "
                        "word level using those word representations. "
                        "(pretrained without Neuspell and BEA misspellings, finetuned on Neuspell training data)"
        ),
        ModelInfo(
            task="sed words",
            name="gnn",
            description="Attentional Graph Neural Network which processes language graphs with "
                        "fully connected word nodes and fully connected sub-word cliques. "
                        "Predicts spelling errors on word level using the word node representations."
        ),
        ModelInfo(
            task="sed words",
            name="gnn neuspell",
            description="Attentional Graph Neural Network which processes language graphs with "
                        "fully connected word nodes and fully connected sub-word cliques. "
                        "Predicts spelling errors on word level using the word node representations. "
                        "(pretrained without Neuspell and BEA misspellings, finetuned on Neuspell training data)"
        ),
        ModelInfo(
            task="sed words",
            name="whitespace correction+",
            description="Transformer based model that detects errors in sequences by first correcting the whitespaces "
                        "and then detecting spelling errors for each word in the repaired text."
        ),
        ModelInfo(
            task="sed words",
            name="whitespace correction++",
            description="Transformer based model that detects errors in sequences by first correcting the whitespaces "
                        "and then detecting spelling errors for each word in the repaired text. Different from "
                        "whitespace correction+ because this model was trained to also correct "
                        "spelling errors (it is also available in nsec)."
        )
    ]


class SpellingErrorDetector(_APIBase):
    """Spelling error detection

    Class to run spelling error detection models.

    """

    def __init__(
            self,
            model_dir: str,
            device: Union[str, int],
            **kwargs: Dict[str, Any]
    ) -> None:
        """Spelling error detection constructor.

        Do not use this explicitly.
        Use the static SpellingErrorDetector.from_pretrained() and SpellingErrorDetector.from_experiment() methods
        instead.

        Args:
            model_dir: directory of the model to load
            device: device to load the model in
        """
        logger = common.get_logger("SPELLING_ERROR_DETECTION")

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
            isinstance(task, graph_sed_words.GraphSEDWords)
            or isinstance(task, sed_words.SEDWords)
            or isinstance(task, graph_sed_sequence.GraphSEDSequence)
            or isinstance(task, sed_sequence.SEDSequence)
            or isinstance(task, tokenization_repair_plus.TokenizationRepairPlus)
        ), \
            f"expected experiment to be of type SEDWords, GraphSEDWords, " \
            f"SEDSequence, GraphSEDSequence or TokenizationRepairPlus, but got {task.__class__.__name__}"

        super().__init__(model, cfg, task, device, logger)

    @property
    def task_name(self) -> str:
        return (
            "sed sequence" if self.task.variant.cfg.type == DatasetVariants.SED_SEQUENCE
            else "sed words"
        )

    @staticmethod
    def from_pretrained(
            task: str = "sed words",
            model: str = "gnn+",
            device: Union[str, int] = "cuda",
            download_dir: Optional[str] = None,
            cache_dir: Optional[str] = None,
            force_download: bool = False
    ) -> "SpellingErrorDetector":
        assert any(model == m.name and task == m.task for m in get_available_spelling_error_detection_models()), \
            f"task {task} and model {model} do not match any of the available models:\n" \
            f"{pprint.pformat(get_available_spelling_error_detection_models())}"

        model_dir, data_dir, config_dir = SpellingErrorDetector._download(
            task,
            model,
            download_dir,
            cache_dir,
            force_download
        )

        return SpellingErrorDetector(
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
    ) -> "SpellingErrorDetector":
        return SpellingErrorDetector(
            experiment_dir,
            device,
            **{"keep_existing_env_vars": {"NSC_DATA_DIR", "NSC_CONFIG_DIR"}}
        )

    @torch.inference_mode()
    def _detect_text_raw(
            self,
            inputs: Union[str, List[str]],
            threshold: float = 0.5,
            batch_size: int = 16,
            batch_max_length_factor: Optional[float] = None,
            sort_by_length: bool = True,
            show_progress: bool = False,
            **kwargs: Any
    ) -> Tuple[List[List[int]], List[str]]:
        inference_kwargs = {
            "threshold": threshold
        }
        is_tokenization_repair_plus = isinstance(
            self.task, tokenization_repair_plus.TokenizationRepairPlus)
        if is_tokenization_repair_plus:
            inference_kwargs["output_type"] = "sed"
            inference_kwargs["no_repair"] = kwargs.get(
                "tokenization_repair_plus_no_repair", False)

        if isinstance(inputs, str):
            inputs = load_text_file(inputs)

        all_outputs = super()._run_raw(
            inputs=inputs,
            batch_size=batch_size,
            batch_max_length_factor=batch_max_length_factor,
            sort_by_length=sort_by_length,
            show_progress=show_progress,
            fix_all_uppercase=True,
            **inference_kwargs
        )

        if is_tokenization_repair_plus:
            return (
                [output["sed"] if output is not None else []
                    for output in all_outputs],
                [output["tokenization_repair"]
                    if output is not None else "" for output in all_outputs]
            )
        else:
            fill_invalid = 0 if self.task_name == "sed_sequence" else []
            return [output if output is not None else fill_invalid for output in all_outputs], inputs

    def detect_text(
            self,
            inputs: StringInputOutput,
            threshold: float = 0.5,
            batch_size: int = 16,
            batch_max_length_factor: Optional[float] = None,
            sort_by_length: bool = True,
            show_progress: bool = False,
            **kwargs: Any
    ) -> Tuple[Union[List[int], List[List[int]]], Union[str, List[str]]]:
        """

        Detect spelling errors in text.

        Args:
            inputs: text to check for errors given as a single string or a list of strings
            threshold: set detection threshold (0 < threshold < 1)
            batch_size: how many sequences to process at once
            batch_max_length_factor: sets the maximum total length of a batch to be
                batch_max_length_factor * model_max_input_length, if a model e.g. has a max input length of 512 tokens
                and batch_max_length_factor is 4 then one batch will contain as many input sequences as fit within
                512 * 4 = 2048 tokens (takes precedence over batch_size if specified)
            sort_by_length: sort the inputs by length before processing them
            show_progress: display progress bar

        Returns: tuple of detections as list of integers or list of lists of integers and output strings as
            str or list of strings

        """
        input_is_string = isinstance(inputs, str)
        assert (
            input_is_string
            or (isinstance(inputs, list) and all(isinstance(ipt, str) for ipt in inputs))
        ), f"input needs to be a string or a list of strings"

        detections, output_strings = self._detect_text_raw(
            [inputs] if input_is_string else inputs,
            threshold,
            batch_size,
            batch_max_length_factor,
            sort_by_length,
            show_progress,
            **kwargs
        )
        return (detections[0], output_strings[0]) if input_is_string else (detections, output_strings)

    def detect_file(
            self,
            input_file_path: str,
            output_file_path: Optional[str] = None,
            threshold: float = 0.5,
            batch_size: int = 16,
            batch_max_length_factor: Optional[float] = None,
            sort_by_length: bool = True,
            show_progress: bool = True,
            **kwargs: Any
    ) -> Optional[Tuple[List[List[int]], List[str]]]:
        """

        Detect spelling errors in a file.

        Args:
            input_file_path: path to an input file, which will be checked for spelling errors line by line
            output_file_path: path to an output file, where the detections will be saved line by line
            threshold: set detection threshold (0 < threshold < 1)
            batch_size: how many sequences to process at once
            batch_max_length_factor: sets the maximum total length of a batch to be
                batch_max_length_factor * model_max_input_length, if a model e.g. has a max input length of 512 tokens
                and batch_max_length_factor is 4 then one batch will contain as many input sequences as fit within
                512 * 4 = 2048 tokens (takes precedence over batch_size if specified)
            sort_by_length: sort the inputs by length before processing them
            show_progress: display progress bar

        Returns: tuple of detections as list of lists of integers and output strings as list of strings
            if output_file_path is not specified else None

        """
        outputs = self._detect_text_raw(
            input_file_path, threshold, batch_size, batch_max_length_factor, sort_by_length, show_progress, **kwargs
        )
        if output_file_path is not None:
            save_text_file(output_file_path, iter(
                inference.inference_output_to_str(output) for output in outputs))
        else:
            return outputs
