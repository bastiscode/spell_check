import pprint
from typing import List, Union, Dict, Any, Optional

import torch

from nsc.api.utils import (
    ModelInfo,
    StringInputOutput,
    load_experiment,
    get_device_info,
    _APIBase,
    load_text_file
)
from nsc.data.utils import clean_sequence
from nsc.modules import inference
from nsc.tasks import tokenization_repair, tokenization_repair_plus
from nsc.utils import common


def get_available_tokenization_repair_models() -> List[ModelInfo]:
    """
    Get available whitespace correction models

    Returns: list of whitespace correction model infos each containing the task name, model name and a short description

    """
    return [
        ModelInfo(
            task="whitespace correction",
            name="eo large",
            description="Large-sized Transformer model (12 layers) that repairs sequences by predicting repair tokens "
                        "for each character (ported from https://github.com/ad-freiburg/whitespace-correction)."
        ),
        ModelInfo(
            task="whitespace correction",
            name="eo medium",
            description="Medium-sized Transformer model (6 layers) that repairs sequences by predicting repair tokens "
                        "for each character (ported from https://github.com/ad-freiburg/whitespace-correction)."
        ),
        ModelInfo(
            task="whitespace correction",
            name="eo small",
            description="Small-sized Transformer model (3 layers) that repairs sequences by predicting repair tokens "
                        "for each character (ported from https://github.com/ad-freiburg/whitespace-correction)."
        ),
        ModelInfo(
            task="whitespace correction",
            name="whitespace correction+",
            description="Same as eo medium arxiv with errors, available here for completeness."
        ),
        ModelInfo(
            task="whitespace correction",
            name="whitespace correction++",
            description="Same as eo medium arxiv with errors, available here for completeness."
        )
    ]


class TokenizationRepairer(_APIBase):
    """whitespace correction

    Class to run whitespace correction models.

    """

    def __init__(
            self,
            model_dir: str,
            device: Union[str, int],
            **kwargs: Dict[str, Any]
    ) -> None:
        """whitespace correction constructor.

        Do not use this explicitly.
        Use the static TokenizationRepairer.from_pretrained() and TokenizationRepairer.from_experiment() methods
        instead.

        Args:
            model_dir: directory of the model to load
            device: device to load the model in
        """
        logger = common.get_logger("TOKENIZATION_REPAIR")

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
            isinstance(task, tokenization_repair.TokenizationRepair)
            or isinstance(task, tokenization_repair_plus.TokenizationRepairPlus)
        ), f"expected experiment to be of type TokenizationRepair or TokenizationRepairPlus, " \
           f"but got {task.__class__.__name__}"

        super().__init__(model, cfg, task, device, logger)

    @property
    def task_name(self) -> str:
        return "whitespace correction"

    @staticmethod
    def from_pretrained(
            task: str = "whitespace correction",
            model: str = "eo large arxiv with errors",
            device: Union[str, int] = "cuda",
            download_dir: Optional[str] = None,
            cache_dir: Optional[str] = None,
            force_download: bool = False
    ) -> "TokenizationRepairer":
        assert any(model == m.name for m in get_available_tokenization_repair_models()), \
            f"model {model} does not match any of the available models:\n" \
            f"{pprint.pformat(get_available_tokenization_repair_models())}"

        model_dir, data_dir, config_dir = TokenizationRepairer._download(
            "whitespace correction",
            model,
            download_dir,
            cache_dir,
            force_download
        )

        return TokenizationRepairer(
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
    ) -> "TokenizationRepairer":
        return TokenizationRepairer(
            experiment_dir,
            device,
            **{"keep_existing_env_vars": {"NSC_DATA_DIR", "NSC_CONFIG_DIR"}}
        )

    @torch.inference_mode()
    def _repair_text_raw(
            self,
            inputs: Union[str, List[str]],
            batch_size: int = 16,
            batch_max_length_factor: Optional[float] = None,
            sort_by_length: bool = True,
            show_progress: bool = False
    ) -> List[List[str]]:
        if isinstance(inputs, str):
            inputs = load_text_file(inputs)

        inputs = [clean_sequence(ipt) for ipt in inputs]

        inference_kwargs = {}
        is_tokenization_repair_plus = isinstance(
            self.task, tokenization_repair_plus.TokenizationRepairPlus)
        if is_tokenization_repair_plus:
            inference_kwargs["output_type"] = "tokenization_repair"

        all_outputs = super()._run_raw(
            inputs=inputs,
            batch_size=batch_size,
            batch_max_length_factor=batch_max_length_factor,
            sort_by_length=sort_by_length,
            show_progress=show_progress,
            **inference_kwargs
        )

        if is_tokenization_repair_plus:
            return [output["tokenization_repair"] if output is not None else "" for output in all_outputs]
        else:
            return [output if output is not None else "" for output in all_outputs]

    def repair_text(
            self,
            inputs: StringInputOutput,
            batch_size: int = 16,
            batch_max_length_factor: Optional[float] = None,
            sort_by_length: bool = True,
            show_progress: bool = False
    ) -> Union[str, List[str]]:
        """

        Repair whitespaces in text.

        Args:
            inputs: text to repair given as a single string or a list of strings
            batch_size: how many sequences to process at once
            batch_max_length_factor: sets the maximum total length of a batch to be
                batch_max_length_factor * model_max_input_length, if a model e.g. has a max input length of 512 tokens
                and batch_max_length_factor is 4 then one batch will contain as many input sequences as fit within
                512 * 4 = 2048 tokens (takes precedence over batch_size if specified)
            sort_by_length: sort the inputs by length before processing them
            show_progress: display progress bar

        Returns: repaired text as string or list of strings

        """
        input_is_string = isinstance(inputs, str)
        assert (
            input_is_string
            or (isinstance(inputs, list) and all(isinstance(ipt, str) for ipt in inputs))
        ), f"input needs to be a string or a list of strings"

        outputs = self._repair_text_raw(
            [inputs] if input_is_string else inputs,
            batch_size,
            batch_max_length_factor,
            sort_by_length,
            show_progress
        )
        return outputs[0] if input_is_string else outputs

    def repair_file(
            self,
            input_file_path: str,
            output_file_path: Optional[str] = None,
            batch_size: int = 16,
            batch_max_length_factor: Optional[float] = None,
            sort_by_length: bool = True,
            show_progress: bool = True
    ) -> Optional[List[str]]:
        """

        Repair whitespaces in a file.

        Args:
            input_file_path: path to an input file, which will be repaired line by line
            output_file_path: path to an output file, where repaired text will be saved line by line
            batch_size: how many sequences to process at once
            batch_max_length_factor: sets the maximum total length of a batch to be
                batch_max_length_factor * model_max_input_length, if a model e.g. has a max input length of 512 tokens
                and batch_max_length_factor is 4 then one batch will contain as many input sequences as fit within
                512 * 4 = 2048 tokens (takes precedence over batch_size if specified)
            sort_by_length: sort the inputs by length before processing them
            show_progress: display progress bar

        Returns: repaired file as list of strings if output_file_path is not specified else None

        """
        outputs = self._repair_text_raw(
            input_file_path, batch_size, batch_max_length_factor, sort_by_length, show_progress
        )
        if output_file_path is not None:
            with open(output_file_path, "w", encoding="utf8") as out_file:
                for output in outputs:
                    out_file.write(inference.inference_output_to_str(output))
                    out_file.write("\n")
            return None
        else:
            return outputs
