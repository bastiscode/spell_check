import pprint
from typing import List, Union, Dict, Any, Optional

import torch
from torch import autocast
from tqdm import tqdm

from gnn_lib.api.utils import (
    ModelInfo,
    StringInputOutput,
    load_experiment,
    get_string_dataset_and_loader,
    reorder_data, get_device_info, _APIBase
)
from gnn_lib.modules import inference
from gnn_lib.tasks import tokenization_repair, tokenization_repair_plus
from gnn_lib.utils import common

__all__ = ["get_available_tokenization_repair_models", "TokenizationRepairer"]


def get_available_tokenization_repair_models() -> List[ModelInfo]:
    return [
        ModelInfo(
            task="tokenization_repair",
            name="transformer_eo_large",
            description="Transformer model that repairs sequences by predicting repair tokens for each character"
        )
    ]


class TokenizationRepairer(_APIBase):
    def __init__(
            self,
            model_dir: str,
            device: Union[str, int],
            **kwargs: Dict[str, Any]
    ) -> None:
        logger = common.get_logger("TOKENIZATION_REPAIR")

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
            isinstance(task, tokenization_repair.TokenizationRepair),
            isinstance(task, tokenization_repair_plus.TokenizationRepairPlus)
        ), f"expected experiment to be of type TokenizationRepair or TokenizationRepairPlus, but got {type(task)}"

        self.max_length = model.cfg.max_length

        super().__init__(model, cfg, task, device, logger)

    @property
    def task_name(self) -> str:
        return "tokenization_repair"

    @staticmethod
    def from_pretrained(
            model: str = "transformer_eo_large",
            device: Union[str, int] = "cuda",
            cache_dir: Optional[str] = None,
            force_download: bool = False
    ) -> "TokenizationRepairer":
        assert any(model == m.name for m in get_available_tokenization_repair_models()), \
            f"model {model} does not match any of the available models:\n" \
            f"{pprint.pformat(get_available_tokenization_repair_models())}"

        model_dir, data_dir, config_dir = super()._download("tokenization_repair", model, cache_dir, force_download)

        return TokenizationRepairer(
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
    ) -> "TokenizationRepairer":
        return TokenizationRepairer(
            experiment_dir,
            device,
            **{
                "keep_existing_env_vars": True
            }
        )

    @torch.inference_mode()
    def _repair_text_raw(
            self,
            inputs: Union[str, List[str]],
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

        inference_kwargs = {}
        if isinstance(self.task, tokenization_repair_plus.TokenizationRepairPlus):
            inference_kwargs["output_type"] = "tokenization_repair"

        all_outputs = []
        for i, (batch, info) in enumerate(pbar):
            batch_length = sum(info["lengths"])
            pbar.set_description(
                f"[Batch {i + 1}] Repairing tokenization in {len(batch):,} sequences "
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
        return reorder_data(all_outputs, dataset.indices)

    def repair_text(
            self,
            inputs: StringInputOutput,
            batch_size: int = 16,
            sort_by_length: bool = True,
            show_progress: bool = False
    ) -> Union[List[int], List[List[int]]]:
        input_is_string = isinstance(inputs, str)
        assert (
                input_is_string
                or (isinstance(inputs, list) and len(inputs) > 0 and isinstance(inputs[0], str))
        ), f"input needs to be a string or a non empty list of strings"

        outputs = self._repair_text_raw(
            [inputs] if input_is_string else inputs,
            batch_size,
            sort_by_length,
            show_progress
        )
        return outputs[0] if input_is_string else outputs

    def repair_file(
            self,
            input_file_path: str,
            output_file_path: Optional[str] = None,
            batch_size: int = 16,
            sort_by_length: bool = True,
            show_progress: bool = True
    ) -> Optional[Union[List[int], List[List[int]]]]:
        outputs = self._repair_text_raw(
            input_file_path, batch_size, sort_by_length, show_progress
        )
        if output_file_path is not None:
            with open(output_file_path, "w", encoding="utf8") as out_file:
                for output in outputs:
                    out_file.write(inference.inference_output_to_str(output))
                    out_file.write("\n")
            return None
        else:
            return outputs
