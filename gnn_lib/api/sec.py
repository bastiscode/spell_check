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
    get_gpu_info, get_string_dataset_and_loader, reorder_data
)
from gnn_lib.data import index
from gnn_lib.modules import inference
from gnn_lib.tasks import sec_nmt, sec_words_nmt
from gnn_lib.utils import common

__all__ = ["get_available_spelling_error_correction_models", "SpellingErrorCorrector"]

Detections = Union[str, List[int], List[List[int]]]


def get_available_spelling_error_correction_models() -> List[ModelInfo]:
    return [
        ModelInfo(
            task="sec",
            name="transformer_nmt",
            description="Transformer model that translates a sequence with spelling errors into a "
                        "sequence without spelling errors"
        ),
        ModelInfo(
            task="sec",
            name="transformer_words_nmt",
            description="Transformer model that corrects sequences by correcting each word individually using a shared "
                        "decoder over all words that translates from misspelled words to correct words"
        )
    ]


class SpellingErrorCorrector:
    def __init__(
            self,
            model_dir: str,
            use_gpu: bool = True,
            **kwargs: Dict[str, Any]
    ) -> None:
        self.logger = common.get_logger("SPELLING_ERROR_CORRECTION")

        if use_gpu:
            if not torch.cuda.is_available():
                self.logger.info(f"could not find a GPU, using CPU {get_cpu_info()} as fallback option")
                device = "cpu"
            else:
                self.logger.info(f"running tokenization repair on GPU {get_gpu_info()}")
                device = "cuda"
        else:
            self.logger.info(f"running tokenization repair on CPU {get_cpu_info()}")
            device = "cpu"

        self.device = torch.device(device)

        _, self.task, self.model = load_experiment(
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

        self.max_length = (
            self.model.embedding.node_embedding.pos_emb["token"].max_len
            if self.model.embedding.node_embedding.pos_emb is not None
            else float("inf")
        )

        self.prefix_index = None
        self.loaded_prefix_index = None

    @staticmethod
    def from_pretrained(
            model: str,
            use_gpu: bool = True,
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
            use_gpu,
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
            use_gpu: bool = True
    ) -> "SpellingErrorCorrector":
        return SpellingErrorCorrector(
            experiment_dir,
            use_gpu,
            ** {
                "keep_existing_env_vars": True
            }
        )

    @torch.inference_mode()
    def _correct_text_raw(
            self,
            inputs: Union[str, List[str]],
            detections: Optional[Detections] = None,
            prefix_index: Optional[str] = None,
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

        inference_kwargs = {
            "inference_mode": "greedy",
            "beam_width": 5,
            "best_first_top_k": 1,
            "sample_top_k": 5
        }
        if prefix_index is not None:
            # only load a new prefix index if it is a different one than already loaded
            if prefix_index != self.loaded_prefix_index:
                self.prefix_index = index.PrefixIndex(prefix_index)
                self.loaded_prefix_index = prefix_index
            inference_kwargs["prefix_index"] = self.prefix_index

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
            prefix_index: Optional[str] = None,
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
            detections,
            prefix_index,
            batch_size,
            sort_by_length,
            show_progress
        )
        return outputs[0] if input_is_string else outputs

    def correct_file(
            self,
            input_file_path: str,
            output_file_path: Optional[str] = None,
            detections: Optional[Detections] = None,
            prefix_index: Optional[str] = None,
            batch_size: int = 16,
            sort_by_length: bool = True,
            show_progress: bool = True
    ) -> Optional[List[str]]:
        outputs = self._correct_text_raw(
            input_file_path, detections, prefix_index, batch_size, sort_by_length, show_progress
        )
        if output_file_path is not None:
            with open(output_file_path, "w", encoding="utf8") as out_file:
                for output in outputs:
                    out_file.write(output)
                    out_file.write("\n")
            return None
        else:
            return outputs
