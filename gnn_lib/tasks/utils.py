import copy
from typing import List, Union, Optional, Set, Dict, Any, Tuple

import dgl
import numpy as np
import torch
from torch import nn

from gnn_lib.data import utils as data_utils
from gnn_lib.data.utils import InferenceInfo
from gnn_lib.modules.utils import tensor_to_python, split
from gnn_lib.utils import DataInput, tokenization_repair

SAMPLE_SEQUENCE = \
    "But I must explain to you how all this mistaken idea of denouncing pleasure and praising pain was " \
    "born and I will give you a complete account of the system, and expound the actual teachings of the great" \
    " explorer of the truth, the master-builder of human happiness. No one rejects, dislikes, or avoids pleas" \
    "ure itself, because it is pleasure, but because those who do not know how to pursue pleasure rationally" \
    " encounter consequences that are extremely painful."


def class_predictions(logits: torch.Tensor, threshold: float = 0.5, temperature: float = 1.0) -> torch.Tensor:
    if logits.shape[-1] > 2:
        # multi-class classification
        return torch.argmax(logits, dim=-1).long()
    else:
        # binary classification
        softmax = torch.softmax(logits / temperature, dim=-1)
        return (softmax[..., -1] >= threshold).long()


def class_voting(
        logits: List[Union[List, torch.Tensor]],
        method: str,
        thresholds: Optional[List[float]] = None,
        temperatures: Optional[List[float]] = None
) -> torch.Tensor:
    if temperatures is None:
        temperatures = [1.0] * len(logits)
    else:
        assert len(temperatures) == len(logits)
    if thresholds is None:
        thresholds = [0.5] * len(logits)
    else:
        assert len(thresholds) == len(logits)

    logits = [torch.tensor(l, dtype=torch.float) if not isinstance(l, torch.Tensor) else l for l in logits]

    if method == "hard":  # predict the class with the most individual predictions
        predictions = torch.stack([
            class_predictions(l, threshold=thres, temperature=temp)
            for l, thres, temp in zip(logits, thresholds, temperatures)
        ], dim=-1)
        most_freq_classes = torch.mode(predictions, dim=-1).values
        return most_freq_classes
    elif method == "soft":  # predict the class with the highest total probability (requires good calibrated models)
        soft_predictions = [torch.softmax(l / t, dim=-1) for l, t in zip(logits, temperatures)]
        summed_predictions = torch.stack(soft_predictions).sum(dim=0)
        return torch.argmax(summed_predictions, dim=-1).long()
    else:
        raise ValueError(f"Unknown method {method}, should be one of [hard, soft]")


class EMA:
    def __init__(self, model: nn.Module, ema_factor: float) -> None:
        self.model = model
        self.ema_model: nn.Module = copy.deepcopy(model)
        for param in self.ema_model.parameters():
            param.detach_()
        self.ema_model = self.ema_model.eval()

        self.ema_factor = ema_factor
        self.one_minus_ema_factor = 1 - ema_factor

    def update(self, overwrite: bool = False):
        for ema_param, model_param in zip(self.ema_model.parameters(), self.model.parameters()):
            if overwrite:
                ema_param.data = model_param.data
            else:
                ema_param.mul_(self.ema_factor).add_(model_param.data, alpha=self.one_minus_ema_factor)

        for ema_buffer, model_buffer in zip(self.ema_model.buffers(), self.model.buffers()):
            ema_buffer.data = model_buffer.data


def get_token_ids_from_graphs(graph: dgl.DGLHeteroGraph) -> List[List[int]]:
    return split(
        tensor_to_python(graph.nodes["token"].data["token_id"], force_list=True),
        tensor_to_python(graph.batch_num_nodes("token"), force_list=True)
    )


def get_batch_size_from_data(data: DataInput) -> int:
    if isinstance(data, dgl.DGLHeteroGraph):
        return data.batch_size
    else:
        return len(data)


def get_unused_parameters(model: nn.Module, **inputs: Any) -> Set[str]:
    def _sum(item: Union[torch.Tensor, List, Dict]) -> Union[torch.Tensor, List, Dict]:
        if isinstance(item, torch.Tensor):
            return item.sum()
        elif isinstance(item, list):
            return sum([_sum(i) for i in item])
        elif isinstance(item, dict):
            return sum(_sum(v) for v in item.values())
        else:
            raise RuntimeError("expected model output to be any nesting of dicts, lists and tensors, "
                               f"but got {type(item)}")

    outputs, _ = model(**inputs)
    loss = _sum(outputs)

    loss.backward()

    unused_parameters = set()
    for name, p in model.named_parameters():
        if p.grad is None and p.requires_grad:
            unused_parameters.add(name)

    return unused_parameters


def get_word_windows(
        sample: data_utils.Sample, max_length: int, context_length: int
) -> List[Tuple[int, int, int, int]]:
    sequence = str(sample)
    words = sequence.split()
    word_lengths = [len(w) for w in words]

    window_length = max_length - 2 * context_length

    num_word_ws_tokens = [0] * len(words)
    word_ws_idx = 0
    for word_tokens, word in zip(sample.tokens, sample.doc):
        num_word_ws_tokens[word_ws_idx] += len(word_tokens)
        if word.whitespace_ == " ":
            word_ws_idx += 1
    assert word_ws_idx == len(words) - 1
    assert sum(num_word_ws_tokens) == sum(len(t) for t in sample.tokens)
    assert all(num_tokens <= window_length for num_tokens in num_word_ws_tokens), \
        f"a single word in the input sequence {sequence} has more tokens " \
        f"than the max window length of {window_length} tokens"

    word_window_start = 0
    windows = []
    while word_window_start < len(words):
        word_window_end = word_window_start + (np.cumsum(num_word_ws_tokens[word_window_start:]) <= window_length).sum()
        assert word_window_end > word_window_start
        word_context_start = word_window_start - (
                np.cumsum(num_word_ws_tokens[:word_window_start][::-1]) <= context_length
        ).sum()
        word_context_end = word_window_end + (np.cumsum(num_word_ws_tokens[word_window_end:]) <= context_length).sum()

        window = (
            max(0, word_context_start - 1 + sum(word_lengths[:word_context_start])),  # ctx start
            word_context_end - 1 + sum(word_lengths[:word_context_end]),  # ctx end
            max(0, word_window_start - 1 + sum(word_lengths[:word_window_start])),  # window start
            word_window_end - 1 + sum(word_lengths[:word_window_end])  # window end
        )
        windows.append(window)

        word_window_start = word_window_end

    return windows


def get_character_windows(
        sample: data_utils.Sample, max_length: int, context_length: int) -> List[Tuple[int, int, int, int]]:
    sequence = str(sample)
    window_length = max_length - 2 * context_length
    windows = []
    for window_start in range(0, len(sequence), window_length):
        windows.append((
            max(0, window_start - context_length),  # ctx start
            min(len(sequence), window_start + window_length + context_length),  # ctx end
            window_start,  # window start
            min(len(sequence), window_start + window_length)  # window end
        ))
    return windows


def get_byte_windows(
        sample: data_utils.Sample, max_length: int, context_length: int) -> List[Tuple[int, int, int, int]]:
    sequence = str(sample)
    window_length = max_length - 2 * context_length
    byte_lengths = [len(char.encode("utf8")) for char in sequence]
    windows = []
    byte_window_start = 0
    while byte_window_start < len(byte_lengths):
        byte_window_end = byte_window_start + (np.cumsum(byte_lengths[byte_window_start:]) <= window_length).sum()
        assert byte_window_end > byte_window_start
        byte_context_start = byte_window_start - (
                np.cumsum(byte_lengths[:byte_window_start][::-1]) <= context_length
        )
        byte_context_end = byte_window_end + (np.cumsum(byte_lengths[byte_window_end:]) <= context_length).sum()

        windows.append((
            byte_context_start,  # ctx start
            byte_context_end,  # ctx end
            byte_window_start,  # window start
            byte_window_end  # window end
        ))

        byte_window_start = byte_window_end

    return windows


def merge_tokenization_repair_outputs(
        sequence: str,
        infos: List[InferenceInfo],
        predictions: List[str],
        return_match_indices: bool = False
) -> Union[str, Tuple[str, List[Tuple[int, int]]]]:
    merged_prediction = ""
    matches = []
    for prediction, info in zip(predictions, infos):
        left_context = sequence[info.ctx_start:info.window_start]
        window = sequence[info.window_start:info.window_end]
        right_context = sequence[info.window_end:info.ctx_end]
        match_start, match_end = tokenization_repair.match_string_ignoring_space(
            prediction,
            left_context,
            window,
            right_context
        )
        merged_prediction += prediction[match_start:match_end]
        matches.append((match_start, match_end))
    assert merged_prediction.replace(" ", "") == sequence.replace(" ", "")
    if return_match_indices:
        return merged_prediction, matches
    else:
        return merged_prediction


def merge_sed_sequence_outputs(
        predictions: List[int]
) -> int:
    assert all(p in {0, 1} for p in predictions)
    return int(any(p for p in predictions))


def align_word_prediction_with_sequence(
        sequence: str,
        context: Tuple[int, int],
        window: Tuple[int, int],
        prediction: List[Any]
) -> Tuple[int, int, int, int]:
    ctx_start, ctx_end = context
    window_start, window_end = window
    assert len(sequence[ctx_start:ctx_end].split()) == len(prediction)
    num_words_up_to_window_start = len(sequence[:window_start].strip().split())
    num_words_up_to_window_end = len(sequence[:window_end].strip().split())
    num_words_in_window = len(sequence[window_start:window_end].strip().split())
    num_context_words_up_to_window_start = len(sequence[ctx_start:window_start].strip().split())
    num_context_words_up_to_window_end = len(sequence[ctx_start:window_end].strip().split())

    assert num_words_up_to_window_start + num_words_in_window in {num_words_up_to_window_end,
                                                                  num_words_up_to_window_end + 1}
    assert num_context_words_up_to_window_start + num_words_in_window in {num_context_words_up_to_window_end,
                                                                          num_context_words_up_to_window_end + 1}

    merged_prediction_start = (
        num_words_up_to_window_start
        if num_words_up_to_window_start + num_words_in_window == num_words_up_to_window_end
        else num_words_up_to_window_start - 1
    )
    merged_prediction_end = merged_prediction_start + num_words_in_window
    prediction_start = (
        num_context_words_up_to_window_start
        if num_context_words_up_to_window_start + num_words_in_window == num_context_words_up_to_window_end
        else num_context_words_up_to_window_start - 1
    )
    prediction_end = prediction_start + num_words_in_window
    assert (
            prediction_start >= 0
            and merged_prediction_start >= 0
            and merged_prediction_end - merged_prediction_start == prediction_end - prediction_start
    )
    return merged_prediction_start, merged_prediction_end, prediction_start, prediction_end


def merge_sed_words_outputs(
        sequence: str,
        infos: List[InferenceInfo],
        predictions: List[List[int]]
) -> List[int]:
    assert all(p in {0, 1} for prediction in predictions for p in prediction)
    merged_prediction = [-1] * len(sequence.split())
    for info, prediction in zip(infos, predictions):
        (
            merged_prediction_start, merged_prediction_end, prediction_start, prediction_end
        ) = align_word_prediction_with_sequence(
            sequence, (info.ctx_start, info.ctx_end), (info.window_start, info.window_end), prediction
        )

        for merged_idx, pred_idx in zip(
                range(merged_prediction_start, merged_prediction_end),
                range(prediction_start, prediction_end)
        ):
            if merged_prediction[merged_idx] == -1:
                merged_prediction[merged_idx] = prediction[pred_idx]
            else:
                merged_prediction[merged_idx] |= prediction[pred_idx]

    assert all(p in {0, 1} for p in merged_prediction)
    return merged_prediction


def merge_sec_nmt_outputs(
        predictions: List[List[str]]
) -> List[str]:
    min_num_predictions = min(len(prediction) for prediction in predictions)
    merged_predictions = [[] for _ in range(min_num_predictions)]
    for prediction in predictions:
        for i in range(min_num_predictions):
            merged_predictions[i].append(prediction[i])
    merged_predictions = [" ".join(s.strip() for s in predictions) for predictions in merged_predictions]
    return merged_predictions


def merge_sec_words_nmt_outputs(
        sequence: str,
        infos: List[InferenceInfo],
        predictions: List[List[List[str]]],
        ensure_unique: bool = True
) -> List[str]:
    min_num_predictions = min(len(prediction) for prediction in predictions)
    merged_predictions: List[List[str]] = [  # type: ignore
        [None for _ in range(len(sequence.split()))]
        for _ in range(min_num_predictions)
    ]
    for info, prediction in zip(infos, predictions):
        for i in range(min_num_predictions):
            predicted_words = prediction[i]
            (
                merged_prediction_start, merged_prediction_end, prediction_start, prediction_end
            ) = align_word_prediction_with_sequence(
                sequence, (info.ctx_start, info.ctx_end), (info.window_start, info.window_end), predicted_words
            )

            for merged_idx, pred_idx in zip(
                    range(merged_prediction_start, merged_prediction_end),
                    range(prediction_start, prediction_end)
            ):
                if ensure_unique:
                    assert merged_predictions[i][merged_idx] is None, \
                        "expected only on output word for each input word, but got more, this happens " \
                        "when you dont split the input sequence between words"
                merged_predictions[i][merged_idx] = predicted_words[pred_idx]

    assert all(word is not None for prediction in merged_predictions for word in prediction)
    merged_predictions = [" ".join(merged_predicted_words) for merged_predicted_words in merged_predictions]
    return merged_predictions
