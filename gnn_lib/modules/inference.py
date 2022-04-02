import math
import queue
from typing import Callable, Tuple, Dict, List, Any, Union, Optional

import Levenshtein
import einops
import torch

from gnn_lib.data import tokenization, index, utils as data_utils
from gnn_lib.modules import utils
from gnn_lib.modules.utils import DecoderMixin


class Beam:
    def __init__(self) -> None:
        self.token_ids: List[int] = []
        self.log_prob: List[float] = []

    @staticmethod
    def from_beam(other: "Beam", log_p: float, token_id: int) -> "Beam":
        beam = Beam()
        beam.log_prob = other.log_prob + [log_p]
        beam.token_ids = other.token_ids + [token_id]
        return beam

    def is_eos(self, eos_token_id: int) -> bool:
        return self.token_ids[-1] == eos_token_id

    def __lt__(self, other: "Beam") -> bool:
        return len(self) < len(other)

    def __eq__(self, other: "Beam") -> bool:
        return self.token_ids == other.token_ids

    def __len__(self) -> int:
        return len(self.token_ids)

    def __repr__(self) -> str:
        return f"Beam(token_ids={self.token_ids}, log_prob={self.log_prob})"


BeamSelectFn = Callable[[List[Tuple[Beam, float]]], Beam]


def greedy_select_fn() -> BeamSelectFn:
    def _greedy(beams: List[Tuple[Beam, float]]) -> Beam:
        return beams[0][0]

    return _greedy


def sample_select_fn(sample_top_k: int) -> BeamSelectFn:
    def _sample(beams: List[Tuple[Beam, float]]) -> Beam:
        sample_idx = torch.randint(min(len(beams), sample_top_k), (1,)).item()
        return beams[sample_idx][0]

    return _sample


DeTokFn = Callable[[List[int]], str]
ScoreFn = Callable[
    [
        # beam with token ids and log probs
        Beam,
        # input string
        Optional[str]
    ],
    float
]


def log_likelihood_score(normalize_by_length: bool = True, alpha: float = 1.0) -> Callable[[Beam], float]:
    def score(beam: Beam, input_str: Optional[str] = None) -> float:
        s = sum(beam.log_prob)
        if normalize_by_length:
            return s / (len(beam.log_prob) ** alpha)
        else:
            return s

    return score


def spelling_correction_score(
        # if not None, must be one of
        # {log_likelihood, dictionary, dictionary_or_in_input, dictionary_or_eq_input, dictionary_or_close_input}
        mode: str = "log_likelihood",
        # prefix index, to check if a word is a prefix of a word in the dictionary
        prefix_index: Optional[index.PrefixIndex] = None,
        # de tokenization function, to get the predicted string from the predicted tokens
        de_tok_fn: Optional[DeTokFn] = None,
        # arguments for log likelihood scoring
        normalize_by_length: bool = True,
        alpha: float = 1.0,
        **kwargs: Any
) -> ScoreFn:
    log_likelihood_score_fn = log_likelihood_score(normalize_by_length, alpha)

    def _score(beam: Beam,
               input_str: Optional[str] = None) -> float:
        s = log_likelihood_score_fn(beam)
        if mode == "log_likelihood":
            return s

        assert (
                prefix_index is not None
                and input_str is not None
                and de_tok_fn is not None
        ), "for all modes other than log_likelihood you need to pass a prefix index, the original input string and " \
           "a de-tokenization function"

        pred_str = de_tok_fn(beam.token_ids)
        pred_str_split = pred_str.split()

        if len(pred_str_split) > 0:
            # get all input words
            input_words, _ = data_utils.tokenize_words_regex(input_str)
            # split current predicted word (by whitespace) further using regex
            pred_words, _ = data_utils.tokenize_words_regex(pred_str_split[-1])

            # check if current predicted word (or its lowercase version)
            # is a prefix of a dictionary word (in prefix tree)
            valid_pred = (
                    len(prefix_index.retrieve(pred_words[-1])) > 0
                    or len(prefix_index.retrieve(pred_words[-1].lower())) > 0
            )
            if mode == "dictionary":
                pass
            # check if current predicted word (or its lowercase version)
            # is a prefix of an input word
            elif mode == "dictionary_or_in_input":
                valid_pred |= (
                        any(ipt_w.startswith(pred_words[-1]) for ipt_w in input_words)
                        or any(ipt_w.startswith(pred_words[-1].lower()) for ipt_w in input_words)
                )

            # check if current predicted string is prefix of the input string
            elif mode == "dictionary_or_eq_input":
                valid_pred |= input_str.startswith(pred_str)

            # check if current predicted string has a smaller prefix edit distance to input string than a
            # given threshold
            elif mode == "dictionary_or_close_input":
                valid_pred |= (
                        Levenshtein.distance(pred_str, input_str[:len(pred_str)])
                        <
                        kwargs.get("max_prefix_edit_distance", 5)
                )

            else:
                raise RuntimeError(f"unknown spell check score mode {mode}")

            if not valid_pred:
                return s - 1_000_000

        return s

    return _score


def _sub_select(inputs: Union[torch.Tensor, Dict[str, torch.Tensor]], mask: Union[int, torch.Tensor]) -> \
        Union[torch.Tensor, Dict[str, torch.Tensor]]:
    if isinstance(inputs, torch.Tensor):
        return inputs[mask]
    elif isinstance(inputs, dict):
        return {k: v[mask] for k, v in inputs.items()}
    else:
        raise ValueError(f"expected inputs to be of type tensor or dict of tensors, but got {type(inputs)}")


@torch.inference_mode()
def token_inference(
        model: DecoderMixin,
        encoder_outputs: Dict[str, torch.Tensor],
        encoder_lengths: Dict[str, torch.Tensor],
        bos_token_id: int,
        eos_token_id: int,
        max_length: int,
        select_fn: BeamSelectFn = greedy_select_fn(),
        score_fn: ScoreFn = spelling_correction_score(),
        input_strings: Optional[List[str]] = None,
        decoder_positions: Optional[torch.Tensor] = None,
        **kwargs: Any
) -> List[List[int]]:
    model.eval()
    device = utils.device_from_model(model)

    batch_size = len(next(iter(encoder_outputs.values())))

    log_prob = torch.full((batch_size, max_length), fill_value=-1.0, device=device)
    log_prob[:, 0] = 0.0
    token_ids = torch.full((batch_size, max_length), fill_value=-1.0, dtype=torch.long, device=device)
    token_ids[:, 0] = bos_token_id
    lengths = torch.ones(batch_size, dtype=torch.long, device=device)
    if decoder_positions is not None:
        positions = torch.stack([
            torch.arange(pos, pos + max_length, device=device)
            for pos in decoder_positions
        ])
    else:
        positions = einops.repeat(torch.arange(max_length, device=device), "l -> b l", b=batch_size)

    non_eos_mask = torch.ones(batch_size, dtype=torch.bool, device=device)
    smaller_max_length_mask = torch.ones(batch_size, dtype=torch.bool, device=device)
    smaller_max_length_mask[lengths + positions[:, 0] >= max_length] = False
    indices_to_decode = non_eos_mask & smaller_max_length_mask

    while True:
        decoder_lengths = _sub_select(lengths, indices_to_decode)
        max_decoder_length = max(decoder_lengths)
        decoder_positions = _sub_select(positions, indices_to_decode)[:, :max_decoder_length]

        decoder_output = model.decode(
            decoder_inputs=_sub_select(token_ids, indices_to_decode)[:, :max_decoder_length],
            decoder_lengths=decoder_lengths,
            encoder_outputs=_sub_select(encoder_outputs, indices_to_decode),
            encoder_lengths=_sub_select(encoder_lengths, indices_to_decode),
            decoder_positions=decoder_positions
        )

        inferred_token_ids = []
        inferred_log_prob = []
        batch_indices = torch.where(indices_to_decode)[0].tolist()
        for i in range(len(decoder_output)):
            length = lengths[indices_to_decode][i]
            log_softmax_scores = torch.log_softmax(decoder_output[i, length - 1], dim=0)

            batch_idx = batch_indices[i]
            beams_and_scores = []
            min_log_prob = math.log(1 / len(log_softmax_scores))
            current_token_ids = token_ids[batch_idx, :length].tolist()
            current_log_prob = log_prob[batch_idx, :length].tolist()
            for token_id, lp in enumerate(log_softmax_scores.tolist()):
                if lp < min_log_prob:
                    continue
                beam = Beam()
                beam.token_ids = current_token_ids + [token_id]
                beam.log_prob = current_log_prob + [lp]
                beams_and_scores.append(
                    (beam, -score_fn(beam, input_strings[batch_idx] if input_strings is not None else None))
                )
            beams_and_scores = sorted(beams_and_scores, key=lambda e: e[1])
            selected_beam = select_fn(beams_and_scores)
            inferred_token_ids.append(selected_beam.token_ids[-1])
            inferred_log_prob.append(selected_beam.log_prob[-1])

        inferred_token_ids = torch.tensor(
            inferred_token_ids, dtype=torch.long, device=token_ids.device
        )
        token_ids[indices_to_decode, lengths[indices_to_decode]] = inferred_token_ids
        inferred_log_prob = torch.tensor(
            inferred_log_prob, dtype=torch.float, device=log_prob.device
        )
        log_prob[indices_to_decode, lengths[indices_to_decode]] = inferred_log_prob

        lengths[indices_to_decode] += 1

        inferred_eos_indices = torch.where(inferred_token_ids == eos_token_id)[0]
        new_eos_indices = torch.where(indices_to_decode)[0][inferred_eos_indices]
        non_eos_mask[new_eos_indices] = False

        max_length_indices = torch.where(lengths + positions[:, 0] >= max_length)[0]
        smaller_max_length_mask[max_length_indices] = False

        indices_to_decode = non_eos_mask & smaller_max_length_mask

        # all sequences are at max length or all finished with eos token
        if torch.sum(indices_to_decode) == 0:
            break

    token_ids = token_ids.tolist()

    outputs = []
    for i in range(batch_size):
        length = lengths[i]
        outputs.append(token_ids[i][:length])

    return outputs


@torch.inference_mode()
def best_first_inference(
        model: DecoderMixin,
        encoder_outputs: Dict[str, torch.Tensor],
        encoder_lengths: Dict[str, torch.Tensor],
        bos_token_id: int,
        eos_token_id: int,
        max_length: int,
        score_fn: ScoreFn,
        top_k: int,
        input_strings: Optional[List[str]] = None,
        decoder_positions: Optional[torch.Tensor] = None,
        **kwargs: Any
) -> List[List[Beam]]:
    model.eval()
    device = utils.device_from_model(model)

    all_beams: List[List[Beam]] = []

    batch_size = len(next(iter(encoder_outputs.values())))

    positions = decoder_positions if decoder_positions is not None else torch.zeros(batch_size, dtype=torch.long)

    for b in range(batch_size):
        # encoder_outputs_b: shape [L, H]
        encoder_outputs_b = _sub_select(encoder_outputs, b)
        encoder_lengths_b = _sub_select(encoder_lengths, b)

        # initialize beams
        beam_queue = queue.PriorityQueue()
        finished_beams = []

        beam = Beam()
        beam.token_ids.append(bos_token_id)
        beam.log_prob.append(0.0)
        beam_queue.put(
            (
                -score_fn(
                    beam,
                    input_strings[b] if input_strings is not None else None
                ),
                beam
            )
        )

        search_depth = 1

        while len(finished_beams) < top_k and positions[b] + search_depth < max_length and not beam_queue.empty():
            beam: Beam = beam_queue.get()[1]

            if beam.is_eos(eos_token_id):
                finished_beams.append(beam)
                continue

            continuations = einops.repeat(
                torch.tensor(beam.token_ids, dtype=torch.long, device=device),
                "l -> b l", b=1
            )
            decoder_lengths = torch.tensor(
                [len(beam.token_ids)],
                dtype=torch.long,
                device=device
            )
            decoder_positions = torch.arange(
                positions[b], positions[b] + len(beam.token_ids),
                device=device,
                dtype=torch.long
            ).unsqueeze(0)

            beam_encoder_feats_b = {k: einops.repeat(v, "l h -> b l h", b=len(continuations))
                                    for k, v in encoder_outputs_b.items()}
            beam_encoder_lengths_b = {k: einops.repeat(v, "-> repeat", repeat=len(continuations))
                                      for k, v in encoder_lengths_b.items()}

            # decoder_output: shape [B, VOC]
            decoder_output = model.decode(
                decoder_inputs=continuations,
                decoder_lengths=decoder_lengths,
                encoder_outputs=beam_encoder_feats_b,
                encoder_lengths=beam_encoder_lengths_b,
                decoder_positions=decoder_positions
            )[:, -1, ...]

            log_softmax_scores = torch.log_softmax(decoder_output, dim=1)[0].tolist()

            min_log_prob = math.log(1 / len(log_softmax_scores))
            for i, score in enumerate(log_softmax_scores):
                if score < min_log_prob:
                    continue
                new_beam = Beam.from_beam(beam, score, i)
                beam_queue.put(
                    (
                        -score_fn(
                            new_beam,
                            input_strings[b] if input_strings is not None else None
                        ),
                        new_beam
                    )
                )
            search_depth = max(search_depth, len(beam.token_ids) + 1)

        while len(finished_beams) < top_k and not beam_queue.empty():
            finished_beams.append(beam_queue.get()[1])

        all_beams.append(finished_beams)

    return all_beams


@torch.inference_mode()
def beam_inference(
        model: DecoderMixin,
        encoder_outputs: Dict[str, torch.Tensor],
        encoder_lengths: Dict[str, torch.Tensor],
        bos_token_id: int,
        eos_token_id: int,
        max_length: int,
        score_fn: ScoreFn,
        beam_width: int,
        input_strings: Optional[List[str]] = None,
        decoder_positions: Optional[torch.Tensor] = None,
        **kwargs: Any
) -> List[List[Beam]]:
    model.eval()
    device = utils.device_from_model(model)

    all_beams: List[List[Beam]] = []

    batch_size = len(next(iter(encoder_outputs.values())))

    positions = decoder_positions if decoder_positions is not None else torch.zeros(batch_size, dtype=torch.long)

    for b in range(batch_size):
        # encoder_outputs_b: shape [L, H]
        encoder_outputs_b = _sub_select(encoder_outputs, b)
        encoder_lengths_b = _sub_select(encoder_lengths, b)

        # initialize beams
        beam_queue = queue.PriorityQueue()
        current_beams = []
        beam = Beam()
        beam.token_ids.append(bos_token_id)
        beam.log_prob.append(0.0)
        current_beams.append(beam)

        search_depth = 1

        while beam_queue.qsize() < beam_width and positions[b] + search_depth < max_length and len(current_beams) > 0:
            decoder_inputs = torch.tensor(
                [beam.token_ids for beam in current_beams],
                dtype=torch.long,
                device=device
            )
            L = decoder_inputs.shape[1]
            decoder_lengths = torch.tensor(
                [L] * len(current_beams),
                dtype=torch.long,
                device=device
            )
            decoder_positions = einops.repeat(
                torch.arange(positions[b], positions[b] + L, dtype=torch.long, device=device),
                "l -> b l",
                b=len(current_beams)
            )

            beam_encoder_feats_b = {k: einops.repeat(v, "l h -> b l h", b=len(current_beams))
                                    for k, v in encoder_outputs_b.items()}
            beam_encoder_lengths_b = {k: einops.repeat(v, "-> repeat", repeat=len(current_beams))
                                      for k, v in encoder_lengths_b.items()}

            decoder_output = model.decode(
                decoder_inputs=decoder_inputs,
                decoder_lengths=decoder_lengths,
                encoder_outputs=beam_encoder_feats_b,
                encoder_lengths=beam_encoder_lengths_b,
                decoder_positions=decoder_positions
            )[:, -1, ...]

            log_softmax_scores = torch.log_softmax(decoder_output, dim=1)

            min_log_prob = math.log(1 / log_softmax_scores.shape[1])
            valid_indices = torch.nonzero(log_softmax_scores >= min_log_prob, as_tuple=True)
            valid_log_prob = log_softmax_scores[valid_indices].tolist()
            valid_indices = torch.stack(valid_indices, dim=-1).tolist()

            beam_candidates = []
            for (beam_idx, token_id), log_p in zip(valid_indices, valid_log_prob):
                beam_candidate = Beam.from_beam(
                    current_beams[beam_idx],
                    log_p=log_p,
                    token_id=token_id
                )
                beam_candidates.append(
                    (
                        beam_candidate, -score_fn(
                            beam_candidate,
                            input_strings[b] if input_strings is not None else None
                        )
                    )
                )

            beam_candidates = sorted(beam_candidates, key=lambda e: e[1])[:2 * beam_width]

            current_beams = []
            for beam, score in beam_candidates:
                if beam.is_eos(eos_token_id):
                    beam_queue.put((score, beam))
                else:
                    current_beams.append(beam)

                if len(current_beams) >= beam_width:
                    break

            search_depth += 1

        if beam_queue.qsize() < beam_width and len(current_beams) > 0:
            # if we did not find beam_width solutions that end in eos,
            # add the highest scoring remaining active beams to queue
            current_beams = [
                (
                    beam, -score_fn(
                        beam,
                        input_strings[b] if input_strings is not None else None
                    )
                )
                for beam in current_beams
            ]
            current_beams = sorted(current_beams, key=lambda e: e[1])
            for beam, score in current_beams:
                beam_queue.put((score, beam))
                if beam_queue.qsize() >= beam_width:
                    break

        output_beams: List[Beam] = [beam_queue.get()[1] for _ in range(min(beam_width, beam_queue.qsize()))]
        all_beams.append(output_beams)

    return all_beams


def get_de_tok_fn(output_tokenizer: tokenization.Tokenizer, bos_token_id: int, eos_token_id: int) \
        -> Callable[[List[int]], str]:
    def de_tokenize(token_ids: List[int]) -> str:
        if len(token_ids) < 2:
            return ""

        assert token_ids[0] == bos_token_id
        token_ids = token_ids[1:]
        if token_ids[-1] == eos_token_id:
            token_ids = token_ids[:-1]
        return output_tokenizer.de_tokenize(token_ids)

    return de_tokenize


def run_inference(
        model: DecoderMixin,
        output_tokenizer: tokenization.Tokenizer,
        encoder_outputs: Dict[str, torch.Tensor],
        encoder_lengths: Dict[str, torch.Tensor],
        max_length: int,
        search_mode: str = "greedy",
        score_fn: ScoreFn = spelling_correction_score(),
        input_strings: Optional[List[str]] = None,
        **kwargs: Any) -> List[List[str]]:
    bos_token_id = output_tokenizer.token_to_id(tokenization.BOS)
    eos_token_id = output_tokenizer.token_to_id(tokenization.EOS)
    de_tok_fn = get_de_tok_fn(output_tokenizer, bos_token_id, eos_token_id)

    if search_mode == "greedy":
        outputs = token_inference(
            model=model,
            encoder_outputs=encoder_outputs,
            encoder_lengths=encoder_lengths,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            max_length=max_length,
            select_fn=greedy_select_fn(),
            score_fn=score_fn,
            input_strings=input_strings,
            **kwargs
        )
    elif search_mode == "sample":
        sample_top_k = kwargs.pop("sample_top_k", 5)
        outputs = token_inference(
            model=model,
            encoder_outputs=encoder_outputs,
            encoder_lengths=encoder_lengths,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            max_length=max_length,
            select_fn=sample_select_fn(sample_top_k),
            score_fn=score_fn,
            input_strings=input_strings,
            **kwargs
        )
    elif search_mode == "beam":
        beam_width = kwargs.pop("beam_width", 5)
        outputs = beam_inference(
            model=model,
            encoder_outputs=encoder_outputs,
            encoder_lengths=encoder_lengths,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            max_length=max_length,
            score_fn=score_fn,
            beam_width=beam_width,
            input_strings=input_strings,
            **kwargs
        )
    elif search_mode == "best_first":
        top_k = kwargs.pop("best_first_top_k", 1)
        outputs = best_first_inference(
            model=model,
            encoder_outputs=encoder_outputs,
            encoder_lengths=encoder_lengths,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            max_length=max_length,
            score_fn=score_fn,
            top_k=top_k,
            input_strings=input_strings,
            **kwargs
        )
    else:
        raise ValueError(f"unknown search mode {search_mode}")

    return [inference_result_to_sequence(ir, de_tok_fn) for ir in outputs]


def inference_result_to_sequence(
        inference_result: Union[List[int], List[Beam]],
        de_tok_fn: DeTokFn
) -> List[str]:
    if isinstance(inference_result, list) and isinstance(inference_result[0], int):
        # greedy or sample inference
        return [de_tok_fn(inference_result)]
    elif isinstance(inference_result, list) and isinstance(inference_result[0], Beam):
        # beam or best first inference
        return [de_tok_fn(beam.token_ids) for beam in inference_result]
    else:
        raise ValueError(f"expected output decoding inference result to be either a list of token ids"
                         f"or a list of beams, but got {type(inference_result)}")


def inference_output_to_str(output: Union[int, List[int], List[str], str]) -> str:
    if isinstance(output, int):
        return str(output)
    elif isinstance(output, list) and all(isinstance(o, int) for o in output):
        return " ".join(str(o) for o in output)
    elif isinstance(output, list) and all(isinstance(o, str) for o in output):
        return output[0]
    elif isinstance(output, str):
        return output
    else:
        raise ValueError(f"Output has to be either an int, a list of ints or strings, or a string, "
                         f"but got {type(output)} ({output})")
