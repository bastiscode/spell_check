import math
import queue
from typing import Callable, Tuple, Dict, List, Any, Union, Optional

import einops
import torch

from gnn_lib.data import tokenization, index, utils as data_utils
from gnn_lib.modules import utils
from gnn_lib.modules.utils import DecoderMixin

TOKEN_FN = Callable[[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]


def greedy_token_fn() -> TOKEN_FN:
    def _greedy(decoder_output: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        values, indices = torch.max(torch.log_softmax(decoder_output, dim=1), dim=1)
        return indices, values

    return _greedy


def sample_token_fn(sample_top_k: int) -> TOKEN_FN:
    def _sample(decoder_output: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        top_k_probabilities, top_k_indices = torch.softmax(decoder_output, dim=1).topk(sample_top_k, dim=1)
        sampled_indices = torch.multinomial(top_k_probabilities, num_samples=1)
        indices = top_k_indices.gather(1, sampled_indices).squeeze(1)
        values = torch.log(top_k_probabilities.gather(1, sampled_indices)).squeeze(1)
        return indices, values

    return _sample


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


SCORE_FN = Callable[
    [
        Beam,  # beam with token ids and log probs
        int,  # bos token id
        int,  # eos token id
        Optional[str],  # input string
        Optional[Callable[[List[int]], str]],  # function from token_ids to string (de-tokenization)
        Optional[index.PrefixIndex]  # prefix index, to check if a word is a prefix of a word in the dictionary
    ],
    float
]


def map_score(normalize_by_length: bool = True, alpha: float = 1.0) -> SCORE_FN:
    def score(beam: Beam,
              bos_token_id: int,
              eo_token_id: int,
              _: Optional[str] = None,
              __: Optional[Callable[[List[int]], str]] = None,
              ___: Optional[Dict[str, int]] = None) -> float:
        s = sum(beam.log_prob)
        if normalize_by_length:
            return s / (len(beam.log_prob) ** alpha)
        else:
            return s

    return score


def spell_check_score(
        normalize_by_length: bool = True,
        alpha: float = 1.0,
        # if not None, must be one of [dictionary, dictionary_or_in_input, dictionary_or_eq_input]
        mode: Optional[str] = None
) -> SCORE_FN:
    def score(beam: Beam,
              bos_token_id: int,
              eos_token_id: int,
              input_str: Optional[str] = None,
              de_tok_fn: Optional[Callable[[List[int]], str]] = None,
              prefix_index: Optional[index.PrefixIndex] = None) -> float:
        assert beam.token_ids[0] == bos_token_id
        token_ids = beam.token_ids[1:]  # strip bos token
        if beam.is_eos(eos_token_id):  # strip eos token
            token_ids = token_ids[:-1]

        pred_str = de_tok_fn(token_ids)
        pred_str_split = pred_str.split()

        if mode is not None and len(pred_str_split) > 0:
            assert prefix_index is not None

            # get all input words
            input_words, input_ws = data_utils.tokenize_words_regex(input_str)
            # split current predicted word (by whitespace) further using regex
            pred_words, pred_ws = data_utils.tokenize_words_regex(pred_str_split[-1])

            valid_pred = False
            # check if current predicted word (or its lowercase version)
            # is a prefix of a dictionary word (in prefix tree)
            valid_pred |= (
                    len(prefix_index.retrieve(pred_words[-1])) > 0
                    or len(prefix_index.retrieve(pred_words[-1].lower())) > 0
            )

            # check if current predicted word (or its lowercase version)
            # is a prefix of an input word
            if mode == "dictionary_or_in_input":
                valid_pred |= (
                        any(ipt_w.startswith(pred_words[-1]) for ipt_w in input_words)
                        or any(ipt_w.startswith(pred_words[-1].lower()) for ipt_w in input_words)
                )

            # check if current predicted string is prefix of the input string
            if mode == "dictionary_or_eq_input":
                valid_pred |= input_str.startswith(pred_str)

            if not valid_pred:
                return -1_000_000

        s = sum(beam.log_prob)
        if normalize_by_length:
            s = s / (len(beam.log_prob) ** alpha)

        return s

    return score


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
        token_fn: TOKEN_FN = greedy_token_fn(),
        input_strings: Optional[List[str]] = None,
        prefix_index: Optional[index.PrefixIndex] = None,
        de_tok_fn: Optional[Callable[[List[int]], str]] = None,
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
    if "decoder_positions" in kwargs:
        positions = torch.stack([
            torch.arange(pos, pos + max_length, device=device)
            for pos in kwargs["decoder_positions"]
        ])
    else:
        positions = einops.repeat(torch.arange(max_length, device=device), "l -> b l", b=batch_size)

    non_eos_mask = torch.ones(batch_size, dtype=torch.bool, device=device)
    smaller_max_length_mask = torch.ones(batch_size, dtype=torch.bool, device=device)
    indices_to_decode = torch.ones(batch_size, dtype=torch.bool, device=device)

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

        selected_decoder_outputs = []
        for i in range(len(decoder_output)):
            length = lengths[indices_to_decode][i]
            selected_decoder_outputs.append(decoder_output[i, length - 1])
        selected_decoder_outputs = torch.stack(selected_decoder_outputs)

        inferred_token_ids, inferred_log_prob = token_fn(selected_decoder_outputs)

        token_ids[indices_to_decode, lengths[indices_to_decode]] = inferred_token_ids
        log_prob[indices_to_decode, lengths[indices_to_decode]] = inferred_log_prob

        lengths[indices_to_decode] += 1

        inferred_eos_indices = torch.where(inferred_token_ids == eos_token_id)[0]
        new_eos_indices = torch.where(indices_to_decode)[0][inferred_eos_indices]
        non_eos_mask[new_eos_indices] = False

        max_length_indices = torch.where(lengths >= max_length)[0]
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
        score_fn: SCORE_FN,
        top_k: int,
        input_strings: Optional[List[str]] = None,
        prefix_index: Optional[index.PrefixIndex] = None,
        de_tok_fn: Optional[Callable[[List[int]], str]] = None,
        **kwargs: Any
) -> List[List[Beam]]:
    model.eval()
    device = utils.device_from_model(model)

    all_beams: List[List[Beam]] = []

    batch_size = len(next(iter(encoder_outputs.values())))

    positions = kwargs.get("decoder_positions", torch.zeros(batch_size, dtype=torch.long))

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
                    bos_token_id,
                    eos_token_id,
                    input_strings[b] if input_strings is not None else None,
                    de_tok_fn,
                    prefix_index
                ),
                beam
            )
        )

        search_depth = 1

        while len(finished_beams) < top_k and search_depth < max_length and not beam_queue.empty():
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
                            bos_token_id,
                            eos_token_id,
                            input_strings[b] if input_strings is not None else None,
                            de_tok_fn,
                            prefix_index
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
        score_fn: SCORE_FN,
        beam_width: int,
        input_strings: Optional[List[str]] = None,
        prefix_index: Optional[index.PrefixIndex] = None,
        de_tok_fn: Optional[Callable[[List[int]], str]] = None,
        **kwargs: Any
) -> List[List[Beam]]:
    model.eval()
    device = utils.device_from_model(model)

    all_beams: List[List[Beam]] = []

    batch_size = len(next(iter(encoder_outputs.values())))

    positions = kwargs.get("decoder_positions", torch.zeros(batch_size, dtype=torch.long))

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

        while beam_queue.qsize() < beam_width and search_depth < max_length and len(current_beams) > 0:
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
                beam_candidates.append((beam_candidate, -score_fn(
                    beam_candidate,
                    bos_token_id,
                    eos_token_id,
                    input_strings[b] if input_strings is not None else None,
                    de_tok_fn,
                    prefix_index
                )))

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
            current_beams = [(beam, -score_fn(
                beam,
                bos_token_id,
                eos_token_id,
                input_strings[b] if input_strings is not None else None,
                de_tok_fn,
                prefix_index
            )) for beam in current_beams]
            current_beams = sorted(current_beams, key=lambda e: e[1])
            for beam, score in current_beams:
                beam_queue.put((score, beam))
                if beam_queue.qsize() >= beam_width:
                    break

        output_beams: List[Beam] = [beam_queue.get()[1] for _ in range(min(beam_width, beam_queue.qsize()))]
        all_beams.append(output_beams)

    return all_beams


def run_inference(
        model: DecoderMixin,
        output_tokenizer: tokenization.Tokenizer,
        encoder_outputs: Dict[str, torch.Tensor],
        encoder_lengths: Dict[str, torch.Tensor],
        max_length: int,
        input_strings: Optional[List[str]] = None,
        prefix_index: Optional[index.PrefixIndex] = None,
        **kwargs: Any) -> List[List[str]]:
    bos_token_id = output_tokenizer.token_to_id(tokenization.BOS)
    eos_token_id = output_tokenizer.token_to_id(tokenization.EOS)
    de_tok_fn = output_tokenizer.de_tokenize

    inference_mode = kwargs.get("inference_mode", "greedy")
    if inference_mode == "greedy":
        outputs = token_inference(
            model=model,
            encoder_outputs=encoder_outputs,
            encoder_lengths=encoder_lengths,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            max_length=max_length,
            token_fn=greedy_token_fn(),
            input_strings=input_strings,
            prefix_index=prefix_index,
            de_tok_fn=de_tok_fn,
            **kwargs
        )
    elif inference_mode == "sample":
        sample_top_k = kwargs.pop("sample_top_k", 5)
        outputs = token_inference(
            model=model,
            encoder_outputs=encoder_outputs,
            encoder_lengths=encoder_lengths,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            max_length=max_length,
            token_fn=sample_token_fn(sample_top_k),
            input_strings=input_strings,
            prefix_index=prefix_index,
            de_tok_fn=de_tok_fn,
            **kwargs
        )
    elif inference_mode == "beam":
        score_fn = kwargs.pop("score_fn", map_score())
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
            prefix_index=prefix_index,
            de_tok_fn=de_tok_fn,
            **kwargs
        )
    elif inference_mode == "best_first":
        score_fn = kwargs.pop("score_fn", map_score())
        top_k = kwargs.pop("best_first_top_k", 5)
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
            prefix_index=prefix_index,
            de_tok_fn=de_tok_fn,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown inference mode {inference_mode}")

    return [
        inference_result_to_sequence(
            ir,
            output_tokenizer,
            bos_token_id,
            eos_token_id
        )
        for ir in outputs
    ]


def token_ids_to_sequence(
        token_ids: List[int],
        output_tokenizer: tokenization.Tokenizer,
        bos_token_id: int,
        eos_token_id: int
) -> str:
    # strip potential bos and eos tokens
    if token_ids[0] == bos_token_id:
        token_ids = token_ids[1:]
    if token_ids[-1] == eos_token_id:
        token_ids = token_ids[:-1]
    return output_tokenizer.de_tokenize(token_ids)


def inference_result_to_sequence(
        inference_result: Union[List[int], List[Beam]],
        output_tokenizer: tokenization.Tokenizer,
        bos_token_id: int,
        eos_token_id: int
) -> List[str]:
    if isinstance(inference_result, list) and isinstance(inference_result[0], int):
        # greedy or sample inference
        return [token_ids_to_sequence(inference_result, output_tokenizer, bos_token_id, eos_token_id)]
    elif isinstance(inference_result, list) and isinstance(inference_result[0], Beam):
        # beam or best first inference
        return [
            token_ids_to_sequence(beam.token_ids, output_tokenizer, bos_token_id, eos_token_id)
            for beam in inference_result
        ]
    else:
        raise ValueError(f"expected output decoding inference result to be either a list of token ids"
                         f"or a list of beams, but got {type(inference_result)}")


def inference_output_to_str(output: Union[int, List[int], List[str], str]) -> str:
    if isinstance(output, int):
        return str(output)
    elif isinstance(output, list) and isinstance(output[0], int):
        return " ".join(str(o) for o in output)
    elif isinstance(output, list) and isinstance(output[0], str):
        return output[0]
    elif isinstance(output, str):
        return output
    else:
        raise ValueError(f"Output has to be either an int, a list of ints or strings, or a string, "
                         f"but got {type(output)} ({output})")
