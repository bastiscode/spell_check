import contextlib
import dataclasses
import logging
import os
import pprint
import sys
import threading
import time
import traceback
from typing import Dict, Generator, List, Optional, Any, Tuple, Set

import omegaconf
from flask import Flask, Response, abort, cli, jsonify, request

import torch.cuda
from omegaconf import MISSING

from nsc import (
    SpellingErrorCorrector,
    SpellingErrorDetector,
    TokenizationRepairer,
    get_available_tokenization_repair_models,
    get_available_spelling_error_detection_models,
    get_available_spelling_error_correction_models, version
)
from nsc.api.utils import ModelInfo, get_cpu_info, get_gpu_info
from nsc.utils import common

# disable flask startup message and set flask mode to development
from nsc.utils.edit import get_edited_words
from nsc.utils.tokenization_repair import get_whitespace_operations

cli.show_server_banner = lambda *_: None
os.environ["FLASK_ENV"] = "development"
server = Flask(__name__)
server.config["MAX_CONTENT_LENGTH"] = 1 * 1000 * 1000  # 1MB max file size
server_base_url = os.environ.get("BASE_URL", "")
flask_logger = logging.getLogger("werkzeug")
flask_logger.disabled = True
logger = common.get_logger("TRT_SERVER")


def _all_models() -> List[ModelInfo]:
    return (
            get_available_tokenization_repair_models()
            + get_available_spelling_error_detection_models()
            + get_available_spelling_error_correction_models()
    )


def _full_name(model: ModelInfo) -> str:
    return f"{model.task}:{model.name}"


class Models:
    def __init__(self) -> None:
        self.timeout = 1.0
        self.precision = "fp32"
        self.lock: threading.Lock = threading.Lock()
        self.streams: Dict[str, Optional[torch.cuda.Stream]] = {}
        self.model_to_device: Dict[str, str] = {}
        self.models_on_gpu: Dict[str, Set[str]] = {}
        self.loaded = False
        self.max_models_per_device = 3
        self.available_models = []
        self.num_devices = torch.cuda.device_count()

    def init(self, model_infos: List[ModelInfo], timeout: float, precision: str, max_models_per_gpu: int) -> None:
        self.available_models = model_infos
        logger.info(f"found {self.num_devices} GPUs")
        self.precision = precision
        self.timeout = timeout
        self.max_models_per_device = max_models_per_gpu
        for i, model_info in enumerate(model_infos):
            logger.info(f"loading model {model_info.name} for task {model_info.task}")
            name = _full_name(model_info)

            if model_info.task == "sed words" or model_info.task == "sed sequence":
                cls = SpellingErrorDetector
            elif model_info.task == "tokenization repair":
                cls = TokenizationRepairer
            elif model_info.task == "sec":
                cls = SpellingErrorCorrector
            else:
                raise RuntimeError("should not happen")

            if self.num_devices > 0:
                self.streams[name] = torch.cuda.Stream()
                device_name = f"cuda:{i % self.num_devices}"
                self.model_to_device[name] = device_name
                if device_name not in self.models_on_gpu:
                    self.models_on_gpu[device_name] = {name}
                elif len(self.models_on_gpu[device_name]) < self.max_models_per_device:
                    self.models_on_gpu[device_name].add(name)
                else:
                    device_name = "cpu"
                spell_checker = cls.from_pretrained(
                    task=model_info.task,
                    model=model_info.name,
                    device=device_name
                )
            else:
                self.streams[name] = None
                spell_checker = cls.from_pretrained(
                    task=model_info.task,
                    model=model_info.name,
                    device="cpu"
                )
                self.model_to_device[name] = "cpu"

            spell_checker.set_precision(precision)
            self.__setattr__(name, spell_checker)

        self.loaded = True

    @property
    def pipeline_tasks(self) -> List[str]:
        return ["tokenization repair", "sed words", "sec"]

    def is_valid_pipeline(self, pipeline_models: Tuple[Optional[str], ...]) -> bool:
        if (
                len(pipeline_models) != 3
                or all(m is None for m in pipeline_models)
                or any(not hasattr(self, f"{task}:{model_name}")
                       for task, model_name
                       in zip(self.pipeline_tasks, pipeline_models) if model_name is not None)
        ):
            return False
        else:
            return True

    def get_lock(self) -> bool:
        return self.lock.acquire(blocking=True, timeout=self.timeout)

    def release_lock(self) -> None:
        if self.lock.locked():
            self.lock.release()

    def get_pipeline(self, pipeline_models: Tuple[Optional[str], ...]) -> Generator:
        try:
            acquired = self.get_lock()
            if not acquired:
                raise RuntimeError("could not acquire lock within timeout")
            pipeline = tuple(
                f"{task}:{model_name}"
                for task, model_name in zip(self.pipeline_tasks, pipeline_models)
                if model_name is not None
            )
            for name in pipeline:
                device = self.model_to_device[name]
                logger.info(f"model {name}, {self.model_to_device[name]}, {self.models_on_gpu[device]}")
                if device != "cpu" and name not in self.models_on_gpu[device]:
                    if len(self.models_on_gpu[device]) >= self.max_models_per_device:
                        model_to_move = None
                        # determine the model to move to cpu to make space for our pipeline models,
                        # should ideally be none of the other pipeline models
                        for model_on_gpu in self.models_on_gpu[device]:
                            if model_on_gpu not in pipeline:
                                model_to_move = model_on_gpu
                                break
                        # should only be true if self.max_models_per_device < 3 (max number of pipeline steps)
                        if model_to_move is None:
                            model_to_move = next(iter(self.models_on_gpu[device]))
                        logger.info(f"moving '{model_to_move}' to CPU to make space for '{name}' on {device}")
                        self.models_on_gpu[device].remove(model_to_move)
                        self.__getattribute__(model_to_move).to("cpu")
                    self.__getattribute__(name).to(device)
                    self.models_on_gpu[device].add(name)
                with torch.cuda.stream(self.streams[name]):
                    logger.info(f"yielding {name}")
                    yield self.__getattribute__(name)
        finally:
            self.release_lock()


models = Models()


@server.after_request
def after_request(response: Response) -> Response:
    response.headers.add("Access-Control-Allow-Origin", os.environ.get("CORS_ORIGIN", "*"))
    response.headers.add("Access-Control-Allow-Private-Network", "true")
    return response


@server.route(f"{server_base_url}/info")
def get_info() -> Response:
    response = jsonify(
        {
            "gpu": [get_gpu_info(i) for i in range(torch.cuda.device_count())],
            "cpu": get_cpu_info(),
            "timeout": models.timeout,
            "precision": models.precision,
            "version": version.__version__
        }
    )
    return response


@server.route(f"{server_base_url}/models")
def get_models() -> Response:
    response = jsonify([
        {"task": model.task, "name": model.name, "description": model.description}
        for model in models.available_models
    ])
    return response


@server.route(f"{server_base_url}/process_text", methods=["POST"])
def process_text() -> Response:
    start_total = time.perf_counter()

    text = request.form.get("text")
    if text is None:
        return abort(Response("request missing required 'text' field in form data", status=400))
    text = [line.strip() for line in text.splitlines()]
    org_text_bytes = sum(len(line.encode("utf8")) for line in text)

    if "pipeline" not in request.args:
        return abort(Response("request missing required 'pipeline' query parameter", status=400))
    pipeline = [m.strip() for m in request.args["pipeline"].split(",")]
    pipeline = tuple(m or None for m in pipeline)
    if not models.is_valid_pipeline(pipeline):
        return abort(
            Response(f"invalid pipeline specification, expected exactly 3 models for tokenization repair, "
                     f"word-level spelling error detection and sec respectively, but got {pipeline}", status=400)
        )

    # success = models.get_lock()
    # # server capacity is maxed out when acquiring the model did not work within timeout range
    # if not success:
    #     return abort(
    #         Response(f"server is overloaded with too many requests, failed to reserve pipeline "
    #                  f"within the {models.timeout:.2f}s timeout limit", status=503)
    #     )

    output: Dict[str, Any] = {}
    runtimes: Dict[str, Any] = {}
    try:
        for spell_checker in models.get_pipeline(pipeline):
            if isinstance(spell_checker, TokenizationRepairer):
                start = time.perf_counter()
                repaired = spell_checker.repair_text(text, show_progress=False)
                output["tokenization repair"] = {"text": repaired}
                if request.args.get("edited", "false") == "true":
                    edited = [get_whitespace_operations(ipt, rep) for ipt, rep in zip(text, repaired)]
                    output["tokenization repair"]["edited"] = edited
                end = time.perf_counter()
                runtime = end - start
                text_bytes = sum(len(line.encode("utf8")) for line in text)
                runtimes["tokenization repair"] = {"s": runtime, "bps": text_bytes / runtime}
                text = repaired
            elif isinstance(spell_checker, SpellingErrorDetector):
                start = time.perf_counter()
                detections, repaired = spell_checker.detect_text(text, show_progress=False)
                output["sed words"] = {"text": repaired, "detections": detections}
                end = time.perf_counter()
                runtime = end - start
                text_bytes = sum(len(line.encode("utf8")) for line in text)
                runtimes["sed words"] = {"s": runtime, "bps": text_bytes / runtime}
                text = repaired
            elif isinstance(spell_checker, SpellingErrorCorrector):
                start = time.perf_counter()
                corrected = spell_checker.correct_text(
                    text,
                    detections=output.get("sed words", {}).get("detections"),
                    show_progress=False
                )
                output["sec"] = {"text": corrected}
                if request.args.get("edited", "false") == "true":
                    edited_in_input, edited_in_correction = get_edited_words(text, corrected)
                    output["sec"]["edited"] = {
                        "input": [sorted(list(e)) for e in edited_in_input],
                        "text": [sorted(list(e)) for e in edited_in_correction]
                    }
                end = time.perf_counter()
                runtime = end - start
                text_bytes = sum(len(line.encode("utf8")) for line in text)
                runtimes["sec"] = {"s": runtime, "bps": text_bytes / runtime}
            else:
                raise RuntimeError("should not happen")
    except RuntimeError as e:
        return abort(Response(str(e), status=503))
    except Exception as e:
        logger.error(f"got exception: {e}")
        return abort(Response(str(e), status=503))

    end_total = time.perf_counter()
    total_runtime = end_total - start_total
    logger.info(f"processing text with {org_text_bytes} bytes with pipeline {pipeline} took {total_runtime:.2f}s")
    runtimes["total"] = {"s": total_runtime, "bps": org_text_bytes / total_runtime}
    response = jsonify(
        {
            "output": output,
            "runtimes": runtimes
        }
    )
    return response


@dataclasses.dataclass
class ServerConfig:
    host: str = "0.0.0.0"
    port: int = MISSING
    precision: str = "fp32"
    timeout: float = 10.0
    models: Dict[str, List[str]] = MISSING
    max_models_per_gpu: int = 3


def run_flask_server(config_path: str) -> None:
    if not os.path.exists(config_path):
        raise RuntimeError(f"server config file {config_path} does not exist")

    dict_config = omegaconf.OmegaConf.load(config_path)
    config: ServerConfig = omegaconf.OmegaConf.structured(ServerConfig(**dict_config))

    logger.info(f"loaded config for server:\n{pprint.pformat(config)}")

    model_infos = [
        model_info for model_info in _all_models()
        if model_info.task in config.models and model_info.name in config.models[model_info.task]
    ]
    logger.info(f"found {len(model_infos)} valid model specifications, loading the following models:\n"
                f"{pprint.pformat(model_infos)}")
    models.init(
        model_infos=model_infos,
        timeout=config.timeout,
        precision=config.precision,
        max_models_per_gpu=max(1, config.max_models_per_gpu)
    )

    logger.info(f"starting server on {config.host}:{config.port}...")
    server.run(config.host, config.port, debug=False, use_reloader=False)
