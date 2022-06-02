import contextlib
import dataclasses
import logging
import os
import pprint
import threading
import time
from typing import Dict, Generator, List, Optional

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
    get_available_spelling_error_correction_models
)
from nsc.api.utils import ModelInfo
from nsc.utils import common

# disable flask startup message and set flask mode to development
from nsc.utils.edit import get_edited_words

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
        self.locks: Dict[str, threading.Lock] = {}
        self.streams: Dict[str, torch.cuda.Stream] = {}
        self.loaded = False
        self.default_models = {}

    def init(self, model_infos: List[ModelInfo], timeout: float, precision: str) -> None:
        num_devices = torch.cuda.device_count()
        logger.info(f"found {num_devices} GPUs")
        self.timeout = timeout
        for i, model_info in enumerate(model_infos):
            name = _full_name(model_info)
            self.locks[name] = threading.Lock()

            if model_info.task == "sed words" or model_info.task == "sed sequence":
                cls = SpellingErrorDetector
            elif model_info.task == "tokenization repair":
                cls = TokenizationRepairer
            elif model_info.task == "sec":
                cls = SpellingErrorCorrector
            else:
                raise RuntimeError("should not happen")

            if num_devices > 0:
                self.streams[name] = torch.cuda.Stream()
                spell_checker = cls.from_pretrained(task=model_info.task, model=model_info.name, device=i % num_devices)
            else:
                spell_checker = cls.from_pretrained(task=model_info.task, model=model_info.name, device="cpu")

            spell_checker.set_precision(precision)
            self.__setattr__(name, spell_checker)
            if model_info.task not in self.default_models:
                self.default_models[model_info.task] = model_info.name

        self.loaded = True

    @property
    def available_models(self) -> List[ModelInfo]:
        return [model for model in _all_models() if _full_name(model) in self.locks]

    @contextlib.contextmanager
    def get_model(self, task: str, model_name: Optional[str] = None) -> Generator:
        if not self.loaded:
            yield "models were not loaded", 500  # internal error, need to load models before using them
            return

        if model_name is None:
            if task not in self.default_models:
                yield f"no model for task {task} found", 400
                return
            model_name = self.default_models[task]

        name = f"{task}:{model_name}"

        # yield either the tokenization repair model or a message + http status code indicating why this did not work
        if not hasattr(self, name):
            yield f"model '{name}' is not available", 400  # user error, cant request a model that does not exist
        else:
            acquired = self.locks[name].acquire(timeout=self.timeout)
            if not acquired:
                # server capacity is maxed out when acquiring the model did not work within timeout range
                yield f"server is overloaded with too many requests, failed to reserve model " \
                      f"within the {self.timeout:.2f}s timeout limit", 503
            else:
                if name in self.streams:
                    with torch.cuda.stream(self.streams[name]):
                        yield self.__getattribute__(name)
                else:
                    yield self.__getattribute__(name)
                self.locks[name].release()


models = Models()


@server.after_request
def after_request(response: Response) -> Response:
    response.headers.add("Access-Control-Allow-Origin", os.environ.get("CORS_ORIGIN", "*"))
    response.headers.add("Access-Control-Allow-Private-Network", "true")
    return response


@server.route(f"{server_base_url}/models")
def get_models() -> Response:
    response = jsonify(
        {
            "models": [
                {"task": model.task, "name": model.name, "description": model.description}
                for model in models.available_models
            ],
            "default": models.default_models
        }
    )
    return response


@server.route(f"{server_base_url}/process_text", methods=["POST"])
def repair_text() -> Response:
    start = time.perf_counter()
    text = request.form.get("text")
    if text is None:
        return abort(Response("request missing required 'text' field in form data", status=400))
    text = [line.strip() for line in text.splitlines()]
    if "task" not in request.args:
        return abort(Response("request missing required 'task' query parameter", status=400))
    task = request.args["task"]
    with models.get_model(task, request.args.get("model")) as spell_checker:
        if isinstance(spell_checker, tuple):
            message, status_code = spell_checker
            logger.warning(f"Repairing text aborted with status {status_code}: {message}")
            return abort(Response(message, status=status_code))
        elif isinstance(spell_checker, TokenizationRepairer):
            repaired = spell_checker.repair_text(text, show_progress=False)
            output = {
                "text": repaired
            }
        elif isinstance(spell_checker, SpellingErrorDetector):
            detections, repaired = spell_checker.detect_text(text, show_progress=False)
            output = {
                "detections": detections,
                "text": repaired
            }
        elif isinstance(spell_checker, SpellingErrorCorrector):
            corrected = spell_checker.correct_text(text, show_progress=False)
            edited_in_input, edited_in_correction = get_edited_words(text, corrected)
            output = {
                "text": corrected,
                "edited": {
                    "input": [sorted(list(e)) for e in edited_in_input],
                    "text": [sorted(list(e)) for e in edited_in_correction]
                }
            }
        else:
            raise RuntimeError("should not happen")

    end = time.perf_counter()
    runtime = end - start
    text_bytes = sum(len(line.encode("utf8")) for line in text)
    logger.info(f"processing text with {text_bytes} bytes for task {task} with model "
                f"{request.args.get('model', models.default_models[task])} took {runtime:.2f}s")
    response = jsonify(
        {
            "output": output,
            "runtime": {
                "total": runtime,
                "bps": text_bytes / runtime
            }
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


def run_flask_server(config_path: str) -> None:
    if not os.path.exists(config_path):
        raise RuntimeError(f"server config file {config_path} does not exist")

    dict_config = omegaconf.OmegaConf.load(config_path)
    config: ServerConfig = omegaconf.OmegaConf.structured(ServerConfig(**dict_config))

    logger.info(f"loaded config for server:\n{pprint.pformat(config)}")

    model_infos = [model_info for model_info in _all_models()
                   if model_info.task in config.models and model_info.name in config.models[model_info.task]]
    logger.info(f"found {len(model_infos)} valid model specifications, loading the following models:\n"
                f"{pprint.pformat(model_infos)}")
    models.init(model_infos=model_infos, timeout=config.timeout, precision=config.precision)

    logger.info(f"starting server on {config.host}:{config.port}...")
    server.run(config.host, config.port, debug=False, use_reloader=False)
