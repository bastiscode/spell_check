import os

from gnn_lib.api.sed import SpellingErrorDetector, get_available_spelling_error_detection_models
from gnn_lib.api.sec import SpellingErrorCorrector, get_available_spelling_error_correction_models

os.environ["TOKENIZERS_PARALLELISM"] = "false"
