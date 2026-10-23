"""TinyVLM — a ~28M-parameter vision-language model that quantises under 50 MB."""

from .config import PRESETS, TextConfig, VisionConfig, VLMConfig, get_config
from .model import TinyVLM, count_parameters
from .tokenizer import TinyTokenizer

__version__ = "0.1.0"

__all__ = [
    "PRESETS",
    "TextConfig",
    "TinyTokenizer",
    "TinyVLM",
    "VLMConfig",
    "VisionConfig",
    "count_parameters",
    "get_config",
]