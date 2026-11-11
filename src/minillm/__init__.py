"""MiniLLM — an 87M-parameter decoder-only Chinese-first language model."""

from .config import ARCH_VERSION, ARCHITECTURE, PRESETS, MiniLLMConfig, get_config
from .model import MiniLLM

__version__ = "0.1.0"

__all__ = [
    "ARCHITECTURE",
    "ARCH_VERSION",
    "MiniLLM",
    "MiniLLMConfig",
    "PRESETS",
    "get_config",
]
