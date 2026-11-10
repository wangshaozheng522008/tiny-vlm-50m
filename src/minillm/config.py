"""MiniLLM configuration.

The default preset is a dense decoder-only Transformer with 86,528,640
parameters, sized so that an INT4 weight file stays below 50,000,000 bytes:

    token embedding   15,728,640
    GQA attention     15,728,640   (16 layers x 983,040)
    SwiGLU FFN        55,050,240   (16 layers x 3,440,640)
    RMSNorm               21,120
    ------------------------------
    total             86,528,640   (LM head shares the embedding, counted once)

Everything is validated on construction, so a mismatched config fails loudly
instead of producing a model that silently differs from the budget.
"""

from __future__ import annotations

import copy
import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path

ARCHITECTURE = "minillm"
ARCH_VERSION = 2


@dataclass
class MiniLLMConfig:
    vocab_size: int = 24576
    hidden_size: int = 640
    num_hidden_layers: int = 16
    num_attention_heads: int = 10
    num_key_value_heads: int = 2
    head_dim: int = 64
    intermediate_size: int = 1792
    max_position_embeddings: int = 2048
    rope_theta: float = 10000.0
    rms_norm_eps: float = 1e-6
    attention_dropout: float = 0.0
    hidden_dropout: float = 0.0
    tie_word_embeddings: bool = True
    attention_bias: bool = False
    mlp_bias: bool = False
    # Token ids are fixed by the tokenizer (see tokenizer.SPECIAL_TOKEN_IDS).
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2
    unk_token_id: int = 3
    ignore_index: int = -100
    # Checkpoint identity: a VLM checkpoint must never load into this model.
    model_type: str = ARCHITECTURE
    arch_version: int = ARCH_VERSION

    def __post_init__(self) -> None:
        if self.hidden_size != self.num_attention_heads * self.head_dim:
            raise ValueError(
                f"hidden_size ({self.hidden_size}) must equal "
                f"num_attention_heads * head_dim ({self.num_attention_heads} * {self.head_dim})"
            )
        if self.num_key_value_heads > self.num_attention_heads:
            raise ValueError("num_key_value_heads cannot exceed num_attention_heads")
        if self.num_attention_heads % self.num_key_value_heads:
            raise ValueError("num_attention_heads must be a multiple of num_key_value_heads")
        if self.vocab_size <= 0 or self.num_hidden_layers <= 0:
            raise ValueError("vocab_size and num_hidden_layers must be positive")
        if self.max_position_embeddings <= 0:
            raise ValueError("max_position_embeddings must be positive")
        if not 0 <= self.pad_token_id < self.vocab_size:
            raise ValueError("pad_token_id outside the vocabulary")

    # ------------------------------------------------------------------ sizes

    @property
    def num_kv_groups(self) -> int:
        return self.num_attention_heads // self.num_key_value_heads

    def param_breakdown(self) -> dict[str, int]:
        """Analytic parameter budget; the LM head is tied and counted once."""
        attn = (
            self.hidden_size * self.num_attention_heads * self.head_dim  # q
            + self.hidden_size * self.num_key_value_heads * self.head_dim * 2  # k, v
            + self.num_attention_heads * self.head_dim * self.hidden_size  # o
        )
        ffn = 3 * self.hidden_size * self.intermediate_size
        norm = self.hidden_size * (2 * self.num_hidden_layers + 1)
        return {
            "embedding": self.vocab_size * self.hidden_size,
            "attention": attn * self.num_hidden_layers,
            "ffn": ffn * self.num_hidden_layers,
            "norm": norm,
            "total": self.vocab_size * self.hidden_size
            + (attn + ffn) * self.num_hidden_layers
            + norm,
        }

    @property
    def num_parameters(self) -> int:
        return self.param_breakdown()["total"]

    # ------------------------------------------------------------------ io

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict) -> "MiniLLMConfig":
        known = {f.name for f in fields(cls)}
        unknown = set(raw) - known
        if unknown:
            raise ValueError(f"unknown config fields: {sorted(unknown)}")
        return cls(**raw)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "MiniLLMConfig":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def copy(self) -> "MiniLLMConfig":
        return copy.deepcopy(self)


# The budget preset. Do not "fix" a parameter mismatch by editing these numbers:
# tests/test_model.py asserts the exact counts from the project spec.
MINILLM_87M = MiniLLMConfig()

# Tiny preset for tests and the end-to-end smoke test (runs in seconds on CPU).
MINILLM_NANO = MiniLLMConfig(
    vocab_size=512,
    hidden_size=128,
    num_hidden_layers=2,
    num_attention_heads=4,
    num_key_value_heads=1,
    head_dim=32,
    intermediate_size=288,
    max_position_embeddings=128,
)

PRESETS: dict[str, MiniLLMConfig] = {
    "minillm-87m": MINILLM_87M,
    "minillm-nano": MINILLM_NANO,
}


def get_config(name: str) -> MiniLLMConfig:
    if name not in PRESETS:
        raise KeyError(f"unknown preset {name!r}, available: {sorted(PRESETS)}")
    return PRESETS[name].copy()
