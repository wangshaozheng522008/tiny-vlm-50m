"""Model configuration for TinyVLM."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class VisionConfig:
    img_size: int = 224
    patch_size: int = 16
    in_channels: int = 3
    d_model: int = 384
    depth: int = 6
    num_heads: int = 6
    mlp_ratio: float = 4.0
    dropout: float = 0.0

    @property
    def num_patches(self) -> int:
        return (self.img_size // self.patch_size) ** 2


@dataclass
class TextConfig:
    vocab_size: int = 16384
    max_seq_len: int = 256
    d_model: int = 384
    depth: int = 6
    num_heads: int = 6
    mlp_ratio: float = 4.0
    dropout: float = 0.0
    tie_embeddings: bool = True


@dataclass
class VLMConfig:
    vision: VisionConfig = field(default_factory=VisionConfig)
    text: TextConfig = field(default_factory=TextConfig)
    image_token_id: int = 1
    pad_token_id: int = 0
    ignore_index: int = -100

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "VLMConfig":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            vision=VisionConfig(**raw["vision"]),
            text=TextConfig(**raw["text"]),
            image_token_id=raw["image_token_id"],
            pad_token_id=raw["pad_token_id"],
            ignore_index=raw["ignore_index"],
        )


# Tuned so that INT8 weights land comfortably under the 50 MB budget.
# ~28M params -> ~28 MB at INT8, ~14 MB at INT4.
PRESETS: dict[str, VLMConfig] = {
    "tiny-50m": VLMConfig(),
    "nano-50m": VLMConfig(
        vision=VisionConfig(d_model=256, depth=4, num_heads=4),
        text=TextConfig(vocab_size=8192, d_model=256, depth=4, num_heads=4),
    ),
}


def get_config(name: str) -> VLMConfig:
    if name not in PRESETS:
        raise KeyError(f"unknown preset {name!r}, available: {sorted(PRESETS)}")
    cfg = PRESETS[name]
    return VLMConfig(
        vision=VisionConfig(**asdict(cfg.vision)),
        text=TextConfig(**asdict(cfg.text)),
        image_token_id=cfg.image_token_id,
        pad_token_id=cfg.pad_token_id,
        ignore_index=cfg.ignore_index,
    )
