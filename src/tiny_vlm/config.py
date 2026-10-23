"""Model configuration for TinyVLM."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class VisionConfig:
    img_size: int = 336
    patch_size: int = 16
    in_channels: int = 3
    d_model: int = 448
    depth: int = 8
    num_heads: int = 7
    mlp_ratio: float = 4.0
    dropout: float = 0.0

    @property
    def num_patches(self) -> int:
        return (self.img_size // self.patch_size) ** 2


@dataclass
class TextConfig:
    vocab_size: int = 16384
    max_seq_len: int = 256
    d_model: int = 448
    depth: int = 8
    num_heads: int = 7
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


# Sizing is driven by "INT8 weight file < 50 MB", which works out to roughly 47M
# parameters once scales and the fp16 leftovers are accounted for.
# Measured: tiny-50m = 46.95M params -> 46.30 MB at INT8, 24.89 MB at INT4.
PRESETS: dict[str, VLMConfig] = {
    "tiny-50m": VLMConfig(
        vision=VisionConfig(img_size=336, patch_size=16, d_model=448, depth=8, num_heads=7),
        text=TextConfig(vocab_size=16384, max_seq_len=256, d_model=448, depth=8, num_heads=7),
    ),
    # Kept small so `scripts/smoke_test.sh` runs in seconds on CPU.
    "nano-50m": VLMConfig(
        vision=VisionConfig(img_size=224, patch_size=16, d_model=256, depth=4, num_heads=4),
        text=TextConfig(vocab_size=8192, max_seq_len=256, d_model=256, depth=4, num_heads=4),
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
