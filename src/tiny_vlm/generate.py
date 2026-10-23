"""Run inference on a single image + text prompt.

    python -m tiny_vlm.generate --ckpt runs/tiny-vlm/model-int8.pt \
        --image cat.jpg --prompt "Describe this image:"
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .config import VLMConfig
from .data import build_transform, load_image
from .model import TinyVLM
from .tokenizer import EOS_ID, TinyTokenizer
from .utils import get_device


def load_model(ckpt_path: str | Path, device: torch.device) -> tuple[TinyVLM, VLMConfig]:
    payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if "quantized" in payload:
        from .quantize import dequantize_model

        cfg = payload["config"]
        model = dequantize_model(payload, cfg)
    else:
        cfg = payload["config"]
        model = TinyVLM(cfg)
        model.load_state_dict(payload["model"])
    return model.to(device).eval(), cfg


def main() -> None:
    p = argparse.ArgumentParser(description="Generate a caption / answer for an image")
    p.add_argument("--ckpt", default="runs/tiny-vlm/model.pt")
    p.add_argument("--tokenizer", default="runs/tiny-vlm/tokenizer.json")
    p.add_argument("--image", required=True)
    p.add_argument("--prompt", default="Describe this image:")
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--temperature", type=float, default=0.0, help="0 = greedy")
    p.add_argument("--top-k", type=int, default=50)
    p.add_argument("--device", default=None)
    args = p.parse_args()

    device = get_device(args.device)
    model, cfg = load_model(args.ckpt, device)
    tokenizer = TinyTokenizer.load(args.tokenizer)

    if tokenizer.vocab_size != cfg.text.vocab_size:
        raise SystemExit(
            f"tokenizer vocab ({tokenizer.vocab_size}) != model vocab ({cfg.text.vocab_size})"
        )

    pixel_values = (
        build_transform(cfg.vision.img_size)(load_image(args.image)).unsqueeze(0).to(device)
    )
    input_ids = torch.tensor([tokenizer.encode(args.prompt, add_special_tokens=True)], device=device)

    out = model.generate(
        input_ids,
        pixel_values=pixel_values,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        eos_token_id=EOS_ID,
    )
    new_ids = out[0, input_ids.shape[1] :].tolist()
    print(tokenizer.decode(new_ids, skip_special_tokens=True).strip())


if __name__ == "__main__":
    main()