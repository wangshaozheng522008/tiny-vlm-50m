"""Generate text from text, an image, or an image with a text prompt.

    python -m tiny_vlm.generate --ckpt runs/tiny-vlm/model-int8.pt \
        --image cat.jpg --prompt "Describe this image:"
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch

from .config import VLMConfig
from .data import build_prompt, build_transform, load_image
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
    model.tokenizer_fingerprint = payload.get("tokenizer_fingerprint")
    return model.to(device).eval(), cfg


def validate_tokenizer(model: TinyVLM, tokenizer: TinyTokenizer) -> None:
    if tokenizer.vocab_size != model.cfg.text.vocab_size:
        raise ValueError("tokenizer vocabulary size differs from model")
    fingerprint = getattr(model, "tokenizer_fingerprint", None)
    if fingerprint is not None and fingerprint != tokenizer.fingerprint:
        raise ValueError("tokenizer vocabulary or merge rules differ from checkpoint")


def main() -> None:
    p = argparse.ArgumentParser(description="Generate from text, an image, or both")
    p.add_argument("--ckpt", default="runs/tiny-vlm/model.pt")
    p.add_argument("--tokenizer", default="runs/tiny-vlm/tokenizer.json")
    p.add_argument("--image", help="optional image path; omit for text-only generation")
    p.add_argument("--prompt", default="", help="optional text; image-only input uses BOS without a prompt")
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--temperature", type=float, default=0.0, help="0 = greedy")
    p.add_argument("--top-k", type=int, default=50)
    p.add_argument("--device", default=None)
    args = p.parse_args()
    if not args.image and not args.prompt.strip():
        p.error("supply --prompt, --image, or both")
    if args.max_new_tokens < 0 or args.top_k < 0 or not math.isfinite(args.temperature) or args.temperature < 0:
        p.error("max-new-tokens, top-k, and temperature must be nonnegative")

    device = get_device(args.device)
    model, cfg = load_model(args.ckpt, device)
    tokenizer = TinyTokenizer.load(args.tokenizer)

    try:
        validate_tokenizer(model, tokenizer)
    except ValueError as exc:
        p.error(str(exc))

    pixel_values = None
    if args.image:
        pixel_values = build_transform(cfg.vision.img_size)(load_image(args.image)).unsqueeze(0).to(device)
    input_ids = torch.tensor([build_prompt(tokenizer, args.prompt)], device=device)
    if input_ids.shape[1] >= cfg.text.max_seq_len:
        p.error(f"prompt must use fewer than {cfg.text.max_seq_len} tokens to leave room for generation")

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
