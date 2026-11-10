"""Generate text with a MiniLLM checkpoint (fp32/int8/int4), with KV cache.

    python -m minillm.generate --ckpt runs/minillm/model.pt \
        --tokenizer runs/minillm/tokenizer.json --prompt "介绍一下你自己"

    # raw completion, no chat template, cache disabled
    python -m minillm.generate ... --raw --no-cache --max-new-tokens 64

Checkpoints from the old vision-language model are rejected explicitly.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from .config import ARCHITECTURE, MiniLLMConfig
from .model import MiniLLM
from .quantize import dequantize_model
from .tokenizer import DEFAULT_SYSTEM, MiniLLMTokenizer
from .utils import get_device, set_seed


def load_any_checkpoint(path: str | Path, device: torch.device) -> tuple[MiniLLM, dict]:
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or payload.get("architecture") != ARCHITECTURE:
        raise SystemExit(
            f"{path}: not a MiniLLM checkpoint (architecture={payload.get('architecture') if isinstance(payload, dict) else None!r}). "
            "TinyVLM vision-language checkpoints cannot be loaded into MiniLLM."
        )
    if "quantized" in payload:
        model = dequantize_model(payload)
    else:
        if "config" not in payload or "model" not in payload:
            raise SystemExit(f"{path}: checkpoint is missing 'config' or 'model'")
        model = MiniLLM(MiniLLMConfig.from_dict(payload["config"]))
        model.load_state_dict(payload["model"])
    return model.to(device).eval(), payload


def main() -> None:
    p = argparse.ArgumentParser(description="Generate with MiniLLM")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--prompt", default="", help="instruction; empty starts an interactive loop")
    p.add_argument("--system", default=DEFAULT_SYSTEM)
    p.add_argument("--raw", action="store_true", help="treat --prompt as raw completion text")
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top-k", type=int, default=0)
    p.add_argument("--no-cache", action="store_true")
    p.add_argument("--device", default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--interactive", action="store_true")
    args = p.parse_args()

    device = get_device(args.device)
    set_seed(args.seed)
    model, payload = load_any_checkpoint(args.ckpt, device)
    tokenizer = MiniLLMTokenizer.load(args.tokenizer)
    tokenizer.assert_vocab_size(model.cfg.vocab_size)
    if payload.get("tokenizer_fingerprint") not in (None, tokenizer.fingerprint):
        raise SystemExit("checkpoint and tokenizer do not come from the same tokenizer")
    print(f"[model] {model.num_parameters():,} params on {device}, "
          f"{'int' + str(payload.get('quantized_bits')) if 'quantized' in payload else 'fp32'} checkpoint")

    def run(text: str) -> None:
        if args.raw:
            ids = tokenizer.encode(text)
        else:
            ids = tokenizer.apply_chat_template(
                [{"role": "user", "content": text}], system=args.system, add_generation_prompt=True
            )
        input_ids = torch.tensor([ids], dtype=torch.long, device=device)
        if input_ids.shape[1] + args.max_new_tokens > model.cfg.max_position_embeddings:
            args.max_new_tokens = max(model.cfg.max_position_embeddings - input_ids.shape[1], 1)
            print(f"[warn] clamped max_new_tokens to {args.max_new_tokens} to fit the context")
        started = time.time()
        output = model.generate(
            input_ids, max_new_tokens=args.max_new_tokens, temperature=args.temperature,
            top_k=args.top_k, eos_token_id=model.cfg.eos_token_id, use_cache=not args.no_cache,
        )
        elapsed = time.time() - started
        new_tokens = output[0, input_ids.shape[1] :]
        text_out = tokenizer.decode(new_tokens.tolist(), skip_special_tokens=True)
        rate = new_tokens.numel() / max(elapsed, 1e-6)
        print(f"[prompt] {text}")
        print(f"[reply] {text_out.strip()}")
        print(f"[stats] {new_tokens.numel()} tokens in {elapsed:.2f}s ({rate:.1f} tok/s, "
              f"cache={not args.no_cache})")

    if args.prompt:
        run(args.prompt)
        return
    if not args.interactive:
        raise SystemExit("pass --prompt or --interactive")
    print("Interactive mode. Empty line quits.")
    while True:
        try:
            line = input("user> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not line:
            break
        run(line)


if __name__ == "__main__":
    main()
