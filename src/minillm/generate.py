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
from typing import Sequence

import torch

from .config import ARCHITECTURE, MiniLLMConfig
from .model import MiniLLM
from .quantize import dequantize_model
from .tokenizer import DEFAULT_SYSTEM, END_OF_TURN_ID, EOS_ID, MiniLLMTokenizer
from .utils import get_device, set_seed
from .text_policy import checkpoint_text_policy


def stop_ids_for(raw: bool) -> tuple[int, ...]:
    """Chat turns end with `<|end_of_turn|>`; raw completions keep plain EOS."""
    return (EOS_ID,) if raw else (EOS_ID, END_OF_TURN_ID)


def truncate_at_stop(tokens: Sequence[int], stop_ids: Sequence[int]) -> list[int]:
    """Drop the first stop marker (and anything after it) from generated ids."""
    stop = set(stop_ids)
    for index, token in enumerate(tokens):
        if token in stop:
            return list(tokens[:index])
    return list(tokens)


def build_reply(
    model: MiniLLM,
    tokenizer: MiniLLMTokenizer,
    text: str,
    *,
    raw: bool = False,
    system: str = DEFAULT_SYSTEM,
    max_new_tokens: int = 128,
    temperature: float = 0.0,
    top_k: int = 0,
    use_cache: bool = True,
) -> dict:
    """Tokenise a prompt, generate, cut at the stop marker and decode."""
    if raw:
        ids = tokenizer.encode(text)
    else:
        ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": text}], system=system, add_generation_prompt=True
        )
    input_ids = torch.tensor([ids], dtype=torch.long, device=next(model.parameters()).device)
    stop_ids = stop_ids_for(raw)
    max_new_tokens = min(max_new_tokens, max(model.cfg.max_position_embeddings - input_ids.shape[1], 1))
    output = model.generate(
        input_ids, max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k,
        eos_token_id=model.cfg.eos_token_id, stop_token_ids=stop_ids, use_cache=use_cache,
    )
    new_tokens = output[0, input_ids.shape[1]:].tolist()
    kept = truncate_at_stop(new_tokens, stop_ids)
    return {
        "text": tokenizer.decode(kept, skip_special_tokens=True),
        "tokens": len(kept),
        "generated": len(new_tokens),
        "stop_ids": stop_ids,
    }


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
    tokenizer.text_policy = checkpoint_text_policy(payload)
    tokenizer.assert_vocab_size(model.cfg.vocab_size)
    if payload.get("tokenizer_fingerprint") not in (None, tokenizer.fingerprint):
        raise SystemExit("checkpoint and tokenizer do not come from the same tokenizer")
    print(f"[model] {model.num_parameters():,} params on {device}, "
          f"{'int' + str(payload.get('quantized_bits')) if 'quantized' in payload else 'fp32'} checkpoint")

    def run(text: str) -> None:
        started = time.time()
        result = build_reply(
            model, tokenizer, text, raw=args.raw, system=args.system,
            max_new_tokens=args.max_new_tokens, temperature=args.temperature,
            top_k=args.top_k, use_cache=not args.no_cache,
        )
        elapsed = time.time() - started
        rate = result["tokens"] / max(elapsed, 1e-6)
        print(f"[prompt] {text}")
        print(f"[reply] {result['text'].strip()}")
        print(f"[stats] {result['tokens']} tokens in {elapsed:.2f}s ({rate:.1f} tok/s, "
              f"cache={not args.no_cache}, stop={result['stop_ids']})")

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
