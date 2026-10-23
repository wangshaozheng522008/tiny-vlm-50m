#!/usr/bin/env python3
"""Score a checkpoint on the synthetic shapes splits.

A falling loss only says the model is fitting something. This parses the generated caption
back into `{(colour, shape): count}` triples and compares them piecewise, so the report
separates the abilities that actually matter here: does it get the colours right, the
shapes, the counts, and both together?

    python scripts/eval_shapes.py --ckpt runs/shapes/model.pt \
        --tokenizer runs/shapes/tokenizer.json --data data/shapes --limit 100
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from tiny_vlm.data import DEFAULT_PROMPT, build_transform, load_image
from tiny_vlm.generate import load_model
from tiny_vlm.tokenizer import EOS_ID, TinyTokenizer
from tiny_vlm.utils import get_device

COUNT_WORDS = {"a": 1, "an": 1, "two": 2, "three": 3, "four": 4}

PLURALS = {
    "circle": "circles", "square": "squares", "triangle": "triangles", "star": "stars",
    "diamond": "diamonds", "cross": "crosses", "oval": "ovals", "hexagon": "hexagons",
}
COLORS = {"red", "blue", "green", "yellow", "purple", "orange", "pink", "black"}

SINGULAR = {plural: single for single, plural in PLURALS.items()}
SINGULAR.update({single: single for single in PLURALS})


def parse_caption(text: str) -> dict[tuple[str, str], int] | None:
    """`"two red circles and a blue square"` -> `{("red","circle"): 2, ("blue","square"): 1}`.

    Returns None when the output does not follow the grammar, which is itself a result
    worth reporting rather than silently scoring as zero.
    """
    text = text.lower().strip().rstrip(".")
    if not text:
        return None
    parsed: dict[tuple[str, str], int] = {}
    for part in text.split(" and "):
        words = part.strip().split()
        if len(words) != 3:
            return None
        count_word, color, noun = words
        if count_word not in COUNT_WORDS or color not in COLORS or noun not in SINGULAR:
            return None
        shape = SINGULAR[noun]
        key = (color, shape)
        if key in parsed:
            return None
        parsed[key] = COUNT_WORDS[count_word]
    return parsed


def normalise(text: str) -> str:
    return " ".join(text.lower().split()).rstrip(".")


@torch.no_grad()
def evaluate(model, tokenizer, records, root: Path, transform, device, max_new_tokens: int):
    prompt_ids = tokenizer.encode(DEFAULT_PROMPT, add_special_tokens=True)
    input_ids = torch.tensor([prompt_ids], device=device)

    stats = {
        "n": 0, "exact": 0, "pairs": 0, "counts": 0, "malformed": 0,
        "ref_pairs": 0, "hit_pairs": 0, "count_correct": 0, "count_total": 0,
    }
    samples = []
    for rec in records:
        pixels = transform(load_image(root / rec["image"])).unsqueeze(0).to(device)
        out = model.generate(
            input_ids, pixel_values=pixels, max_new_tokens=max_new_tokens,
            temperature=0.0, eos_token_id=EOS_ID,
        )
        pred_text = tokenizer.decode(out[0, input_ids.shape[1]:].tolist(), skip_special_tokens=True)

        ref = parse_caption(rec["caption"])
        pred = parse_caption(pred_text)

        stats["n"] += 1
        if normalise(pred_text) == normalise(rec["caption"]):
            stats["exact"] += 1
        if pred is None:
            stats["malformed"] += 1
            pred = {}
        if set(pred) == set(ref):
            stats["pairs"] += 1
        if pred == ref:
            stats["counts"] += 1

        for pair, want in ref.items():
            stats["count_total"] += 1
            stats["ref_pairs"] += 1
            if pair in pred:
                stats["hit_pairs"] += 1
                if pred[pair] == want:
                    stats["count_correct"] += 1

        if len(samples) < 8:
            samples.append((rec["caption"], pred_text.strip()))

    return stats, samples


def report(name: str, stats: dict) -> None:
    n = max(stats["n"], 1)
    pairs = max(stats["ref_pairs"], 1)
    count_total = max(stats["count_total"], 1)
    print(f"\n=== {name}  ({stats['n']} images) ===")
    print(f"  exact match        {stats['exact'] / n:6.1%}")
    print(f"  colour+shape set   {stats['pairs'] / n:6.1%}")
    print(f"  counts correct     {stats['counts'] / n:6.1%}")
    print(f"  pair recall        {stats['hit_pairs'] / pairs:6.1%}   (of {stats['ref_pairs']} reference pairs)")
    print(f"  count acc | found  {stats['count_correct'] / count_total:6.1%}")
    print(f"  malformed output   {stats['malformed'] / n:6.1%}")


def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate a checkpoint on the shapes splits")
    p.add_argument("--ckpt", default="runs/shapes/model.pt")
    p.add_argument("--tokenizer", default="runs/shapes/tokenizer.json")
    p.add_argument("--data", default="data/shapes")
    p.add_argument("--splits", default="test_seen,test_unseen")
    p.add_argument("--limit", type=int, default=100, help="images per split (0 = all)")
    p.add_argument("--max-new-tokens", type=int, default=32)
    p.add_argument("--device", default=None)
    args = p.parse_args()

    device = get_device(args.device)
    model, cfg = load_model(args.ckpt, device)
    tokenizer = TinyTokenizer.load(args.tokenizer)
    transform = build_transform(cfg.vision.img_size)
    root = Path(args.data)

    for split in [s.strip() for s in args.splits.split(",") if s.strip()]:
        path = root / f"{split}.jsonl"
        if not path.exists():
            print(f"[skip] {path} not found")
            continue
        records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if args.limit:
            records = records[: args.limit]
        stats, samples = evaluate(
            model, tokenizer, records, root, transform, device, args.max_new_tokens
        )
        report(split, stats)
        print("  samples:")
        for ref, got in samples:
            flag = "ok " if normalise(got) == normalise(ref) else "   "
            print(f"    {flag}ref={ref!r}")
            print(f"       got={got!r}")


if __name__ == "__main__":
    main()