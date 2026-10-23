#!/usr/bin/env python3
"""Generate a synthetic captioning dataset of coloured shapes.

The point of a synthetic dataset is that the caption is *derivable* from the image, so
"did the model learn" is measurable rather than inferred from a falling loss. A fixed set
of (colour, shape) pairs is held out of training entirely and used to build an unseen-combination
split, which asks the interesting question for a small VLM: given it has seen `purple` and
`star` separately, can it describe a purple star it has never seen together?

Layout::

    data/shapes/
      images/000000.png ...
      train.jsonl          pairs from the training combination set
      test_seen.jsonl      held-out images, pairs seen during training
      test_unseen.jsonl    held-out images, pairs never seen together
      captions.txt         every training caption, for BPE training
      meta.json            the split definitions, so evaluation can reuse them

    python scripts/gen_shapes.py --out data/shapes --train 2000 --test 300
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw

COLORS: dict[str, tuple[int, int, int]] = {
    "red": (214, 39, 40),
    "blue": (31, 78, 201),
    "green": (44, 160, 44),
    "yellow": (230, 190, 30),
    "purple": (140, 60, 200),
    "orange": (235, 125, 25),
    "pink": (235, 110, 175),
    "black": (35, 35, 35),
}

SHAPES = ["circle", "square", "triangle", "star", "diamond", "cross", "oval", "hexagon"]

PLURALS = {
    "circle": "circles",
    "square": "squares",
    "triangle": "triangles",
    "star": "stars",
    "diamond": "diamonds",
    "cross": "crosses",
    "oval": "ovals",
    "hexagon": "hexagons",
}

COUNT_WORDS = {1: "a", 2: "two", 3: "three", 4: "four"}

BACKGROUNDS = [(255, 255, 255), (245, 245, 240), (238, 242, 248), (250, 245, 238)]


def polygon(
    cx: float, cy: float, r: float, n: int, rotation: float, inner: float | None = None
) -> list[tuple[float, float]]:
    """Regular n-gon, or a star when `inner` is given (alternating radii)."""
    points = []
    steps = n * 2 if inner else n
    for i in range(steps):
        radius = r if (inner is None or i % 2 == 0) else r * inner
        angle = rotation + 2 * math.pi * i / steps
        points.append((cx + radius * math.cos(angle), cy + radius * math.sin(angle)))
    return points


def shape_points(name: str, cx: float, cy: float, r: float, rng: random.Random):
    """Vertex list for a shape. Square and diamond are left unrotated so they stay distinct."""
    if name == "circle":
        return polygon(cx, cy, r, 64, 0.0)
    if name == "oval":
        # Elongated circle: scale x after building the polygon.
        pts = polygon(cx, cy, r, 64, 0.0)
        return [(cx + (x - cx) * 1.45, y) for x, y in pts]
    if name == "square":
        return polygon(cx, cy, r * 0.92, 4, math.pi / 4)
    if name == "diamond":
        return polygon(cx, cy, r * 1.1, 4, 0.0)
    if name == "triangle":
        return polygon(cx, cy, r * 1.05, 3, math.pi / 2 + rng.uniform(-0.3, 0.3))
    if name == "hexagon":
        return polygon(cx, cy, r, 6, rng.uniform(0, math.pi / 3))
    if name == "star":
        return polygon(cx, cy, r * 1.05, 5, math.pi / 2, inner=0.45)
    if name == "cross":
        arm, thick = r, r * 0.34
        return [
            (cx - thick, cy - arm), (cx + thick, cy - arm), (cx + thick, cy - thick),
            (cx + arm, cy - thick), (cx + arm, cy + thick), (cx + thick, cy + thick),
            (cx + thick, cy + arm), (cx - thick, cy + arm), (cx - thick, cy + thick),
            (cx - arm, cy + thick), (cx - arm, cy - thick), (cx - thick, cy - thick),
        ]
    raise ValueError(f"unknown shape {name!r}")


@dataclass
class Placement:
    color: str
    shape: str
    cx: float
    cy: float
    r: float


def place(rng: random.Random, size: int, pairs: list[tuple[str, str]]) -> list[Placement]:
    """Scatter one shape per pair, rejection-sampling to keep them from overlapping."""
    margin = 0.12 * size
    placed: list[Placement] = []
    for color, shape in pairs:
        for _ in range(200):
            r = rng.uniform(0.075, 0.13) * size
            cx = rng.uniform(margin + r, size - margin - r)
            cy = rng.uniform(margin + r, size - margin - r)
            if all(math.hypot(cx - p.cx, cy - p.cy) > (r + p.r) * 1.12 for p in placed):
                placed.append(Placement(color, shape, cx, cy, r))
                break
        else:
            # Give up on non-overlap rather than loop forever; jittered retry keeps the
            # distribution sane even for crowded images.
            r = rng.uniform(0.06, 0.10) * size
            placed.append(
                Placement(color, shape, rng.uniform(r, size - r), rng.uniform(r, size - r), r)
            )
    return placed


def render(size: int, placements: list[Placement], rng: random.Random) -> Image.Image:
    img = Image.new("RGB", (size, size), rng.choice(BACKGROUNDS))
    draw = ImageDraw.Draw(img)
    for p in placements:
        draw.polygon(shape_points(p.shape, p.cx, p.cy, p.r, rng), fill=COLORS[p.color])
    return img


def article(color: str) -> str:
    return "an" if color[0] in "aeiou" else "a"


def caption_for(counts: dict[tuple[str, str], int]) -> str:
    parts = []
    for (color, shape) in sorted(counts, key=lambda cs: (cs[1], cs[0])):
        n = counts[(color, shape)]
        noun = shape if n == 1 else PLURALS[shape]
        word = article(color) if n == 1 else COUNT_WORDS[n]
        parts.append(f"{word} {color} {noun}")
    return " and ".join(parts)


def sample_counts(rng: random.Random, pool: list[tuple[str, str]]) -> dict[tuple[str, str], int]:
    """1-3 distinct (colour, shape) pairs, each repeated 1-3 times, 1-5 shapes total."""
    k = rng.choices([1, 2, 3], weights=[0.35, 0.45, 0.20])[0]
    k = min(k, len(pool))
    selected = rng.sample(pool, k)
    counts: dict[tuple[str, str], int] = {}
    budget = 5
    for i, pair in enumerate(selected):
        max_n = min(3, budget - (len(selected) - i - 1))
        if max_n < 1:
            break
        n = rng.choices(range(1, max_n + 1), weights=[0.6, 0.3, 0.1][:max_n])[0]
        counts[pair] = n
        budget -= n
    return counts


def build_split(
    rng: random.Random,
    pool: list[tuple[str, str]],
    n: int,
    size: int,
    out_dir: Path,
    records: list[dict],
    start_index: int,
) -> int:
    idx = start_index
    for _ in range(n):
        counts = sample_counts(rng, pool)
        if not counts:
            continue
        pairs = [pair for pair, c in counts.items() for _ in range(c)]
        placements = place(rng, size, pairs)
        img = render(size, placements, rng)
        rel = f"images/{idx:06d}.png"
        img.save(out_dir / rel)
        records.append({"image": rel, "caption": caption_for(counts)})
        idx += 1
    return idx


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out", default="data/shapes")
    p.add_argument("--size", type=int, default=336, help="image side length")
    p.add_argument("--train", type=int, default=2000)
    p.add_argument("--test", type=int, default=300)
    p.add_argument("--holdout-pairs", type=int, default=10, help="(colour, shape) pairs kept out of training")
    p.add_argument("--seed", type=int, default=1234)
    args = p.parse_args()

    out = Path(args.out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    all_pairs = [(c, s) for s in SHAPES for c in COLORS]
    holdout = set(rng.sample(all_pairs, args.holdout_pairs))
    train_pool = [p for p in all_pairs if p not in holdout]
    holdout_pool = sorted(holdout)

    def write(name: str, records: list[dict]) -> None:
        with (out / name).open("w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    train: list[dict] = []
    idx = build_split(rng, train_pool, args.train, args.size, out, train, 0)
    write("train.jsonl", train)

    seen: list[dict] = []
    idx = build_split(rng, train_pool, args.test, args.size, out, seen, idx)
    write("test_seen.jsonl", seen)

    # Must continue from the updated index, or the two test splits overwrite each other's
    # image files while both keep referencing them.
    unseen: list[dict] = []
    build_split(rng, holdout_pool, args.test, args.size, out, unseen, idx)
    write("test_unseen.jsonl", unseen)

    (out / "captions.txt").write_text(
        "\n".join(rec["caption"] for rec in train) + "\n", encoding="utf-8"
    )
    (out / "meta.json").write_text(
        json.dumps(
            {
                "size": args.size,
                "seed": args.seed,
                "colors": list(COLORS),
                "shapes": SHAPES,
                "holdout_pairs": [list(p) for p in holdout_pool],
                "counts": {
                    "train": len(train),
                    "test_seen": len(seen),
                    "test_unseen": len(unseen),
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"train       {len(train):5d} images")
    print(f"test_seen   {len(seen):5d} images")
    print(f"test_unseen {len(unseen):5d} images")
    print(f"held out    {len(holdout_pool)} pairs: {', '.join(f'{c} {s}' for c, s in holdout_pool)}")
    print(f"vocab       {len(COLORS)} colours x {len(SHAPES)} shapes, counts 1-4")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
