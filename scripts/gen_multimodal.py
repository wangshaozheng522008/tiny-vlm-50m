#!/usr/bin/env python3
"""Generate reproducible Chinese/English text, image and visual-question datasets.

Labels come from arithmetic/string operations or the rendered scene, with no network
or external model dependency. Images and text prompts are disjoint across splits.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import string
from collections import Counter
from pathlib import Path

from PIL import Image, ImageDraw

COLORS = {
    "red": ("红色", (220, 40, 40)),
    "green": ("绿色", (40, 170, 60)),
    "blue": ("蓝色", (40, 70, 220)),
}
SHAPES = {"circle": "圆形", "square": "正方形", "triangle": "三角形"}
QUESTIONS = {
    "zh": ["图中的图形是什么颜色？", "图中是什么形状？", "图中有几个图形？只回答数字。"],
    "en": ["What color are the shapes?", "What shape is shown?", "How many shapes are there? Answer with a number."],
}


def text_example(rng, task, language):
    a, b = rng.randrange(100), rng.randrange(100)
    zh = language == "zh"
    if task == 0:
        prompt = f"计算 {a} + {b}，只回答结果。" if zh else f"Calculate {a} + {b}. Answer with the result."
        answer = str(a + b)
    elif task == 1:
        a, b = max(a, b), min(a, b)
        prompt = f"计算 {a} - {b}，只回答结果。" if zh else f"Calculate {a} - {b}. Answer with the result."
        answer = str(a - b)
    elif task == 2:
        prompt = f"{a}和{b}中哪个数更大？只回答数字。" if zh else f"Which is larger, {a} or {b}? Answer with the number."
        answer = str(max(a, b))
    else:
        word = "".join(rng.choices(string.ascii_lowercase, k=rng.randrange(3, 9)))
        if task == 3:
            prompt = f"原样输出：{word}" if zh else f"Repeat exactly: {word}"
            answer = word
        elif task == 4:
            prompt = f"字符串{word}有几个字母？只回答数字。" if zh else f"How many letters are in {word}? Answer with a number."
            answer = str(len(word))
        else:
            prompt = f"倒序输出字符串：{word}" if zh else f"Reverse this string: {word}"
            answer = word[::-1]
    return {"prompt": prompt, "answer": answer}


def render_scene(rng, size, color, shape, count):
    image = Image.new("RGB", (size, size), tuple(rng.randrange(240, 256) for _ in range(3)))
    draw = ImageDraw.Draw(image)
    cell = size / count
    for index in range(count):
        radius = int(min(cell * 0.31, size * 0.22) * rng.uniform(0.75, 1.0))
        cx = int(cell * (index + 0.5) + rng.uniform(-0.06, 0.06) * cell)
        cy = int(size * rng.uniform(0.38, 0.62))
        bounds = (cx - radius, cy - radius, cx + radius, cy + radius)
        fill = COLORS[color][1]
        if shape == "circle":
            draw.ellipse(bounds, fill=fill)
        elif shape == "square":
            draw.rectangle(bounds, fill=fill)
        else:
            draw.polygon([(cx, cy - radius), (cx - radius, cy + radius),
                          (cx + radius, cy + radius)], fill=fill)
    return image


def generate_dataset(out, train_scenes=200, val_scenes=40, test_scenes=40,
                     size=336, seed=42, language="both"):
    out = Path(out)
    languages = ("zh", "en") if language == "both" else (language,)
    image_language = "zh" if "zh" in languages else "en"
    image_hashes, text_prompts = set(), set()
    manifest = {"seed": seed, "size": size, "language": language,
                "image_caption_language": image_language, "splits": {}}
    for split_id, (split, scenes) in enumerate([
        ("train", train_scenes), ("val", val_scenes), ("test", test_scenes)
    ]):
        rng = random.Random(seed + split_id * 1_000_003)
        folder = out / "images" / split
        folder.mkdir(parents=True, exist_ok=True)
        records, hashes = [], []
        for index in range(scenes):
            color, shape, count = rng.choice(list(COLORS)), rng.choice(list(SHAPES)), rng.randint(1, 3)
            while True:
                image = render_scene(rng, size, color, shape, count)
                digest = hashlib.sha256(image.tobytes()).hexdigest()
                if digest not in image_hashes:
                    image_hashes.add(digest)
                    hashes.append(digest)
                    break
            relative = f"images/{split}/{index:06d}.png"
            image.save(out / relative)
            if image_language == "zh":
                caption = f"{count}个{COLORS[color][0]}{SHAPES[shape]}。"
            else:
                caption = f"{count} {color} {shape}{'s' if count > 1 else ''}."
            records.append({"image": relative, "answer": caption})
            for lang in languages:
                answers = [COLORS[color][0], SHAPES[shape], str(count)] if lang == "zh" else [color, shape, str(count)]
                for question, answer in zip(QUESTIONS[lang], answers):
                    records.append({"image": relative, "prompt": question, "answer": answer})
            for task in range(6):
                for attempt in range(1000):
                    sample = text_example(rng, task, rng.choice(languages))
                    if sample["prompt"] not in text_prompts:
                        text_prompts.add(sample["prompt"])
                        records.append(sample)
                        break
                else:
                    raise ValueError("too many unique text tasks requested; reduce scene counts or expand task templates")
        rng.shuffle(records)
        path = out / f"{split}.jsonl"
        path.write_text("".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
                        encoding="utf-8")
        counts = Counter("text" if "image" not in r else ("mixed" if r.get("prompt") else "image") for r in records)
        manifest["splits"][split] = {"scenes": scenes, "samples": len(records),
                                     "modes": dict(counts), "image_hashes": hashes}
    (out / "meta.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="data/multimodal")
    parser.add_argument("--train-scenes", type=int, default=200)
    parser.add_argument("--val-scenes", type=int, default=40)
    parser.add_argument("--test-scenes", type=int, default=40)
    parser.add_argument("--size", type=int, default=336)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--language", choices=["zh", "en", "both"], default="both")
    args = parser.parse_args()
    if min(args.train_scenes, args.val_scenes, args.test_scenes) <= 0 or args.size < 32:
        parser.error("scene counts must be positive and image size must be at least 32")
    meta = generate_dataset(args.out, args.train_scenes, args.val_scenes, args.test_scenes,
                            args.size, args.seed, args.language)
    for split, details in meta["splits"].items():
        print(f"{split}: {details['scenes']} images, {details['samples']} samples, {details['modes']}")
    print(f"saved to {args.out}; exact image content and text prompts are disjoint across splits")


if __name__ == "__main__":
    main()
