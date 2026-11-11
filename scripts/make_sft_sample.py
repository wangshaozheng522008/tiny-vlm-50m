#!/usr/bin/env python3
"""Build a small extractive instruction dataset for SFT demos.

This is deliberately simple and honest: it turns plain text into
"summarise / continue / shorten" conversations whose answers are excerpts of the
same document. It exists so the SFT pipeline can be exercised end-to-end without
a proprietary instruction corpus — not as a substitute for one.

    python scripts/make_sft_sample.py --text runs/minillm-87m/corpus/train.text.jsonl \
        --out runs/minillm-87m/sft --count 2000
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from minillm.data import stream_documents  # noqa: E402

SYSTEM = "你是一个有帮助的中文助手。"
SENTENCE_RE = re.compile(r"[^。！？!?\n]{8,}[。！？!?]")
TEMPLATES = [
    ("请用一句话概括下面的内容：\n{text}", "概括："),
    ("请续写下面的句子：\n{head}", "续写："),
    ("把下面的内容改写得更简洁：\n{text}", "简洁版："),
]


def first_sentence(text: str) -> str | None:
    match = SENTENCE_RE.search(text)
    return match.group(0).strip() if match else None


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--text", nargs="+", required=True, help='JSONL {"text": ...}')
    p.add_argument("--out", required=True, help="output directory for train.jsonl/val.jsonl")
    p.add_argument("--count", type=int, default=2000)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--min-chars", type=int, default=160)
    args = p.parse_args()

    rng = random.Random(args.seed)
    records = []
    for text in stream_documents([Path(t) for t in args.text]):
        text = " ".join(text.split())
        if len(text) < args.min_chars:
            continue
        sentence = first_sentence(text)
        if not sentence or len(sentence) < 20:
            continue
        template = TEMPLATES[len(records) % len(TEMPLATES)]
        if template[1] == "续写：":
            head = text[: max(len(text) // 3, 40)]
            question = template[0].format(head=head)
            answer = text[len(head) : len(head) + 80].strip()
        else:
            question = template[0].format(text=text[:400])
            answer = sentence[:200]
        if len(answer) < 10:
            continue
        records.append({
            "messages": [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer},
            ]
        })
        if len(records) >= args.count:
            break
    if not records:
        raise SystemExit("no instruction records could be built; lower --min-chars")

    rng.shuffle(records)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    val_count = max(1, int(len(records) * 0.05))
    for name, subset in (("train", records[val_count:]), ("val", records[:val_count])):
        path = out / f"{name}.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for record in subset:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"{name}: {len(subset)} records -> {path}")


if __name__ == "__main__":
    main()
