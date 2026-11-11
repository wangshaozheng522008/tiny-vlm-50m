"""Train the MiniLLM tokenizer on a Chinese-first, weighted corpus mix.

The corpus files passed here must be *training* data. Pass the validation/test
files to ``--forbid`` as a guard: if any of them appears in the corpus, the
script refuses to run, which is what keeps the tokenizer from memorising the
held-out splits.

    python scripts/train_tokenizer.py \
        --zh data/corpus/train.text.jsonl \
        --en data/en/train.txt --code data/code/train.jsonl \
        --weights 0.8 0.15 0.05 --max-chars 30000000 \
        --vocab-size 24576 --out data/corpus/tokenizer.json \
        --forbid data/corpus/val.text.jsonl data/corpus/test.text.jsonl \
        --eval-zh data/corpus/val.text.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

from minillm.data import stream_documents
from minillm.tokenizer import MiniLLMTokenizer, normalise_text

CATEGORIES = ("zh", "en", "code")


def build_sample(paths: list[Path], limit_chars: int = 200_000) -> list[str]:
    """A small held-out sample used only for efficiency metrics."""
    sample: list[str] = []
    chars = 0
    for path in paths:
        for text in stream_documents([path]):
            text = normalise_text(text)
            if len(text) < 20:
                continue
            sample.append(text)
            chars += len(text)
            if chars >= limit_chars:
                return sample
    return sample


def weighted_documents(
    pools: dict[str, list[Path]],
    weights: dict[str, float],
    max_chars: int,
    seed: int,
    skip_below: int = 20,
):
    """Yield documents from several corpora, sampling categories by weight.

    Each category is consumed as a stream; when one runs out its weight is
    dropped and the remaining weights are renormalised, so a small English or
    code corpus still contributes its full share before the run ends.
    """
    rng = random.Random(seed)
    iterators = {name: stream_documents(paths) for name, paths in pools.items() if paths}
    remaining = {name: weights.get(name, 0.0) for name in iterators}
    chars = 0
    docs = 0
    used = {name: 0 for name in iterators}
    while iterators:
        names = [n for n in iterators if remaining[n] > 0]
        if not names:
            names = list(iterators)
        total = sum(remaining[n] for n in names)
        pick = rng.random() * total
        chosen = names[-1]
        running = 0.0
        for name in names:
            running += remaining[name]
            if pick <= running:
                chosen = name
                break
        try:
            text = next(iterators[chosen])
        except StopIteration:
            iterators.pop(chosen)
            remaining.pop(chosen, None)
            continue
        if len(text) < skip_below:
            continue
        text = normalise_text(text)
        if not text:
            continue
        chars += len(text)
        docs += 1
        used[chosen] += 1
        yield text
        if chars >= max_chars:
            break
    weighted_documents.stats = {"docs": docs, "chars": chars, "per_category": used}  # type: ignore[attr-defined]


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the MiniLLM tokenizer")
    parser.add_argument("--zh", nargs="+", default=[], help="Chinese training files")
    parser.add_argument("--en", nargs="+", default=[], help="English training files")
    parser.add_argument("--code", nargs="+", default=[], help="code training files")
    parser.add_argument("--weights", nargs=3, type=float, default=[0.8, 0.15, 0.05],
                        metavar=("ZH", "EN", "CODE"))
    parser.add_argument("--vocab-size", type=int, default=24576)
    parser.add_argument("--max-chars", type=int, default=30_000_000)
    parser.add_argument("--out", required=True)
    parser.add_argument("--report", help="optional JSON report path")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--forbid", nargs="*", default=[],
                        help="held-out files that must never be used for training")
    parser.add_argument("--eval-zh", nargs="*", default=[])
    parser.add_argument("--eval-en", nargs="*", default=[])
    parser.add_argument("--eval-code", nargs="*", default=[])
    args = parser.parse_args()

    pools = {name: [Path(p) for p in getattr(args, name)] for name in CATEGORIES}
    if not any(pools.values()):
        raise SystemExit("no training corpus given; use --zh/--en/--code")

    forbidden = {Path(p).resolve() for p in args.forbid}
    used_paths = {p.resolve() for paths in pools.values() for p in paths}
    overlap = used_paths & forbidden
    if overlap:
        raise SystemExit(f"refusing to train the tokenizer on held-out data: {sorted(map(str, overlap))}")

    weights = dict(zip(CATEGORIES, args.weights))
    if abs(sum(weights.values()) - 1.0) > 1e-6:
        raise SystemExit("--weights must sum to 1.0")
    for name, paths in pools.items():
        for path in paths:
            if not path.exists():
                raise SystemExit(f"missing corpus file: {path}")

    started = time.time()
    print(f"training tokenizer: vocab={args.vocab_size}, weights={weights}, "
          f"max_chars={args.max_chars:,}")
    documents = weighted_documents(pools, weights, args.max_chars, args.seed)
    tokenizer = MiniLLMTokenizer.train(documents, vocab_size=args.vocab_size)
    elapsed = time.time() - started
    tokenizer.save(args.out)
    stats = getattr(weighted_documents, "stats", {})
    print(f"  vocab={tokenizer.vocab_size}, fingerprint={tokenizer.fingerprint[:16]}…, "
          f"{stats.get('chars', 0):,} chars in {elapsed:.0f}s")

    report = {
        "vocab_size": tokenizer.vocab_size,
        "fingerprint": tokenizer.fingerprint,
        "special_token_ids": tokenizer.special_ids,
        "corpus": {"weights": weights, "paths": {k: [str(p) for p in v] for k, v in pools.items()}},
        "training": {**stats, "seed": args.seed, "max_chars": args.max_chars, "seconds": round(elapsed, 1)},
    }
    samples = {
        "zh": build_sample([Path(p) for p in args.eval_zh]) if args.eval_zh else [],
        "en": build_sample([Path(p) for p in args.eval_en]) if args.eval_en else [],
        "code": build_sample([Path(p) for p in args.eval_code]) if args.eval_code else [],
    }
    samples = {k: v for k, v in samples.items() if v}
    if samples:
        report["efficiency"] = tokenizer.efficiency_report(samples)
        print(f"  efficiency: {report['efficiency']}")
    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
