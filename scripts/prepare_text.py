"""Build a token-chunked language-modelling dataset from raw ``{"text": ...}`` JSONL.

Pipeline (order matters):

1. stream every source line, normalise whitespace and quality-filter it,
2. drop exact and near duplicates,
3. reservoir-sample N raw texts uniformly from what survives,
4. split the *raw texts* before chunking, so no source document leaks between
   train/validation/test,
5. chunk each split into windows of at most ``max_seq_len - 2`` tokens, leaving
   room for BOS/EOS, and break only on UTF-8 character boundaries,
6. write ``train.jsonl`` / ``val.jsonl`` / ``test.jsonl`` plus ``dataset.json``.

The output records use the existing ``{"text": ...}`` format, so both
``MultimodalDataset`` and ``IndexedTextDataset`` can read them unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from tiny_vlm.tokenizer import TinyTokenizer

# Characters that are legal in normal prose: CJK, CJK punctuation, fullwidth
# forms, general punctuation, ASCII, tab and newline. Everything else (control
# bytes, private-use code points, broken surrogates, emoji) counts against the
# printable ratio.
CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f\ud800-\udfff]")
DISALLOWED_RE = re.compile(
    r"[^\u4e00-\u9fff\u3400-\u4dbf\u3000-\u303f\uff00-\uffef\u2000-\u206f"
    r"\n\t\x20-\x7e]"
)
CJK_RE = re.compile(r"[\u4e00-\u9fff\u3400-\u4dbf]")
LATIN_RE = re.compile(r"[A-Za-z]")
NEWLINE_RUN_RE = re.compile(r"\n{3,}")
SPACE_RUN_RE = re.compile(r"[ \t]{2,}")
REPEATED_CHAR_RE = re.compile(r"(.)\1{19,}")
REPEATED_PHRASE_RE = re.compile(r"(.{3,40}?)\1{3,}")
REPLACEMENT = "\ufffd"

MIN_PRINTABLE_RATIO = 0.98
MIN_LETTER_RATIO = 0.15


def normalise(text: str) -> str:
    text = (
        text.replace("\r\n", "\n")
        .replace("\r", "\n")
        .replace("\u00a0", " ")
        .replace("\u3000", " ")
    )
    text = NEWLINE_RUN_RE.sub("\n\n", text)
    text = SPACE_RUN_RE.sub(" ", text)
    return text.strip()


def quality_issue(text: str, min_chars: int, max_chars: int, deep: bool) -> str | None:
    """Return a rejection reason, or None when the text passes the filters."""
    length = len(text)
    if length < min_chars:
        return "too_short"
    if length > max_chars:
        return "too_long"
    if REPLACEMENT in text:
        return "replacement_char"
    if CONTROL_RE.search(text):
        return "control_char"
    if len(DISALLOWED_RE.findall(text)) / length > 1.0 - MIN_PRINTABLE_RATIO:
        return "low_printable_ratio"
    letters = len(CJK_RE.findall(text)) + len(LATIN_RE.findall(text))
    if letters / length < MIN_LETTER_RATIO:
        return "low_letter_ratio"
    if REPEATED_CHAR_RE.search(text):
        return "repeated_char"
    if len(set(text)) / length < 0.04:
        return "low_char_diversity"
    if deep:
        # These are slower, so they only run on the reservoir candidates.
        if REPEATED_PHRASE_RE.search(text):
            return "repeated_phrase"
        lines = [line.strip() for line in text.split("\n") if len(line.strip()) > 12]
        if len(lines) >= 6:
            top = Counter(lines).most_common(1)[0][1]
            if top / len(lines) > 0.5:
                return "repeated_line"
    return None


@dataclass
class SplitStats:
    texts: int = 0
    chunks: int = 0
    tokens: int = 0
    chars: int = 0
    dropped_duplicate_chunks: int = 0
    chunk_lengths: Counter = field(default_factory=Counter)

    def as_dict(self) -> dict:
        lengths = sorted(self.chunk_lengths.elements()) if self.chunks else []
        return {
            "texts": self.texts,
            "chunks": self.chunks,
            "chunk_tokens": self.tokens,
            "text_chars": self.chars,
            "dropped_duplicate_chunks": self.dropped_duplicate_chunks,
            "chars_per_token": round(self.chars / self.tokens, 4) if self.tokens else None,
            "chunks_per_text": round(self.chunks / self.texts, 3) if self.texts else None,
            "chunk_tokens_mean": round(sum(lengths) / len(lengths), 2) if lengths else None,
            "chunk_tokens_min": lengths[0] if lengths else None,
            "chunk_tokens_max": lengths[-1] if lengths else None,
        }


def clean_boundary(tokenizer: TinyTokenizer, ids: list[int], start: int, end: int) -> int:
    """Move a split point left until the left chunk decodes to valid UTF-8."""
    while end > start + 1:
        if REPLACEMENT not in tokenizer.decode(ids[start:end]):
            return end
        end -= 1
    return end


def chunk_text(tokenizer: TinyTokenizer, text: str, limit: int,
               min_tokens: int) -> list[str]:
    """Split ``text`` into nearly equal, character-aligned pieces of <= limit tokens.

    Windows are ``limit - 4`` tokens wide so that repairing a boundary (which
    can hand at most a four-byte character, and therefore at most four tokens,
    to the next window) still fits. A plain ``size`` window is used for every
    non-final piece; the final piece takes whatever remains and is re-windowed
    if earlier repairs made it too long.
    """
    ids = tokenizer.encode(text, add_special_tokens=False)
    if not ids:
        return []
    if len(ids) <= limit:
        return [text] if len(ids) >= min_tokens else []
    margin = 4
    size = math.ceil(len(ids) / math.ceil(len(ids) / max(limit - margin, 1)))
    groups: list[tuple[int, int]] = []
    start = 0
    while start < len(ids):
        end = len(ids) if len(ids) - start <= limit else start + size
        end = clean_boundary(tokenizer, ids, start, end)
        if end <= start:
            raise ValueError("could not find a UTF-8 safe chunk boundary")
        groups.append((start, end))
        start = end
    # Merge a tiny tail into the previous chunk when it still fits.
    if len(groups) > 1:
        tail_start, tail_end = groups[-1]
        prev_start, prev_end = groups[-2]
        if (tail_end - tail_start < min_tokens
                and tail_end - prev_start <= limit):
            groups[-2:] = [(prev_start, tail_end)]

    pieces: list[str] = []
    for start, end in groups:
        piece = tokenizer.decode(ids[start:end])
        reencoded = tokenizer.encode(piece, add_special_tokens=False)
        if len(reencoded) > limit:  # Defensive: byte-level round trips should be stable.
            raise ValueError(f"chunk re-encodes to {len(reencoded)} tokens > limit {limit}")
        if REPLACEMENT in piece:
            raise ValueError("chunk decoded to invalid UTF-8")
        if len(reencoded) < min_tokens:
            continue
        pieces.append(piece)
    return pieces


class Reservoir:
    """Uniform sampling of the filtered stream with a little headroom."""

    def __init__(self, size: int, seed: int):
        self.size = size
        self.rng = random.Random(seed)
        self.items: list[str] = []
        self.seen = 0

    def offer(self, item: str) -> None:
        self.seen += 1
        if self.size <= 0:
            self.items.append(item)
            return
        if len(self.items) < self.size:
            self.items.append(item)
            return
        index = self.rng.randrange(self.seen)
        if index < self.size:
            self.items[index] = item


def iter_source(path: Path):
    """Yield (raw_line_bytes, text) for a ``{"text": ...}`` JSONL file."""
    with path.open("rb") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                raw = json.loads(line.decode("utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError):
                yield line, None
                continue
            text = raw.get("text") if isinstance(raw, dict) else None
            if not isinstance(text, str) or not text.strip():
                yield line, None
                continue
            yield line, text


def write_split(path: Path, pieces: list[str]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for piece in pieces:
            handle.write(json.dumps({"text": piece}, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", nargs="+", required=True, help="raw {\"text\"} JSONL files")
    parser.add_argument("--out", required=True, help="output directory")
    parser.add_argument("--tokenizer", required=True, help="existing tokenizer.json (kept unchanged)")
    parser.add_argument("--sample", type=int, default=50000,
                        help="raw texts to keep after filtering; 0 = every eligible text")
    parser.add_argument("--max-seq-len", type=int, default=256,
                        help="model context; BOS/EOS are reserved inside it")
    parser.add_argument("--val-ratio", type=float, default=0.02)
    parser.add_argument("--test-ratio", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-chars", type=int, default=80)
    parser.add_argument("--max-chars", type=int, default=20000)
    parser.add_argument("--min-chunk-tokens", type=int, default=8)
    parser.add_argument("--limit-lines", type=int, default=0,
                        help="debug: stop after N source lines")
    args = parser.parse_args()

    if args.max_seq_len < 8:
        parser.error("max-seq-len must be at least 8")
    if args.sample < 0 or args.limit_lines < 0:
        parser.error("sample and limit-lines must be nonnegative")
    if not 0 <= args.val_ratio < 0.5 or not 0 <= args.test_ratio < 0.5:
        parser.error("val/test ratios must be in [0, 0.5)")
    limit = args.max_seq_len - 2  # BOS + EOS
    if args.min_chunk_tokens < 1 or args.min_chunk_tokens > limit:
        parser.error("min-chunk-tokens must be within [1, max-seq-len - 2]")

    tokenizer = TinyTokenizer.load(args.tokenizer)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Reservoir includes headroom so deep-quality rejections can be replaced.
    target = args.sample if args.sample else 0
    reservoir = Reservoir(math.ceil(target * 1.2) if target else 0, args.seed)
    exact_seen: set[bytes] = set()
    near_seen: set[bytes] = set()
    rejected: Counter = Counter()
    duplicates = Counter()
    sources = []
    scanned = 0
    started = time.time()

    for source in args.input:
        path = Path(source)
        digest = hashlib.sha256()
        byte_count = 0
        for raw_line, text in iter_source(path):
            digest.update(raw_line)
            byte_count += len(raw_line)
            scanned += 1
            if text is None:
                rejected["malformed"] += 1
                continue
            text = normalise(text)
            issue = quality_issue(text, args.min_chars, args.max_chars, deep=False)
            if issue:
                rejected[issue] += 1
                continue
            encoded = text.encode("utf-8")
            exact = hashlib.blake2b(encoded, digest_size=8).digest()
            if exact in exact_seen:
                duplicates["exact"] += 1
                continue
            exact_seen.add(exact)
            # Same long prefix and same length is almost always a duplicated
            # document with a trailing edit, not a shared instruction template.
            near = hashlib.blake2b(
                (text[:160] + "\x00" + str(len(text))).encode("utf-8"), digest_size=8
            ).digest()
            if near in near_seen:
                duplicates["near"] += 1
                continue
            near_seen.add(near)
            reservoir.offer(text)
            if args.limit_lines and scanned >= args.limit_lines:
                break
        sources.append({"path": str(path), "bytes": byte_count,
                        "sha256": digest.hexdigest()})
        if args.limit_lines and scanned >= args.limit_lines:
            break
        print(f"[scan] {path}: {scanned} lines so far, {len(reservoir.items)} candidates",
              flush=True)

    if not reservoir.items:
        sys.exit("no text passed the filters; relax --min-chars or check the input")

    # Deep checks only run on the sampled candidates.
    kept: list[str] = []
    for text in reservoir.items:
        issue = quality_issue(text, args.min_chars, args.max_chars, deep=True)
        if issue:
            rejected[issue] += 1
            continue
        kept.append(text)
        if target and len(kept) >= target:
            break
    if target and len(kept) < target:
        print(f"[warn] only {len(kept)} of {target} requested texts passed every filter",
              flush=True)

    rng = random.Random(args.seed)
    rng.shuffle(kept)
    n = len(kept)
    n_test = int(round(n * args.test_ratio))
    n_val = int(round(n * args.val_ratio))
    n_train = n - n_val - n_test
    if n_train <= 0:
        sys.exit("not enough texts for a non-empty training split; lower the ratios")
    splits = {
        "train": kept[:n_train],
        "val": kept[n_train:n_train + n_val],
        "test": kept[n_train + n_val:],
    }

    stats = {name: SplitStats() for name in splits}
    output = {}
    seen_chunks: set[bytes] = set()
    for name, texts in splits.items():
        pieces: list[str] = []
        for text in texts:
            chunks = chunk_text(tokenizer, text, limit, args.min_chunk_tokens)
            stats[name].texts += 1
            stats[name].chars += len(text)
            for chunk in chunks:
                # Different source documents can still share a verbatim chunk
                # (templated openings). Train wins; the copy is dropped so no
                # chunk is ever identical across splits.
                digest = hashlib.blake2b(chunk.encode("utf-8"), digest_size=8).digest()
                if digest in seen_chunks:
                    stats[name].dropped_duplicate_chunks += 1
                    continue
                seen_chunks.add(digest)
                length = len(tokenizer.encode(chunk, add_special_tokens=False))
                stats[name].chunks += 1
                stats[name].tokens += length
                stats[name].chunk_lengths[length] += 1
                pieces.append(chunk)
        write_split(out_dir / f"{name}.jsonl", pieces)
        output[name] = stats[name].as_dict()
        print(f"[{name}] {len(texts)} texts -> {len(pieces)} chunks "
              f"({output[name]['chunk_tokens']} tokens)", flush=True)

    # Invariant: after the global chunk de-duplication no chunk repeats anywhere.
    chunk_owner: dict[bytes, str] = {}
    for name in splits:
        for chunk in (json.loads(line)["text"] for line in
                      (out_dir / f"{name}.jsonl").read_text(encoding="utf-8").splitlines()):
            digest = hashlib.blake2b(chunk.encode("utf-8"), digest_size=8).digest()
            if digest in chunk_owner:
                sys.exit(f"leakage: identical chunk in {chunk_owner[digest]} and {name} splits")
            chunk_owner[digest] = name
    leakage = {"overlapping_chunks": 0,
               "method": "splits chosen on raw texts before chunking, then identical chunks "
                         "de-duplicated with train priority"}

    report = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "elapsed_seconds": round(time.time() - started, 1),
        "seed": args.seed,
        "sample_requested": args.sample,
        "sample_kept": n,
        "tokenizer": {"path": args.tokenizer, "fingerprint": tokenizer.fingerprint,
                      "vocab_size": tokenizer.vocab_size},
        "context": {
            "max_seq_len": args.max_seq_len,
            "reserved_special_tokens": 2,
            "chunk_tokens": limit,
            "min_chunk_tokens": args.min_chunk_tokens,
        },
        "filters": {
            "min_chars": args.min_chars, "max_chars": args.max_chars,
            "min_printable_ratio": MIN_PRINTABLE_RATIO,
            "min_letter_ratio": MIN_LETTER_RATIO,
            "repeated_char_run": 20,
            "deep_checks": ["repeated_phrase (>=4 copies)", "repeated_line"],
        },
        "sources": sources,
        "scan": {"records": scanned, "rejected": dict(rejected.most_common()),
                 "duplicates": dict(duplicates), "candidates": len(reservoir.items)},
        "splits": output,
        "ratios": {"train": round(n_train / n, 4), "val": round(n_val / n, 4),
                   "test": round((n - n_train - n_val) / n, 4)},
        "leakage": leakage,
        "notes": [
            "Chunks are non-overlapping; each raw text belongs to exactly one split.",
            "Records use {\"text\": ...} and are at most max_seq_len tokens incl. BOS/EOS.",
        ],
    }
    (out_dir / "dataset.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    preview = {name: [json.loads(line)["text"]
                      for line in (out_dir / f"{name}.jsonl").read_text(encoding="utf-8").splitlines()[:3]]
               for name in splits}
    (out_dir / "preview.json").write_text(
        json.dumps(preview, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[done] {out_dir} in {report['elapsed_seconds']}s | totals: "
          + ", ".join(f"{name} {output[name]['chunks']} chunks" for name in splits))


if __name__ == "__main__":
    main()
