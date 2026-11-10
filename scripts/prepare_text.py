"""Turn raw ``{"text": ...}`` corpora into packed uint16 token streams.

Two stages, deliberately separate so the tokenizer can never see validation data:

1. ``--split`` (default): stream every source, normalise and quality-filter it,
   drop exact and near duplicates, reservoir-sample N documents, split the
   *raw texts* 96/2/2, and write ``train.text.jsonl`` / ``val.text.jsonl`` /
   ``test.text.jsonl`` plus ``dataset.json``.
2. ``--tokenize-only --tokenizer PATH``: read those split files, tokenise each
   document (terminating it with EOS), pack the documents back to back, and
   write ``train.bin`` / ``val.bin`` / ``test.bin`` (uint16).

Splitting before tokenising is what guarantees that no document — and therefore
no packed window — crosses train/validation/test boundaries.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from minillm.data import TOKEN_DTYPE, stream_documents
from minillm.tokenizer import MiniLLMTokenizer

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
        text.replace("\r\n", "\n").replace("\r", "\n")
        .replace("\u00a0", " ").replace("\u3000", " ")
    )
    text = NEWLINE_RUN_RE.sub("\n\n", text)
    return SPACE_RUN_RE.sub(" ", text).strip()


def quality_issue(text: str, min_chars: int, max_chars: int, deep: bool) -> str | None:
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
        if REPEATED_PHRASE_RE.search(text):
            return "repeated_phrase"
        lines = [line.strip() for line in text.split("\n") if len(line.strip()) > 12]
        if len(lines) >= 6:
            top = Counter(lines).most_common(1)[0][1]
            if top / len(lines) > 0.5:
                return "repeated_line"
    return None


def file_sha256(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk):
            digest.update(block)
    return digest.hexdigest()


@dataclass
class SplitStats:
    texts: int = 0
    chars: int = 0
    tokens: int = 0
    documents: int = 0

    def as_dict(self) -> dict:
        return {
            "texts": self.texts,
            "docs": self.documents,
            "text_chars": self.chars,
            "tokens": self.tokens,
            "chars_per_token": round(self.chars / self.tokens, 4) if self.tokens else None,
        }


@dataclass
class ScanStats:
    seen: int = 0
    kept: int = 0
    rejected: Counter = field(default_factory=Counter)
    duplicates_exact: int = 0
    duplicates_near: int = 0
    empty: int = 0

    def as_dict(self) -> dict:
        return {
            "records_seen": self.seen,
            "kept": self.kept,
            "empty": self.empty,
            "rejected": dict(self.rejected),
            "duplicates_exact": self.duplicates_exact,
            "duplicates_near": self.duplicates_near,
        }


def scan_corpus(paths: list[Path], args, log=print) -> tuple[list[str], ScanStats]:
    """Filter, dedupe and reservoir-sample raw documents."""
    stats = ScanStats()
    seen_exact: set[str] = set()
    seen_near: set[str] = set()
    reservoir: list[str] = []
    rng = random.Random(args.seed)
    started = time.time()
    for path in paths:
        for raw in stream_documents([path]):
            if args.max_records and stats.seen >= args.max_records:
                break
            stats.seen += 1
            text = normalise(raw)
            if not text:
                stats.empty += 1
                continue
            issue = quality_issue(text, args.min_chars, args.max_chars, deep=False)
            if issue:
                stats.rejected[issue] += 1
                continue
            digest = hashlib.blake2b(text.encode("utf-8"), digest_size=16).hexdigest()
            if digest in seen_exact:
                stats.duplicates_exact += 1
                continue
            seen_exact.add(digest)
            near = hashlib.blake2b(text[:256].encode("utf-8"), digest_size=16).hexdigest()
            if near in seen_near:
                stats.duplicates_near += 1
                continue
            seen_near.add(near)
            issue = quality_issue(text, args.min_chars, args.max_chars, deep=True)
            if issue:
                stats.rejected[issue] += 1
                continue
            stats.kept += 1
            if args.sample and len(reservoir) < args.sample:
                reservoir.append(text)
            elif args.sample:
                index = rng.randrange(stats.kept)
                if index < args.sample:
                    reservoir[index] = text
            else:
                reservoir.append(text)
            if stats.seen % 200_000 == 0:
                log(f"  scanned {stats.seen:,} records, kept {stats.kept:,}, "
                    f"{time.time() - started:.0f}s")
    if not reservoir:
        raise SystemExit("no documents survived filtering; check --min-chars/--sample")
    rng.shuffle(reservoir)
    log(f"  scanned {stats.seen:,} records, kept {stats.kept:,}, sampled {len(reservoir):,}")
    return reservoir, stats


def split_documents(texts: list[str], fractions: tuple[float, float, float]) -> dict[str, list[str]]:
    if abs(sum(fractions) - 1.0) > 1e-6:
        raise SystemExit("split fractions must sum to 1.0")
    n = len(texts)
    n_val = max(1, int(n * fractions[1])) if n >= 3 else 0
    n_test = max(1, int(n * fractions[2])) if n >= 3 else 0
    n_train = n - n_val - n_test
    if n_train <= 0:
        raise SystemExit("not enough documents for a train split; raise --sample")
    return {
        "train": texts[:n_train],
        "val": texts[n_train : n_train + n_val],
        "test": texts[n_train + n_val :],
    }


def write_split(path: Path, texts: list[str]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for text in texts:
            handle.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")


def read_split(path: Path) -> list[str]:
    texts = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                texts.append(json.loads(line)["text"])
    return texts


def tokenize_split(texts: list[str], tokenizer: MiniLLMTokenizer, out_path: Path,
                   shuffle_seed: int) -> SplitStats:
    order = list(range(len(texts)))
    random.Random(shuffle_seed).shuffle(order)
    chunks: list[np.ndarray] = []
    stats = SplitStats()
    for index in order:
        text = texts[index]
        ids = tokenizer.encode_document(text)
        if len(ids) <= 1:
            continue
        chunks.append(np.asarray(ids, dtype=TOKEN_DTYPE))
        stats.documents += 1
        stats.chars += len(text)
        stats.tokens += len(ids)
    if not chunks:
        raise SystemExit(f"{out_path.name}: no tokens produced")
    packed = np.concatenate(chunks)
    if int(packed.max()) >= 65536:
        raise SystemExit("token id exceeds uint16; lower the vocabulary size")
    packed.tofile(out_path)
    stats.texts = len(texts)
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Build packed pretraining token streams")
    parser.add_argument("--input", nargs="+", required=True,
                        help='JSONL {"text": ...} or plain-text files')
    parser.add_argument("--out", required=True, help="output directory")
    parser.add_argument("--sample", type=int, default=0, help="documents to keep (0 = all)")
    parser.add_argument("--max-records", type=int, default=0,
                        help="stop scanning a source after N records (0 = all)")
    parser.add_argument("--min-chars", type=int, default=80)
    parser.add_argument("--max-chars", type=int, default=20000)
    parser.add_argument("--split", nargs=3, type=float, default=[0.96, 0.02, 0.02],
                        metavar=("TRAIN", "VAL", "TEST"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shuffle-seed", type=int, default=13)
    parser.add_argument("--tokenizer", help="tokenizer.json for --tokenize-only")
    parser.add_argument("--tokenize-only", action="store_true",
                        help="tokenise the existing *.text.jsonl splits into *.bin")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    sources = [{"path": str(Path(p)), "bytes": Path(p).stat().st_size,
                "sha256": file_sha256(Path(p))} for p in args.input]
    meta_path = out / "dataset.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}

    if not args.tokenize_only:
        reservoir, scan = scan_corpus([Path(p) for p in args.input], args)
        parts = split_documents(reservoir, tuple(args.split))
        for name, texts in parts.items():
            write_split(out / f"{name}.text.jsonl", texts)
        meta = {
            "stage": "split",
            "sources": sources,
            "seed": args.seed,
            "sample": args.sample,
            "split_fractions": args.split,
            "scan": scan.as_dict(),
            "texts": {name: {"documents": len(texts), "chars": sum(map(len, texts))}
                      for name, texts in parts.items()},
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }
        meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False))
        print(f"split -> {out} ({', '.join(f'{k}={len(v)}' for k, v in parts.items())})")
        return

    if not args.tokenizer:
        raise SystemExit("--tokenize-only requires --tokenizer")
    tokenizer = MiniLLMTokenizer.load(args.tokenizer)
    meta.setdefault("sources", sources)
    meta["stage"] = "tokenized"
    meta["tokenizer"] = {
        "path": str(Path(args.tokenizer)),
        "fingerprint": tokenizer.fingerprint,
        "vocab_size": tokenizer.vocab_size,
    }
    if tokenizer.vocab_size >= 65536:
        raise SystemExit("vocabulary must be smaller than 65536 for uint16 storage")
    meta["splits"] = {}
    for name in ("train", "val", "test"):
        split_path = out / f"{name}.text.jsonl"
        if not split_path.exists():
            raise SystemExit(f"missing {split_path}; run the split stage first")
        texts = read_split(split_path)
        stats = tokenize_split(texts, tokenizer, out / f"{name}.bin", args.shuffle_seed)
        meta["splits"][name] = stats.as_dict()
        print(f"  {name}: {stats.documents:,} docs, {stats.tokens:,} tokens -> {name}.bin")
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    print(f"tokenized -> {out}")


if __name__ == "__main__":
    main()
