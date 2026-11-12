"""Data pipeline: packed pretraining tokens, SFT conversations, deterministic batching.

Pretraining data lives in flat ``uint16`` token files (one per split) produced by
``scripts/prepare_text.py``. Each file is a single continuous stream of tokens:
documents are tokenised separately, each terminated with EOS, then packed back to
back. A sample is a ``seq_len`` slice plus the next token as its final target, so
there is no padding inside an epoch and no document ever shares a window with a
different split.

    train.bin  ->  [doc1 <eos> doc2 <eos> doc3 <eos> ...]

Chat data lives in JSONL (``{"messages": [...]}``) and is tokenised with the same
chat template the web UI uses, so training and inference cannot drift apart.
"""

from __future__ import annotations

import json
import random
from array import array
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from .tokenizer import DEFAULT_SYSTEM, MiniLLMTokenizer

TOKEN_DTYPE = np.dtype("<u2")  # vocab_size must stay below 65536


def index_jsonl(path: str | Path) -> array:
    """Byte offsets of every non-blank line, in one sequential pass."""
    offsets = array("q")
    position = 0
    with Path(path).open("rb") as handle:
        for line in handle:
            if line.strip():
                offsets.append(position)
            position += len(line)
    return offsets


class JsonlRecords:
    """Random access to a JSONL file without loading it into memory."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.offsets = index_jsonl(self.path)
        self._handle = None

    def __len__(self) -> int:
        return len(self.offsets)

    def __getitem__(self, index: int) -> dict:
        if self._handle is None:
            self._handle = self.path.open("rb")
        self._handle.seek(self.offsets[index])
        line = self._handle.readline()
        try:
            raw = json.loads(line.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"{self.path}: offset {self.offsets[index]}: invalid JSON") from exc
        if not isinstance(raw, dict):
            raise ValueError(f"{self.path}: offset {self.offsets[index]}: expected an object")
        return raw

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["_handle"] = None  # workers reopen lazily
        return state

    def __del__(self):
        handle = getattr(self, "_handle", None)
        if handle is not None:
            try:
                handle.close()
            except OSError:
                pass


class PackedPretrainDataset(Dataset):
    """Fixed-length slices of a packed token stream (memmap, no padding)."""

    def __init__(self, path: str | Path, seq_len: int):
        if seq_len <= 0:
            raise ValueError("seq_len must be positive")
        self.path = Path(path)
        self.seq_len = seq_len
        self.tokens = np.memmap(self.path, dtype=TOKEN_DTYPE, mode="r")
        if self.tokens.size < seq_len + 1:
            raise ValueError(
                f"{self.path} holds {self.tokens.size} tokens, fewer than seq_len+1={seq_len + 1}"
            )
        # sample i = tokens[i*L : (i+1)*L + 1]; the target is the next token
        self.num_samples = (self.tokens.size - 1) // seq_len

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        if not 0 <= index < self.num_samples:
            raise IndexError(index)
        start = index * self.seq_len
        chunk = torch.from_numpy(np.asarray(self.tokens[start : start + self.seq_len + 1]).astype(np.int64))
        inputs = chunk[:-1]
        return {"input_ids": inputs, "labels": inputs.clone()}

    @property
    def num_tokens(self) -> int:
        return int(self.tokens.size)


class SFTDataset(Dataset):
    """Instruction data as chat transcripts with assistant-only supervision."""

    def __init__(
        self,
        path: str | Path,
        tokenizer: MiniLLMTokenizer,
        max_seq_len: int,
        *,
        supervise_all_assistant_turns: bool = True,
        system: str | None = DEFAULT_SYSTEM,
        records: Sequence[dict] | None = None,
    ):
        self.path = Path(path)
        self.tokenizer = tokenizer
        self.text_policy = tokenizer.text_policy
        self.max_seq_len = max_seq_len
        self.system = system
        self.supervise_all = supervise_all_assistant_turns
        self.raw = list(records) if records is not None else self._read(self.path)
        self.stats = {"records": len(self.raw), "dropped": 0, "truncated": 0, "supervised_tokens": 0}
        self.items: list[dict[str, torch.Tensor]] = []
        seen: set[str] = set()
        for record in self.raw:
            messages = record.get("messages")
            if not isinstance(messages, list) or not messages:
                self.stats["dropped"] += 1
                continue
            digest = json.dumps({"messages":messages,"category":record["category"]} if "category" in record else messages,
                                sort_keys=True, ensure_ascii=False)
            if digest in seen:  # exact duplicate conversation
                self.stats["dropped"] += 1
                continue
            seen.add(digest)
            if not any(m.get("role") == "assistant" and m.get("content") for m in messages):
                self.stats["dropped"] += 1
                continue
            encoding = tokenizer.chat_encoding(
                messages,
                system=self.system,
                supervise_all_assistant_turns=self.supervise_all,
                category=record.get("category", "auto"),
            )
            ids, labels = encoding.ids, encoding.labels
            if len(ids) > max_seq_len:
                # Keep the tail: the most recent assistant turn is the supervision
                # we care about, and dropping old context cannot corrupt labels.
                ids, labels = ids[-max_seq_len:], labels[-max_seq_len:]
                self.stats["truncated"] += 1
            if all(label == tokenizer.ignore_index for label in labels):
                self.stats["dropped"] += 1
                continue
            self.stats["supervised_tokens"] += sum(1 for label in labels if label != tokenizer.ignore_index)
            self.items.append(
                {
                    "input_ids": torch.tensor(ids, dtype=torch.long),
                    "labels": torch.tensor(labels, dtype=torch.long),
                }
            )
        if not self.items:
            raise ValueError(f"{self.path}: no usable SFT records")

    @staticmethod
    def _read(path: Path) -> list[dict]:
        records = []
        with path.open("r", encoding="utf-8") as handle:
            for number, line in enumerate(handle, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{number}: invalid JSON") from exc
                if not isinstance(record, dict):
                    raise ValueError(f"{path}:{number}: expected an object")
                records.append(record)
        return records

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return self.items[index]


def collate_batch(
    batch: Sequence[dict[str, torch.Tensor]],
    pad_token_id: int = 0,
    ignore_index: int = -100,
) -> dict[str, torch.Tensor]:
    """Right-pad token ids, labels and an attention mask."""
    max_len = max(item["input_ids"].shape[0] for item in batch)
    b = len(batch)
    ids = torch.full((b, max_len), pad_token_id, dtype=torch.long)
    labels = torch.full((b, max_len), ignore_index, dtype=torch.long)
    mask = torch.zeros((b, max_len), dtype=torch.long)
    for row, item in enumerate(batch):
        n = item["input_ids"].shape[0]
        ids[row, :n] = item["input_ids"]
        labels[row, :n] = item["labels"]
        mask[row, :n] = 1
    return {"input_ids": ids, "labels": labels, "attention_mask": mask}


class EpochBatchSampler:
    """Deterministic shuffling per epoch, resumable at a (epoch, batch) position."""

    def __init__(self, num_samples: int, batch_size: int, seed: int = 42, drop_last: bool = False):
        if num_samples <= 0 or batch_size <= 0:
            raise ValueError("num_samples and batch_size must be positive")
        self.num_samples = num_samples
        self.batch_size = batch_size
        self.seed = seed
        self.drop_last = drop_last
        self.num_batches = num_samples // batch_size if drop_last else -(-num_samples // batch_size)

    def indices(self, epoch: int) -> list[int]:
        order = list(range(self.num_samples))
        random.Random(self.seed + epoch).shuffle(order)
        return order

    def batches(self, epoch: int, start_batch: int = 0) -> Iterable[list[int]]:
        if not 0 <= start_batch <= self.num_batches:
            raise ValueError("micro-batch cursor outside dataset range")
        order = self.indices(epoch)
        for start in range(start_batch * self.batch_size, len(order), self.batch_size):
            batch = order[start : start + self.batch_size]
            if len(batch) < self.batch_size and self.drop_last:
                break
            yield batch

    def __len__(self) -> int:
        return self.num_batches


def stream_corpus_records(path: str | Path, *, max_record_bytes: int = 1048576):
    """Bounded JSONL records or blank-line-delimited TXT paragraphs.

    Oversize records are drained, never split into accidentally valid tail records.
    TXT paragraphs carry a common file parent; JSONL can supply parent_id explicitly.
    Rejection events let the pipeline account for malformed input without guessing.
    """
    path = Path(path)
    if max_record_bytes <= 0:
        raise ValueError("max_record_bytes must be positive")
    with path.open("rb") as handle:
        paragraph = bytearray()
        discarded = False
        line_number = 0
        while True:
            line = handle.readline(max_record_bytes + 1)
            if not line:
                break
            line_number += 1
            oversize = len(line) > max_record_bytes
            if oversize and not line.endswith(b"\n"):
                while tail := handle.readline(max_record_bytes + 1):
                    if tail.endswith(b"\n"):
                        break
            if path.suffix.lower() == ".jsonl":
                if not line.strip():
                    continue
                if oversize:
                    yield {"_issue": "record_too_large", "_line": line_number}
                    continue
                try:
                    record = json.loads(line.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError, RecursionError):
                    yield {"_issue": "invalid_json_or_utf8", "_line": line_number}
                    continue
                if not isinstance(record, dict) or not isinstance(record.get("text"), str):
                    yield {"_issue": "missing_text", "_line": line_number}
                    continue
                fields = ("text", "source", "category", "language", "license", "parent_id")
                yield {**{k: record[k] for k in fields if k in record}, "_line": line_number}
            else:
                if not line.strip():
                    if discarded:
                        yield {"_issue": "record_too_large", "_line": line_number}
                    elif paragraph:
                        try:
                            yield {"text": paragraph.decode("utf-8"), "_line": line_number}
                        except UnicodeDecodeError:
                            yield {"_issue": "invalid_utf8", "_line": line_number}
                    paragraph.clear()
                    discarded = False
                elif oversize or len(paragraph) + len(line) > max_record_bytes:
                    discarded = True
                    paragraph.clear()
                elif not discarded:
                    paragraph.extend(line)
        if path.suffix.lower() != ".jsonl":
            if discarded:
                yield {"_issue": "record_too_large", "_line": line_number}
            elif paragraph:
                try:
                    yield {"text": paragraph.decode("utf-8"), "_line": line_number}
                except UnicodeDecodeError:
                    yield {"_issue": "invalid_utf8", "_line": line_number}


def stream_documents(paths: Iterable[str | Path]) -> Iterable[str]:
    """Bounded text-only adapter shared by tokenizer training and evaluation."""
    for path in paths:
        records = stream_corpus_records(path)
        try:
            for record in records:
                if "_issue" not in record:
                    yield record["text"]
        finally:
            records.close()


def encode_document_tokens(texts: Iterable[str], tokenizer: MiniLLMTokenizer, *,
                           drop_empty: bool = True, log_fn=None, category: str = "auto") -> Iterable[np.ndarray]:
    """Yield one uint16 array at a time; never aggregate the corpus token stream."""
    total = 0
    for index, text in enumerate(texts):
        if drop_empty and not text.strip():
            continue
        ids = tokenizer.encode_document(text, category=category)
        if any(type(token) is not int or not 0 <= token < min(tokenizer.vocab_size, 65536) for token in ids):
            raise ValueError("token id outside vocabulary/uint16 range")
        total += len(ids)
        yield np.asarray(ids, dtype="<u2")
        if log_fn is not None and index and index % 20000 == 0:
            log_fn(f"encoded {index} documents, {total:,} tokens")
