"""Text, image-only and image+text supervision, with legacy caption support."""

from __future__ import annotations

import json
import random
from array import array
from bisect import bisect_right
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch
from torch.utils.data import Dataset

from .tokenizer import BOS_ID, EOS_ID, TinyTokenizer

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
DEFAULT_PROMPT = "Describe this image:"
MODES = ("text", "image", "mixed")


def build_transform(img_size: int):
    from torchvision import transforms

    return transforms.Compose([
        transforms.Resize(img_size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def load_image(path: str | Path) -> Any:
    from PIL import Image

    with Image.open(path) as image:
        return image.convert("RGB")


def build_prompt(tokenizer: TinyTokenizer, prompt: str) -> list[int]:
    """The same prefix at training and inference; EOS belongs only after the answer."""
    return [BOS_ID, *tokenizer.encode(prompt + " " if prompt else "", add_special_tokens=False)]


def build_example(
    tokenizer: TinyTokenizer,
    prompt: str,
    answer: str,
    max_seq_len: int,
    ignore_index: int = -100,
) -> tuple[list[int], list[int]]:
    if max_seq_len < 3:
        raise ValueError("max_seq_len must be at least 3 (BOS, answer, EOS)")
    if not answer.strip():
        raise ValueError("answer must contain non-whitespace text")
    prefix = build_prompt(tokenizer, prompt)
    # Keep answer supervision even when a prompt is too long. Reserve EOS as well.
    answer_ids = tokenizer.encode(answer, add_special_tokens=False)[: max_seq_len - 2]
    prefix = prefix[: max_seq_len - len(answer_ids) - 1]
    input_ids = [*prefix, *answer_ids, EOS_ID]
    labels = [ignore_index] * len(prefix) + [*answer_ids, EOS_ID]
    return input_ids, labels


@dataclass(frozen=True)
class TrainingRecord:
    image: Path | None
    prompt: str
    answer: str

    @property
    def mode(self) -> str:
        return "text" if self.image is None else ("mixed" if self.prompt else "image")


def read_records(paths: str | Path | Sequence[str | Path]) -> list[TrainingRecord]:
    """Resolve each relative image path against its own JSONL source."""
    if isinstance(paths, (str, Path)):
        paths = [paths]
    records = []
    for source in paths:
        path = Path(source)
        for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
                if not isinstance(raw, dict):
                    raise ValueError("record must be an object")
                image = raw.get("image")
                if image is not None and (not isinstance(image, str) or not image.strip()):
                    raise ValueError("image must be a nonempty path or null")
                if "answer" in raw:
                    answer = raw["answer"]
                    prompt = raw.get("prompt", "")
                elif "caption" in raw and image is not None:
                    answer = raw["caption"]
                    prompt = raw.get("prompt", DEFAULT_PROMPT)
                elif "text" in raw and image is None:
                    answer, prompt = raw["text"], ""
                else:
                    raise ValueError("expected {prompt, answer}, {image, answer}, {image, caption}, or {text}")
                if not isinstance(prompt, str) or not isinstance(answer, str) or not answer.strip():
                    raise ValueError("prompt must be text and answer must contain non-whitespace text")
                if image is None and not prompt.strip() and "text" not in raw:
                    raise ValueError("text-only instruction samples need a prompt; use {text} for language modelling")
                records.append(TrainingRecord(
                    image=path.parent / image if image is not None else None,
                    prompt=prompt if prompt.strip() else "",
                    answer=answer,
                ))
            except (ValueError, TypeError) as exc:
                raise ValueError(f"{path}:{line_no}: {exc}") from exc
    if not records:
        raise ValueError("dataset contains no records")
    return records


def record_texts(records: Sequence[TrainingRecord]) -> Iterable[str]:
    for record in records:
        if record.prompt:
            yield record.prompt + " "
        yield record.answer


class MultimodalDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, img_size=224, max_seq_len=256,
                 ignore_index=-100, records=None):
        self.records = list(records) if records is not None else read_records(jsonl_path)
        self.modes = [record.mode for record in self.records]
        self.tokenizer = tokenizer
        self.transform = build_transform(img_size) if any(r.image is not None for r in self.records) else None
        self.max_seq_len = max_seq_len
        self.ignore_index = ignore_index

    @property
    def mode_counts(self) -> dict[str, int]:
        return dict(Counter(self.modes))

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        record = self.records[idx]
        input_ids, labels = build_example(
            self.tokenizer, record.prompt, record.answer, self.max_seq_len, self.ignore_index
        )
        item = {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "mode_id": torch.tensor(MODES.index(record.mode), dtype=torch.long),
        }
        if record.image is not None:
            item["pixel_values"] = self.transform(load_image(record.image))
        return item


class CaptionDataset(MultimodalDataset):
    """Backwards-compatible image-caption dataset constructor."""

    def __init__(self, jsonl_path, tokenizer, img_size=224, max_seq_len=256,
                 prompt=DEFAULT_PROMPT, ignore_index=-100):
        records = [TrainingRecord(r.image, prompt, r.answer) for r in read_records(jsonl_path)]
        super().__init__(jsonl_path, tokenizer, img_size, max_seq_len, ignore_index, records)


class IndexedTextDataset(Dataset):
    """Random access to pre-chunked ``{"text": ...}`` JSONL without keeping it in RAM.

    Only byte offsets are indexed (8 bytes per line); each sample is read and
    tokenised on demand. This is the reader to use for a full pretraining run,
    where the chunked corpus is far larger than the model.
    """

    def __init__(self, paths, tokenizer, max_seq_len: int = 256,
                 ignore_index: int = -100):
        if isinstance(paths, (str, Path)):
            paths = [paths]
        self.paths = [Path(p) for p in paths]
        if not self.paths:
            raise ValueError("indexed dataset needs at least one JSONL file")
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.ignore_index = ignore_index
        self._handles: dict[Path, Any] = {}
        self._files: list[tuple[Path, array]] = []
        self._starts = [0]
        total = 0
        for path in self.paths:
            offsets = index_jsonl(path)
            if not offsets:
                continue
            self._files.append((path, offsets))
            total += len(offsets)
            self._starts.append(total)
        if not total:
            raise ValueError("indexed dataset contains no text records")
        self._length = total
        self.modes = ["text"] * total

    @property
    def mode_counts(self) -> dict[str, int]:
        return {"text": self._length}

    def __len__(self) -> int:
        return self._length

    def _locate(self, index: int) -> tuple[Path, int]:
        file_index = bisect_right(self._starts, index) - 1
        path, offsets = self._files[file_index]
        return path, offsets[index - self._starts[file_index]]

    def _read(self, path: Path, offset: int) -> dict:
        handle = self._handles.get(path)
        if handle is None:
            handle = path.open("rb")
            self._handles[path] = handle
        handle.seek(offset)
        line = handle.readline()
        try:
            raw = json.loads(line.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"{path}: offset {offset}: invalid JSON record") from exc
        if not isinstance(raw, dict) or not isinstance(raw.get("text"), str):
            raise ValueError(f"{path}: offset {offset}: expected {{\"text\": ...}}")
        return raw

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        path, offset = self._locate(index)
        record = self._read(path, offset)
        input_ids, labels = build_example(
            self.tokenizer, "", record["text"], self.max_seq_len, self.ignore_index
        )
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "mode_id": torch.tensor(MODES.index("text"), dtype=torch.long),
        }

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["_handles"] = {}  # worker processes reopen lazily
        return state

    def __del__(self):
        for handle in getattr(self, "_handles", {}).values():
            try:
                handle.close()
            except OSError:
                pass


def index_jsonl(path: str | Path) -> array:
    """Byte offsets of every nonblank line, with one sequential pass over the file."""
    offsets = array("q")
    position = 0
    with Path(path).open("rb") as handle:
        for line in handle:
            if line.strip():
                offsets.append(position)
            position += len(line)
    return offsets


SYNTHETIC_COLORS = {"red": (220, 30, 30), "green": (30, 170, 30), "blue": (30, 30, 220)}
SYNTHETIC_SHAPES = ("circle", "square", "triangle")


def synthetic_record(index: int, modes: Sequence[str] = MODES, seed: int = 0):
    rng = random.Random(seed + index // len(modes))
    color = rng.choice(list(SYNTHETIC_COLORS))
    shape = rng.choice(SYNTHETIC_SHAPES)
    mode = modes[index % len(modes)]
    if mode == "text":
        prompt, answer = f"What color is a {color} {shape}?", color
    elif mode == "image":
        prompt, answer = "", f"a {color} {shape}"
    elif rng.randrange(2):
        prompt, answer = "What color is the shape?", color
    else:
        prompt, answer = "What shape is shown?", shape
    return mode, color, shape, prompt, answer


def synthetic_texts() -> Iterable[str]:
    for color in SYNTHETIC_COLORS:
        for shape in SYNTHETIC_SHAPES:
            yield f"What color is a {color} {shape}? {color}"
            yield f"a {color} {shape}"
            yield f"What color is the shape? {color}"
            yield f"What shape is shown? {shape}"
    yield DEFAULT_PROMPT


class SyntheticMultimodalDataset(Dataset):
    """Grounded toy examples for exercising all input modes; not general training data."""

    def __init__(self, size, tokenizer, img_size=224, max_seq_len=256,
                 ignore_index=-100, modes=MODES, seed=0):
        if size <= 0 or not modes or any(mode not in MODES for mode in modes):
            raise ValueError(f"size must be positive and modes must come from {MODES}")
        self.records = [synthetic_record(i, modes, seed) for i in range(size)]
        self.modes = [r[0] for r in self.records]
        self.tokenizer = tokenizer
        self.img_size = img_size
        self.max_seq_len = max_seq_len
        self.ignore_index = ignore_index
        self.transform = build_transform(img_size) if any(m != "text" for m in self.modes) else None

    @property
    def mode_counts(self):
        return dict(Counter(self.modes))

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        from PIL import Image, ImageDraw

        mode, color, shape, prompt, answer = self.records[idx]
        ids, labels = build_example(self.tokenizer, prompt, answer, self.max_seq_len, self.ignore_index)
        item = {"input_ids": torch.tensor(ids), "labels": torch.tensor(labels),
                "mode_id": torch.tensor(MODES.index(mode))}
        if mode != "text":
            size = self.img_size
            image = Image.new("RGB", (size, size), "white")
            draw = ImageDraw.Draw(image)
            bounds = (size // 4, size // 4, 3 * size // 4, 3 * size // 4)
            if shape == "circle":
                draw.ellipse(bounds, fill=SYNTHETIC_COLORS[color])
            elif shape == "square":
                draw.rectangle(bounds, fill=SYNTHETIC_COLORS[color])
            else:
                draw.polygon([(size // 2, size // 4), (size // 4, 3 * size // 4),
                              (3 * size // 4, 3 * size // 4)], fill=SYNTHETIC_COLORS[color])
            item["pixel_values"] = self.transform(image)
        return item


class SyntheticCaptionDataset(SyntheticMultimodalDataset):
    """Compatibility alias for the original synthetic caption API."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, modes=("image",), **kwargs)


def collate_fn(batch: Sequence[dict[str, torch.Tensor]], pad_token_id: int = 0,
               ignore_index: int = -100) -> dict[str, torch.Tensor]:
    max_len = max(item["input_ids"].shape[0] for item in batch)
    b = len(batch)
    ids = torch.full((b, max_len), pad_token_id, dtype=torch.long)
    labels = torch.full((b, max_len), ignore_index, dtype=torch.long)
    mask = torch.zeros((b, max_len), dtype=torch.long)
    for row, item in enumerate(batch):
        n = item["input_ids"].shape[0]
        ids[row, :n], labels[row, :n], mask[row, :n] = item["input_ids"], item["labels"], 1
    result = {"input_ids": ids, "labels": labels, "attention_mask": mask}
    if all("mode_id" in item for item in batch):
        result["mode_id"] = torch.stack([item["mode_id"] for item in batch])
    images = [item.get("pixel_values") for item in batch]
    template = next((image for image in images if image is not None), None)
    if template is not None:
        result["pixel_values"] = torch.stack([
            image if image is not None else torch.zeros_like(template) for image in images
        ])
        result["image_mask"] = torch.tensor([image is not None for image in images])
    return result
