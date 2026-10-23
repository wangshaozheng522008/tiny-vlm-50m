"""Datasets and preprocessing for image-caption training.

The expected on-disk format is JSONL, one object per line::

    {"image": "images/cat.jpg", "caption": "a cat sleeping on a sofa"}

Relative image paths are resolved against the JSONL file's directory.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Sequence

import torch
from torch.utils.data import Dataset

from .tokenizer import BOS_ID, EOS_ID, TinyTokenizer

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

DEFAULT_PROMPT = "Describe this image:"


def build_transform(img_size: int):
    from torchvision import transforms

    return transforms.Compose(
        [
            transforms.Resize(img_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def load_image(path: str | Path) -> "Any":
    from PIL import Image

    return Image.open(path).convert("RGB")


def build_example(
    tokenizer: TinyTokenizer,
    prompt: str,
    answer: str,
    max_seq_len: int,
    ignore_index: int = -100,
) -> tuple[list[int], list[int]]:
    """Tokenise `prompt + answer`, masking the prompt so loss only covers the answer."""
    prompt_ids = tokenizer.encode(prompt + " ", add_special_tokens=False)
    answer_ids = tokenizer.encode(answer, add_special_tokens=False)
    input_ids = [BOS_ID, *prompt_ids, *answer_ids, EOS_ID]
    labels = [ignore_index] * (1 + len(prompt_ids)) + list(answer_ids) + [EOS_ID]

    input_ids = input_ids[:max_seq_len]
    labels = labels[:max_seq_len]
    return input_ids, labels


class CaptionDataset(Dataset):
    def __init__(
        self,
        jsonl_path: str | Path,
        tokenizer: TinyTokenizer,
        img_size: int = 224,
        max_seq_len: int = 256,
        prompt: str = DEFAULT_PROMPT,
        ignore_index: int = -100,
    ):
        self.path = Path(jsonl_path)
        self.root = self.path.parent
        self.records = [
            json.loads(line)
            for line in self.path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        self.tokenizer = tokenizer
        self.transform = build_transform(img_size)
        self.max_seq_len = max_seq_len
        self.prompt = prompt
        self.ignore_index = ignore_index

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        rec = self.records[idx]
        image = load_image(self.root / rec["image"])
        pixels = self.transform(image)
        input_ids, labels = build_example(
            self.tokenizer, self.prompt, rec["caption"], self.max_seq_len, self.ignore_index
        )
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "pixel_values": pixels,
        }


class SyntheticCaptionDataset(Dataset):
    """Random-noise images with templated captions, for smoke-testing the pipeline."""

    SHAPES = ["circle", "square", "triangle", "star"]
    COLORS = ["red", "green", "blue", "yellow", "purple"]

    def __init__(
        self,
        size: int,
        tokenizer: TinyTokenizer,
        img_size: int = 224,
        max_seq_len: int = 256,
        prompt: str = DEFAULT_PROMPT,
        ignore_index: int = -100,
    ):
        self.size = size
        self.tokenizer = tokenizer
        self.img_size = img_size
        self.max_seq_len = max_seq_len
        self.prompt = prompt
        self.ignore_index = ignore_index

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        g = torch.Generator().manual_seed(idx)
        pixels = torch.randn(3, self.img_size, self.img_size, generator=g)
        caption = f"a {random.Random(idx).choice(self.COLORS)} {random.Random(idx + 1).choice(self.SHAPES)}"
        input_ids, labels = build_example(
            self.tokenizer, self.prompt, caption, self.max_seq_len, self.ignore_index
        )
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "pixel_values": pixels,
        }


def collate_fn(batch: Sequence[dict[str, torch.Tensor]], pad_token_id: int = 0) -> dict[str, torch.Tensor]:
    """Right-pad token sequences to the longest item in the batch."""
    max_len = max(item["input_ids"].shape[0] for item in batch)
    input_ids, labels, masks = [], [], []
    for item in batch:
        ids, lab = item["input_ids"], item["labels"]
        pad = max_len - ids.shape[0]
        input_ids.append(torch.cat([ids, torch.full((pad,), pad_token_id, dtype=torch.long)]))
        labels.append(torch.cat([lab, torch.full((pad,), -100, dtype=torch.long)]))
        masks.append(torch.cat([torch.ones(ids.shape[0], dtype=torch.long), torch.zeros(pad, dtype=torch.long)]))
    return {
        "input_ids": torch.stack(input_ids),
        "labels": torch.stack(labels),
        "attention_mask": torch.stack(masks),
        "pixel_values": torch.stack([item["pixel_values"] for item in batch]),
    }
