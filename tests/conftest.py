"""Shared fixtures: a small Chinese-first tokenizer and packed token files.

The tokenizer is trained once per session on synthetic Chinese text so the test
suite runs without the (gitignored) real corpus. The real-corpus efficiency
check lives in ``test_tokenizer.py`` and skips when the data is absent.
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pytest

from minillm.config import MINILLM_NANO, MiniLLMConfig
from minillm.data import TOKEN_DTYPE
from minillm.tokenizer import MiniLLMTokenizer

WORDS = [
    "人工智能", "机器学习", "模型", "训练", "数据", "语言", "中文", "英文", "代码",
    "数学", "推理", "逻辑", "问题", "答案", "学习", "优化", "计算", "网络", "信息",
    "知识", "技术", "系统", "方法", "结果", "实验", "工具", "任务", "目标", "能力",
]
PATTERNS = [
    "{a}是{b}的重要组成部分，{c}需要{d}的支持。",
    "在研究{a}时，我们经常讨论{b}和{c}，并关注{d}的效果。",
    "{a}可以帮助我们理解{b}，但它并不能代替{c}的思考。",
    "如果{a}和{b}同时出现，那么{c}通常会影响{d}。",
]


def synthetic_texts(count: int = 1200, seed: int = 0) -> list[str]:
    rng = random.Random(seed)
    texts = []
    for _ in range(count):
        pattern = rng.choice(PATTERNS)
        texts.append(pattern.format(
            a=rng.choice(WORDS), b=rng.choice(WORDS), c=rng.choice(WORDS), d=rng.choice(WORDS)
        ))
    return texts


@pytest.fixture(scope="session")
def tiny_tokenizer(tmp_path_factory) -> MiniLLMTokenizer:
    path = tmp_path_factory.mktemp("tok") / "tokenizer.json"
    tokenizer = MiniLLMTokenizer.train(synthetic_texts(), vocab_size=1024, save_path=path)
    return tokenizer


@pytest.fixture(scope="session")
def nano_cfg(tiny_tokenizer) -> MiniLLMConfig:
    cfg = MINILLM_NANO.copy()
    cfg.vocab_size = tiny_tokenizer.vocab_size
    return cfg


def pack_tokens(path: Path, tokenizer: MiniLLMTokenizer, texts: list[str]) -> int:
    tokens = []
    for text in texts:
        tokens.extend(tokenizer.encode_document(text))
    array = np.asarray(tokens, dtype=TOKEN_DTYPE)
    array.tofile(path)
    return int(array.size)


@pytest.fixture(scope="session")
def packed_data(tmp_path_factory, tiny_tokenizer):
    """(train.bin, val.bin) covering a few hundred tokens."""
    directory = tmp_path_factory.mktemp("packed")
    texts = synthetic_texts(120, seed=7)
    train = directory / "train.bin"
    val = directory / "val.bin"
    pack_tokens(train, tiny_tokenizer, texts[:80])
    pack_tokens(val, tiny_tokenizer, texts[80:])
    return train, val
