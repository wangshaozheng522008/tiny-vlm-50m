"""External, versioned text policy. Never changes tokenizer JSON or vocabulary.

Legacy experiments use identity-v1. Width folding is opt-in for a NEW model
experiment, and deliberately avoids NFKC's changes to math and source code.
"""
from __future__ import annotations

import re

IDENTITY = "identity-v1"
WIDTH_FOLD = "width-fold-prose-v1"
POLICIES = (IDENTITY, WIDTH_FOLD)
TEXT_CATEGORIES = ("auto", "zh", "en", "code", "mixed", "unknown")
_CODE = re.compile(r"(?m)^\s*(?:def |class |import |from \w+ import |return |print\(|[A-Za-z_]\w*\s*=)|```|~~~|#include")
_LITERAL = re.compile(r"(`+).*?\1|\$[^$\n]+\$", re.S)
_WIDTH = str.maketrans({**{chr(i): chr(i - 0xFEE0) for i in range(0xFF01, 0xFF5F)}, "\u3000": " "})


def validate_policy(policy: str) -> str:
    if policy not in POLICIES:
        raise ValueError(f"unknown text policy: {policy!r}; supported: {POLICIES}")
    return policy


def validate_text_category(category: str) -> str:
    if category not in TEXT_CATEGORIES:
        raise ValueError(f"unknown text category: {category!r}; supported: {TEXT_CATEGORIES}")
    return category


def apply_text_policy(text: str, policy: str = IDENTITY, *, category: str = "auto") -> str:
    validate_policy(policy)
    validate_text_category(category)
    if policy == IDENTITY or category == "code" or _CODE.search(text):
        return text
    # Keep explicitly delimited literals. No strip, whitespace collapse, NFC or
    # NFKC here: callers must see exactly the same code points at fit/encode time.
    pieces, offset = [], 0
    for match in _LITERAL.finditer(text):
        pieces.extend((text[offset:match.start()].translate(_WIDTH), match.group()))
        offset = match.end()
    pieces.append(text[offset:].translate(_WIDTH))
    return "".join(pieces)


def checkpoint_text_policy(payload: dict) -> str:
    train = payload.get("manifest", {}).get("dataset", {}).get("train", {})
    inherited = train.get("text_policy", IDENTITY)
    policy = validate_policy(payload.get("text_policy", inherited))
    if "text_policy" in train and policy != inherited:
        raise ValueError("checkpoint text policy disagrees with training manifest")
    return policy
