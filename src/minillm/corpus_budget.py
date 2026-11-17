"""Effective-token corpus budget in the same unit the trainer supervises.

``PackedPretrainDataset`` slices a token stream into ``seq_len`` windows and the
model shifts labels internally.  With ``N`` tokens, ``W = (N - 1) // seq_len``
windows are used and exactly ``W * (seq_len - 1)`` target tokens receive loss:
position 0 is input-only and every window's first token is not a target.  The
final ``N - W * seq_len - 1`` tokens are never supervised.  This module uses
that exact definition; it never equates "tokens in a file" with "training
tokens".

Per-category supervision is computed exactly from per-document token lengths and
their order in the packed stream: a token at absolute position ``p`` is
supervised iff ``1 <= p <= W * seq_len`` and ``p % seq_len != 0``.  No category
ratio is estimated from characters.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

from .corpus_quality import V4_CATEGORIES
from .corpus_utils import identity

STATUSES = ("READY_FOR_TRAINING", "INSUFFICIENT_CORPUS", "NOT_ASSESSED")


def packed_effective_tokens(raw_tokens: int, seq_len: int) -> tuple[int, int]:
    """(supervised target tokens, packed windows) for a flat token stream."""
    if type(raw_tokens) is not int or raw_tokens < 0:
        raise ValueError("raw_tokens must be a nonnegative integer")
    if type(seq_len) is not int or seq_len < 2:
        raise ValueError("seq_len must be an integer >= 2")
    windows = max(0, (raw_tokens - 1) // seq_len)
    return windows * (seq_len - 1), windows


def _multiples_in_range(start: int, end: int, seq_len: int) -> int:
    """Count multiples of seq_len in the half-open range [start, end)."""
    if end <= start:
        return 0
    return (end - 1) // seq_len - (start - 1) // seq_len


def supervision_by_category(documents, seq_len: int) -> dict:
    """Exact per-category supervised tokens from ordered (category, tokens) pairs.

    ``documents`` must be in packed-stream order.  It is consumed lazily so a
    one-million-document index does not need to be materialised.
    """
    if type(seq_len) is not int or seq_len < 2:
        raise ValueError("seq_len must be an integer >= 2")
    lengths, categories, total = [], [], 0
    for category, tokens in documents:
        if category not in V4_CATEGORIES:
            raise ValueError(f"unknown v4 category in token index: {category}")
        if type(tokens) is not int or tokens <= 0:
            raise ValueError("document token counts must be positive integers")
        lengths.append(tokens)
        categories.append(category)
        total += tokens
    windows = (total - 1) // seq_len
    limit = windows * seq_len
    counts = {category: 0 for category in V4_CATEGORIES}
    documents_by_category = {category: 0 for category in V4_CATEGORIES}
    cursor = 0
    for length, category in zip(lengths, categories):
        start, end = cursor, cursor + length
        a, b = max(start, 1), min(end, limit + 1)
        if b > a:
            counts[category] += (b - a) - _multiples_in_range(a, b, seq_len)
        documents_by_category[category] += 1
        cursor = end
    return {"raw_tokens": total, "windows": windows,
            "effective_tokens": windows * (seq_len - 1),
            "unsupervised_tail_tokens": total - limit,
            "by_category": counts, "documents_by_category": documents_by_category}


def read_token_index(paths: list[str | Path]) -> dict:
    """Read one or more ``*.token-index.jsonl`` files in file order."""
    records = []
    for path in paths:
        path = Path(path)
        with path.open(encoding="utf-8") as handle:
            for number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if not isinstance(row, dict) or not isinstance(row.get("tokens"), int) or row["tokens"] <= 0:
                    raise ValueError(f"{path}:{number}: invalid token index row")
                if row.get("category") not in V4_CATEGORIES:
                    raise ValueError(f"{path}:{number}: unknown category")
                records.append(row)
    return {"records": records}


def plan(available: dict[str, dict], target_effective_tokens: int,
         category_token_ratios: dict[str, float], *, seq_len: int,
         tokenizer_fingerprint: str | None, split_summary: dict | None = None,
         license_review_status: str = "NOT_ASSESSED",
         contamination_status: str = "NOT_ASSESSED",
         allow_repetition: bool = False) -> dict:
    """Compare independent corpus supply against the requested budget.

    ``available[category]`` contains at least ``documents`` and
    ``effective_tokens`` (packed supervised targets).  No category is topped up
    silently; a shortfall is reported as ``INSUFFICIENT_CORPUS``.
    """
    if type(target_effective_tokens) is not int or target_effective_tokens <= 0:
        raise ValueError("target_effective_tokens must be a positive integer")
    if type(seq_len) is not int or seq_len < 2:
        raise ValueError("seq_len must be an integer >= 2")
    if set(category_token_ratios) != set(V4_CATEGORIES):
        raise ValueError("category_token_ratios must name every v4 category")
    if any(not math.isfinite(value) or value < 0 for value in category_token_ratios.values()):
        raise ValueError("category ratios must be finite and nonnegative")
    if not math.isclose(sum(category_token_ratios.values()), 1.0, abs_tol=1e-6):
        raise ValueError("category ratios must sum to one")
    missing = [category for category in V4_CATEGORIES if category not in available]
    if missing:
        raise ValueError(f"missing available supply for categories: {missing}")
    total_available = sum(available[category]["effective_tokens"] for category in V4_CATEGORIES)
    categories, shortfall = {}, 0
    for category in V4_CATEGORIES:
        supply = available[category]
        if not isinstance(supply.get("documents"), int) or supply["documents"] < 0:
            raise ValueError("available documents must be nonnegative integers")
        if not isinstance(supply.get("effective_tokens"), int) or supply["effective_tokens"] < 0:
            raise ValueError("available effective_tokens must be nonnegative integers")
        target = target_effective_tokens * category_token_ratios[category]
        gap = max(0.0, target - supply["effective_tokens"])
        shortfall += gap
        categories[category] = {
            "available_documents": supply["documents"],
            "available_effective_tokens": supply["effective_tokens"],
            "target_effective_tokens": target,
            "gap_effective_tokens": gap,
            "required_repetition_factor": (max(1.0, target / supply["effective_tokens"])
                                           if supply["effective_tokens"] else None),
            "coverage": (supply["effective_tokens"] / target) if target else 1.0,
            "satisfied": gap <= 1e-9,
            "available_share": (supply["effective_tokens"] / total_available) if total_available else 0.0,
        }
    insufficient = any(not row["satisfied"] for row in categories.values())
    if not tokenizer_fingerprint:
        status = "NOT_ASSESSED"
    elif insufficient:
        status = "INSUFFICIENT_CORPUS"
    else:
        status = "READY_FOR_TRAINING"
    report = {
        "schema_version": 1,
        "status": status,
        "target_effective_train_tokens": target_effective_tokens,
        "seq_len": seq_len,
        "category_token_ratios": dict(category_token_ratios),
        "categories": categories,
        "total_available_effective_tokens": total_available,
        "independent_effective_tokens": total_available,
        "resampled_effective_tokens": ((max(0, target_effective_tokens - total_available))
                                       if allow_repetition else 0),
        "unresolved_gap_effective_tokens": shortfall,
        "total_gap_effective_tokens": shortfall,
        "overall_required_repetition_factor": (max(1.0, target_effective_tokens / total_available)
                                               if total_available else None),
        "allow_repetition": allow_repetition,
        "basis": ("supervised targets: floor((raw_tokens-1)/seq_len)*(seq_len-1); "
                  "category counts are exact positions in the packed stream, not character estimates"),
        "tokenizer_fingerprint": tokenizer_fingerprint,
        "split_summary": split_summary or {},
        "license_review_status": license_review_status,
        "contamination_status": contamination_status,
        "blocking_conditions": [],
    }
    if target_effective_tokens > total_available:
        report["blocking_conditions"].append("TOTAL_CORPUS_SHORTFALL")
    if insufficient:
        report["blocking_conditions"].append("INSUFFICIENT_CORPUS")
        if not allow_repetition:
            report["blocking_conditions"].append("REPETITION_NOT_ALLOWED")
    if license_review_status != "APPROVED":
        report["blocking_conditions"].append("LICENSE_REVIEW_NOT_APPROVED")
    if contamination_status not in ("CHECKED_NO_HITS", "REVIEWED"):
        report["blocking_conditions"].append("CONTAMINATION_NOT_CLEARED")
    report["content_id"] = identity({key: report[key] for key in (
        "target_effective_train_tokens", "seq_len", "category_token_ratios",
        "categories", "tokenizer_fingerprint", "status")})
    return report
