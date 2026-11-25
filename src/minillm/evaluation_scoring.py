"""Whole-answer scoring, optional lexical diagnostics, and manual review."""
from __future__ import annotations

import math
import re
from decimal import Decimal, InvalidOperation

RULES = {"exact_match", "normalized_match", "numeric_match", "multiple_choice", "manual_review", "keyword_match"}
NORMALIZATION = {"strip", "collapse_whitespace", "casefold", "terminal_punctuation"}
NUMBER = re.compile(r"[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?\Z")


def normalize_answer(text, allowed):
    for rule in allowed:
        if rule == "strip":
            text = text.strip()
        elif rule == "collapse_whitespace":
            text = re.sub(r"\s+", " ", text)
        elif rule == "casefold":
            text = text.casefold()
        elif rule == "terminal_punctuation":
            text = text.rstrip("。.!！?？")
        else:
            raise ValueError(f"unknown normalization rule: {rule}")
    return text


def number(value):
    if isinstance(value, bool) or not NUMBER.fullmatch(str(value).strip()):
        raise ValueError("numeric answer must be one finite decimal, without units or explanation")
    try:
        result = Decimal(str(value).strip())
        if not result.is_finite() or abs(result.adjusted()) > 1000:
            raise ValueError("numeric answer outside supported range")
        return result
    except InvalidOperation as exc:
        raise ValueError("invalid decimal") from exc


def validate_task(task):
    for field in ("id", "category", "prompt"):
        if not isinstance(task.get(field), str) or not task[field].strip():
            raise ValueError(f"benchmark requires nonempty {field}")
    if task.get("generation_mode") not in ("raw", "chat", "both"):
        raise ValueError("generation_mode must be raw, chat or both")
    if type(task.get("max_new_tokens", 32)) is not int or task.get("max_new_tokens", 32) <= 0:
        raise ValueError("invalid task max_new_tokens")
    rule = task.get("scoring")
    if rule not in RULES:
        raise ValueError(f"unknown scoring rule: {rule}")
    if rule == "manual_review":
        return
    reference = task.get("reference")
    if not isinstance(reference, str) or not reference.strip():
        raise ValueError("automatic scoring requires a nonempty reference string")
    allowed = task.get("normalization", ["strip"])
    if not isinstance(allowed, list) or any(x not in NORMALIZATION for x in allowed):
        raise ValueError("invalid explicit normalization rules")
    aliases = task.get("aliases", [])
    if (not isinstance(aliases, list) or any(not isinstance(alias, str) or not alias.strip() for alias in aliases)
            or (aliases and rule not in ("normalized_match", "keyword_match"))):
        raise ValueError("invalid answer aliases")
    if rule == "keyword_match" and any(not normalize_answer(a, allowed) for a in [reference, *aliases]):
        raise ValueError("empty normalized keyword")
    if rule == "numeric_match":
        number(reference)
        for key in ("atol", "rtol"):
            value = task.get(key, 0)
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError("numeric tolerances must be finite and nonnegative")
        if task.get("numeric_type", "decimal") not in ("decimal", "integer"):
            raise ValueError("numeric_type must be decimal or integer")
        if task.get("numeric_type") == "integer" and number(reference) != number(reference).to_integral():
            raise ValueError("integer reference is fractional")
    if rule == "multiple_choice":
        options = task.get("options")
        if (not isinstance(options, dict) or len(options) < 2
                or any(not re.fullmatch("[A-Z]", k) or not isinstance(v, str) or not v for k, v in options.items())
                or reference not in options):
            raise ValueError("multiple_choice requires valid options and one reference letter")
        if any(f"{k}." not in task["prompt"] and f"{k}：" not in task["prompt"] for k in options):
            raise ValueError("choice labels must already appear in the prompt")


def score_answer(output, task):
    validate_task(task)
    rule = task["scoring"]
    if rule == "manual_review":
        return {"correct": None, "status": "manual_review"}
    reference = task["reference"]
    if rule == "exact_match":
        correct = output == reference
    elif rule in ("normalized_match", "keyword_match"):
        allowed = task.get("normalization", ["strip"])
        actual = normalize_answer(output, allowed)
        expected = [normalize_answer(value, allowed) for value in (reference, *task.get("aliases", []))]
        if rule == "keyword_match":
            return {"correct": None, "status": "manual_review", "keyword_hit": any(a in actual for a in expected)}
        correct = actual in expected
    elif rule == "multiple_choice":
        correct = output.strip() == reference
    else:
        try:
            actual, expected = number(output), number(reference)
            correct = abs(actual - expected) <= Decimal(str(task.get("atol", 0))) + Decimal(str(task.get("rtol", 0))) * abs(expected)
            if task.get("numeric_type") == "integer":
                correct = correct and actual == actual.to_integral()
        except ValueError:
            correct = False
    return {"correct": bool(correct), "status": "scored"}


def aggregate_scores(predictions):
    result = {}
    for mode in ("raw", "chat"):
        rows = [r for r in predictions if r.get("kind") == "benchmark" and r["generation_mode"] == mode]
        categories = {}
        for category in sorted({r["category"] for r in rows}):
            subset = [r for r in rows if r["category"] == category]
            scored = [r for r in subset if r["score"]["correct"] is not None]
            correct = sum(r["score"]["correct"] for r in scored)
            categories[category] = {"correct": correct, "scored": len(scored),
                                    "manual_review": len(subset) - len(scored),
                                    "accuracy": correct / len(scored) if scored else None}
            keyword_hits = [r["score"]["keyword_hit"] for r in subset if "keyword_hit" in r["score"]]
            if keyword_hits:
                categories[category]["keyword_matching"] = {"hits": sum(keyword_hits), "tasks": len(keyword_hits)}
        result[mode] = {"categories": categories, "tasks": len(rows),
                        "interpretation": "completion ability" if mode == "raw" else "instruction following; pretrain-only low scores require stage context"}
    return result
