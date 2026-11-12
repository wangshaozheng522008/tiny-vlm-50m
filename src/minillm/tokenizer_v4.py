"""Tokenizer v4 candidate training, fixed-set evaluation and parameter budgets.

Training data is the v4 **train split only**.  Selection metadata is built in one
streaming pass, documents are chosen by a seeded hash priority and per-category
character budgets, and the exact same selection is replayed for every candidate.
Candidates are written to fresh directories and never overwrite a historical
tokenizer.

Evaluation uses a fixed, repository-authored text set instead of training text so
that compression numbers are not inflated by fitting the same documents.  The
report includes Chinese, English, world-knowledge, math, Python, C++ and special
character round-trip metrics.  No single compression number is declared a
winner; the report exposes the trade-off with total parameters, FP32 weight
memory, AdamW state and the T1200 conservative memory budget.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import re
import statistics
import time
import unicodedata
from pathlib import Path

from .checkpoint import atomic_write
from .config import MINILLM_87M, MiniLLMConfig
from .corpus_quality import V4_CATEGORIES
from .corpus_utils import file_info, identity, sha256
from .data import JsonlRecords
from .text_policy import apply_text_policy, validate_policy
from .tokenizer import EOS_ID, MiniLLMTokenizer

CANDIDATE_SCHEMA_VERSION = 1
_NAME = re.compile(r"^[a-z0-9][a-z0-9._-]{1,63}$")
DEFAULT_TRAINING = {
    "max_chars": 30_000_000,
    "seed": 42,
    "category_weights": {"zh_general": 0.40, "zh_encyclopedic": 0.25,
                         "zh_education": 0.15, "en_general": 0.10, "math_logic_code": 0.10},
    "min_documents": 1,
}
DEFAULT_EVALUATION = {"min_gain_fraction": 0.03, "max_param_increase_fraction": 0.02}


def validate_candidates(candidates: list[dict]) -> list[dict]:
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("at least one tokenizer candidate is required")
    seen = set()
    result = []
    for candidate in candidates:
        if not isinstance(candidate, dict) or set(candidate) != {"name", "vocab_size"}:
            raise ValueError("tokenizer candidates need exactly name and vocab_size")
        name, vocab = candidate["name"], candidate["vocab_size"]
        if not isinstance(name, str) or not _NAME.match(name) or name in seen:
            raise ValueError("tokenizer candidate names must be unique and [a-z0-9._-]")
        if type(vocab) is not int or not 264 <= vocab <= 65536:
            raise ValueError("tokenizer vocab_size must be an integer in [264, 65536]")
        seen.add(name)
        result.append({"name": name, "vocab_size": vocab})
    return result


def validate_eval_texts(rows: list[dict]) -> list[dict]:
    allowed = {"zh_prose", "zh_knowledge", "zh_education", "en", "math", "python", "cpp", "special"}
    seen = set()
    result = []
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("text"), str) or not row["text"]:
            raise ValueError("evaluation text rows need non-empty text")
        if row.get("category") not in allowed:
            raise ValueError(f"unknown tokenizer evaluation category: {row.get('category')}")
        if not isinstance(row.get("id"), str) or row["id"] in seen:
            raise ValueError("tokenizer evaluation ids must be unique")
        text_category = row.get("text_category", "auto")
        if text_category not in ("auto", "zh", "en", "code", "mixed", "unknown"):
            raise ValueError("invalid evaluation text_category")
        seen.add(row["id"])
        result.append({"id": row["id"], "category": row["category"],
                       "text_category": text_category, "text": row["text"]})
    if not result:
        raise ValueError("empty tokenizer evaluation text set")
    return result


def load_eval_texts(path: str | Path) -> list[dict]:
    path = Path(path)
    rows = []
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{number}: invalid JSON") from exc
            rows.append(row)
    return validate_eval_texts(rows)


def _selection_metadata(records: JsonlRecords) -> tuple[list[dict], dict]:
    metadata, per_category = [], {}
    for index, offset in enumerate(records.offsets):
        row = records[index]
        category, text = row.get("category"), row.get("text")
        language = row.get("language")
        if category not in V4_CATEGORIES or not isinstance(text, str) or not text:
            raise ValueError(f"train split record {index} has invalid category/text")
        if language not in ("zh", "en", "mixed", "code", "unknown"):
            raise ValueError(f"train split record {index} has invalid language")
        entry = {"index": index, "category": category, "language": language,
                 "characters": len(text), "document_id": row.get("document_id", str(index))}
        metadata.append(entry)
        per_category.setdefault(category, {"characters": 0, "documents": 0})
        per_category[category]["characters"] += len(text)
        per_category[category]["documents"] += 1
    return metadata, per_category


def _category_budgets(per_category: dict, weights: dict, max_chars: int) -> dict[str, int]:
    """Water-filling allocation so exhausted categories do not waste the budget."""
    if set(weights) != set(V4_CATEGORIES):
        raise ValueError("tokenizer category_weights must name every v4 category")
    if any(not math.isfinite(value) or value < 0 for value in weights.values()):
        raise ValueError("tokenizer category weights must be finite and nonnegative")
    if not math.isclose(sum(weights.values()), 1.0, abs_tol=1e-6):
        raise ValueError("tokenizer category weights must sum to one")
    budgets = {category: 0 for category in V4_CATEGORIES}
    active = {category for category in V4_CATEGORIES
              if per_category.get(category, {}).get("characters", 0) > 0 and weights[category] > 0}
    remaining = max_chars
    while active and remaining > 0:
        total_weight = sum(weights[category] for category in active)
        progressed = False
        for category in sorted(active):
            share = int(remaining * weights[category] / total_weight)
            capacity = per_category[category]["characters"] - budgets[category]
            take = min(share, capacity)
            if take > 0:
                budgets[category] += take
                remaining -= take
                progressed = True
            if budgets[category] >= per_category[category]["characters"]:
                active.discard(category)
        if not progressed:
            break
    return budgets


def build_selection(train_path: str | Path, *, category_weights: dict,
                    max_chars: int, seed: int) -> dict:
    """Choose document indices deterministically; returns a bounded manifest."""
    if type(max_chars) is not int or max_chars <= 0:
        raise ValueError("max_chars must be a positive integer")
    if type(seed) is not int:
        raise ValueError("seed must be an integer")
    records = JsonlRecords(train_path)
    try:
        metadata, per_category = _selection_metadata(records)
        if not metadata:
            raise ValueError("empty train split")
        budgets = _category_budgets(per_category, category_weights, max_chars)
        selected: dict[str, list[dict]] = {category: [] for category in V4_CATEGORIES}
        for entry in metadata:
            entry["priority"] = hashlib.sha256(
                f"{seed}:{entry['document_id']}".encode("utf-8")).hexdigest()
        for category in V4_CATEGORIES:
            budget = budgets.get(category, 0)
            if not budget:
                continue
            used = 0
            for entry in sorted((e for e in metadata if e["category"] == category),
                                key=lambda e: (e["priority"], e["document_id"])):
                if used + entry["characters"] > budget:
                    continue
                selected[category].append(entry)
                used += entry["characters"]
                if used >= budget:
                    break
        chosen = [entry for category in V4_CATEGORIES for entry in selected[category]]
        chosen.sort(key=lambda entry: (entry["category"], entry["priority"], entry["document_id"]))
        if not chosen:
            raise ValueError("tokenizer selection is empty; increase max_chars")
        return {"train_path": str(Path(train_path).resolve()),
                "train_sha256": sha256(Path(train_path)),
                "seed": seed, "max_chars": max_chars,
                "category_weights": dict(category_weights),
                "category_budgets": budgets,
                "available_chars_by_category": {category: per_category.get(category, {}).get("characters", 0)
                                                for category in V4_CATEGORIES},
                "selected_documents": len(chosen),
                "selected_chars": sum(entry["characters"] for entry in chosen),
                "selected_by_category": {category: {
                    "documents": len(selected[category]),
                    "characters": sum(entry["characters"] for entry in selected[category])}
                    for category in V4_CATEGORIES},
                "selection": [{key: entry[key] for key in ("index", "category", "language",
                                                           "characters", "document_id", "priority")}
                              for entry in chosen]}
    finally:
        records._handle.close() if getattr(records, "_handle", None) is not None else None


def iter_selected_texts(selection: dict, train_path: str | Path, *, text_policy: str):
    """Replay the exact selected documents in deterministic order."""
    records = JsonlRecords(train_path)
    try:
        for entry in selection["selection"]:
            row = records[entry["index"]]
            text = row["text"]
            language = row["language"]
            if apply_text_policy(text, text_policy, category=language) != text:
                raise ValueError("selected train text is not canonical under the configured text policy")
            yield text
    finally:
        if getattr(records, "_handle", None) is not None:
            records._handle.close()


def selection_digest(selection: dict, train_path: str | Path, *, text_policy: str) -> str:
    digest = hashlib.sha256()
    for text in iter_selected_texts(selection, train_path, text_policy=text_policy):
        data = text.encode("utf-8")
        digest.update(len(data).to_bytes(8, "little"))
        digest.update(data)
    return digest.hexdigest()


def guard_train_only(train_path: str | Path, split_manifest_path: str | Path) -> dict:
    train_path, manifest_path = Path(train_path), Path(split_manifest_path)
    if train_path.name != "train.text.jsonl":
        raise ValueError("tokenizer candidates may only be fitted on train.text.jsonl")
    if not manifest_path.is_file():
        raise ValueError("missing split manifest; tokenizer fitting requires the frozen split plan")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    files = manifest.get("files", {})
    entry = files.get(train_path.name)
    if not entry or file_info(train_path) != entry:
        raise ValueError("train split does not match the split manifest hash")
    for name in ("val.text.jsonl", "test.text.jsonl"):
        if name in files and (train_path.parent / name).exists():
            if file_info(train_path.parent / name) != files[name]:
                raise ValueError(f"held-out split {name} was modified; refusing tokenizer fitting")
    return manifest


def train_candidate(selection: dict, train_path: Path, *, name: str, vocab_size: int,
                    out_root: Path, text_policy: str, split_manifest_sha256: str,
                    training_config: dict, demo: bool = False) -> dict:
    directory = Path(out_root) / name
    if directory.exists() and any(directory.iterdir()):
        raise ValueError(f"refusing to overwrite tokenizer candidate: {directory}")
    directory.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    texts = iter_selected_texts(selection, train_path, text_policy=text_policy)
    tokenizer = MiniLLMTokenizer.train(texts, vocab_size=vocab_size)
    tokenizer.text_policy = text_policy
    target = directory / "tokenizer.json"
    atomic_write(target, lambda handle: handle.write(tokenizer._tok.to_str().encode("utf-8")))
    elapsed = time.monotonic() - started
    metadata = {
        "schema_version": CANDIDATE_SCHEMA_VERSION, "name": name,
        "vocab_size": tokenizer.vocab_size, "fingerprint": tokenizer.fingerprint,
        "special_token_ids": tokenizer.special_ids, "text_policy": text_policy,
        "train_split_sha256": sha256(train_path), "split_manifest_sha256": split_manifest_sha256,
        "training_selection": {key: selection[key] for key in (
            "seed", "max_chars", "category_weights", "category_budgets",
            "selected_documents", "selected_chars", "selected_by_category")},
        "training_corpus_sha256": selection_digest(selection, train_path, text_policy=text_policy),
        "tokenizers_version": importlib.metadata.version("tokenizers"),
        "unicode_version": unicodedata.unidata_version,
        "demo_only": bool(demo), "seconds": round(elapsed, 3),
        "selection_deterministic": True,
    }
    from .corpus_utils import write_json
    write_json(directory / "candidate.json", metadata)
    metadata["files"] = {path.name: file_info(path) for path in sorted(directory.iterdir()) if path.is_file()}
    metadata["content_id"] = identity({key: metadata[key] for key in (
        "name", "vocab_size", "fingerprint", "text_policy", "train_split_sha256",
        "split_manifest_sha256", "training_corpus_sha256", "tokenizers_version")})
    write_json(directory / "candidate-manifest.json", metadata)
    return {"name": name, "vocab_size": tokenizer.vocab_size,
            "fingerprint": tokenizer.fingerprint, "path": str(target.resolve()),
            "directory": str(directory.resolve()), "metadata": metadata,
            "files": metadata["files"]}


def _percentiles(values: list[int]) -> dict:
    if not values:
        return {"p10": 0, "p50": 0, "p90": 0, "mean": 0.0}
    ordered = sorted(values)
    def at(fraction):
        return ordered[min(len(ordered) - 1, int(fraction * (len(ordered) - 1) + 0.5))]
    return {"p10": at(0.1), "p50": at(0.5), "p90": at(0.9),
            "mean": round(statistics.fmean(values), 3)}


def evaluate_tokenizer(tokenizer: MiniLLMTokenizer, texts: list[dict]) -> dict:
    """Measure a tokenizer on fixed, evaluation-only text."""
    categories: dict[str, dict] = {}
    token_lengths: dict[str, list[int]] = {}
    roundtrip_failures, boundary_failures = [], []
    for row in texts:
        category, text = row["category"], row["text"]
        ids = tokenizer.encode(text, category=row["text_category"])
        document_ids = tokenizer.encode_document(text, category=row["text_category"])
        if tokenizer.decode(ids, skip_special_tokens=True) != text:
            roundtrip_failures.append(row["id"])
        if (not document_ids or document_ids[-1] != EOS_ID
                or tokenizer.decode(document_ids, skip_special_tokens=True) != text):
            boundary_failures.append(row["id"])
        bucket = categories.setdefault(category, {
            "documents": 0, "characters": 0, "tokens": 0, "words": 0,
            "characters_per_token": 0.0, "tokens_per_character": 0.0,
            "tokens_per_word": None, "mean_tokens_per_document": 0.0})
        bucket["documents"] += 1
        bucket["characters"] += len(text)
        bucket["tokens"] += len(ids)
        bucket["words"] += max(1, len(text.split()))
        token_lengths.setdefault(category, []).append(len(ids))
    for category, bucket in categories.items():
        bucket["characters_per_token"] = bucket["characters"] / max(bucket["tokens"], 1)
        bucket["tokens_per_character"] = bucket["tokens"] / max(bucket["characters"], 1)
        if category == "en":
            bucket["tokens_per_word"] = bucket["tokens"] / max(bucket["words"], 1)
        bucket["mean_tokens_per_document"] = bucket["tokens"] / max(bucket["documents"], 1)
        bucket["token_length_distribution"] = _percentiles(token_lengths[category])
    zh_chars = sum(bucket["characters"] for category, bucket in categories.items()
                   if category in ("zh_prose", "zh_knowledge", "zh_education"))
    zh_tokens = sum(bucket["tokens"] for category, bucket in categories.items()
                    if category in ("zh_prose", "zh_knowledge", "zh_education"))
    code_tokens = sum(bucket["tokens"] for category, bucket in categories.items()
                      if category in ("python", "cpp"))
    code_chars = sum(bucket["characters"] for category, bucket in categories.items()
                     if category in ("python", "cpp"))
    return {
        "vocab_size": tokenizer.vocab_size, "fingerprint": tokenizer.fingerprint,
        "text_policy": tokenizer.text_policy,
        "categories": categories,
        "zh": {"characters": zh_chars, "tokens": zh_tokens,
               "characters_per_token": zh_chars / max(zh_tokens, 1),
               "tokens_per_character": zh_tokens / max(zh_chars, 1),
               "characters_per_1024_tokens": 1024 * zh_chars / max(zh_tokens, 1)},
        "code": {"characters": code_chars, "tokens": code_tokens,
                 "characters_per_token": code_chars / max(code_tokens, 1),
                 "tokens_per_document": code_tokens / max(
                     sum(b["documents"] for c, b in categories.items() if c in ("python", "cpp")), 1)},
        "roundtrip_ok": not roundtrip_failures,
        "roundtrip_failures": roundtrip_failures,
        "document_boundary_ok": not boundary_failures,
        "document_boundary_failures": boundary_failures,
        "special_token_ids": tokenizer.special_ids,
    }


def _delta(candidate: float, baseline: float) -> float | None:
    if baseline in (None, 0):
        return None
    return (candidate - baseline) / baseline


def compare_candidates(candidates: list[dict], texts: list[dict], *,
                       baseline_path: str | Path | None = None,
                       text_policy: str = "identity-v1",
                       evaluation_config: dict | None = None) -> dict:
    config = {**DEFAULT_EVALUATION, **(evaluation_config or {})}
    for key in config:
        if key not in DEFAULT_EVALUATION:
            raise ValueError(f"unknown tokenizer evaluation field: {key}")
    results = {}
    loaded = {}
    for candidate in candidates:
        tokenizer = MiniLLMTokenizer.load(candidate["path"])
        tokenizer.text_policy = validate_policy(text_policy)
        loaded[candidate["name"]] = tokenizer
        results[candidate["name"]] = evaluate_tokenizer(tokenizer, texts)
    baseline = None
    if baseline_path is not None:
        path = Path(baseline_path)
        if path.is_file():
            tokenizer = MiniLLMTokenizer.load(path)
            tokenizer.text_policy = validate_policy(text_policy)
            baseline = {"path": str(path.resolve()), "sha256": sha256(path),
                        "status": "AVAILABLE", **evaluate_tokenizer(tokenizer, texts)}
            loaded["baseline"] = tokenizer
        else:
            baseline = {"path": str(path), "status": "NOT_AVAILABLE",
                        "reason": "baseline tokenizer file does not exist"}
    comparisons = {}
    for name, result in results.items():
        row = {"vocab_size": result["vocab_size"], "fingerprint": result["fingerprint"],
               "zh_characters_per_token": result["zh"]["characters_per_token"],
               "zh_tokens_per_character": result["zh"]["tokens_per_character"],
               "en_tokens_per_word": result["categories"].get("en", {}).get("tokens_per_word"),
               "code_characters_per_token": result["code"]["characters_per_token"],
               "roundtrip_ok": result["roundtrip_ok"],
               "document_boundary_ok": result["document_boundary_ok"]}
        if baseline and baseline["status"] == "AVAILABLE":
            row["relative_to_baseline"] = {
                "zh_characters_per_token": _delta(result["zh"]["characters_per_token"],
                                                  baseline["zh"]["characters_per_token"]),
                "en_tokens_per_word": _delta(
                    result["categories"].get("en", {}).get("tokens_per_word"),
                    baseline["categories"].get("en", {}).get("tokens_per_word")),
                "code_characters_per_token": _delta(result["code"]["characters_per_token"],
                                                    baseline["code"]["characters_per_token"]),
                "vocab_size": _delta(result["vocab_size"], baseline["vocab_size"]),
            }
        comparisons[name] = row
    recommendation = _selection_hint(comparisons, config)
    return {"schema_version": 1, "status": "COMPARED",
            "fixed_texts": {"documents": len(texts),
                            "categories": sorted({row["category"] for row in texts}),
                            "sha256": identity(texts)},
            "candidates": results, "baseline": baseline, "comparison": comparisons,
            "recommendation": recommendation,
            "selection_rule": ("do not reward Chinese compression alone; compare world-knowledge, "
                               "math/code, total parameters, memory and inference cost; a small gain "
                               "must not justify a materially larger model")}


def _selection_hint(comparisons: dict, config: dict) -> dict:
    if len(comparisons) < 2:
        return {"status": "INSUFFICIENT_CANDIDATES", "reason": "at least two candidates are needed"}
    ordered = sorted(comparisons.items(), key=lambda item: item[1]["vocab_size"])
    small_name, small = ordered[0]
    large_name, large = ordered[-1]
    small_zh = small["zh_characters_per_token"]
    large_zh = large["zh_characters_per_token"]
    gain = (large_zh - small_zh) / max(small_zh, 1e-9)
    param_increase = (large["vocab_size"] - small["vocab_size"]) / max(small["vocab_size"], 1)
    if gain < config["min_gain_fraction"] and param_increase > config["max_param_increase_fraction"]:
        choice, reason = small_name, (
            f"larger vocabulary improves Chinese characters/token by {gain:.2%} "
            f"(< {config['min_gain_fraction']:.0%}) while adding {param_increase:.2%} vocabulary rows")
    elif gain >= 0.05 and param_increase <= 0.08:
        choice, reason = large_name, (
            f"larger vocabulary improves Chinese characters/token by {gain:.2%} at a "
            f"{param_increase:.2%} vocabulary-row cost; still requires explicit human selection")
    else:
        choice, reason = None, (
            f"Chinese gain {gain:.2%} and vocabulary increase {param_increase:.2%} do not fall in a "
            "clear rule; compare world-knowledge, code, memory and inference budgets before choosing")
    return {"status": "HUMAN_DECISION_REQUIRED", "preferred_lower_cost": choice,
            "reason": reason, "requires_explicit_selection": True,
            "smaller": small_name, "larger": large_name,
            "zh_compression_gain": gain, "vocabulary_row_increase": param_increase}


def parameter_budget(vocab_sizes: list[int], *, base_config: MiniLLMConfig | None = None,
                     seq_len: int = 1024, batch: int = 1,
                     hardware: dict | None = None) -> dict:
    """Analytic parameter/memory comparison; never changes the transformer shape."""
    base = (base_config or MINILLM_87M).copy()
    if type(seq_len) is not int or seq_len < 2 or seq_len > base.max_position_embeddings:
        raise ValueError("seq_len must be within the model's position range")
    default_hardware = {"device": "NVIDIA T1200 Laptop", "memory_bytes": 3_865_470_566,
                        "budget_bytes": 3_116_105_728, "adamw_bytes_per_parameter": 20,
                        "overhead_bytes": 805_306_368}
    hardware = {**default_hardware, **(hardware or {})}
    rows = {}
    baseline_total = base.param_breakdown()["total"]
    for vocab in sorted(set(vocab_sizes)):
        if type(vocab) is not int or not 264 <= vocab <= 65536:
            raise ValueError("vocab_size must be an integer in [264, 65536]")
        cfg = base.copy()
        cfg.vocab_size = vocab
        breakdown = cfg.param_breakdown()
        embedding = breakdown["embedding"]
        lm_head_extra = 0 if cfg.tie_word_embeddings else embedding
        total = breakdown["total"] + lm_head_extra
        weight_bytes = total * 4
        grad_bytes = total * 4
        adamw_bytes = total * 8
        optimizer_total = weight_bytes + grad_bytes + adamw_bytes
        conservative = total * hardware["adamw_bytes_per_parameter"] + hardware["overhead_bytes"]
        logits_bytes = batch * seq_len * vocab * 4
        rows[str(vocab)] = {
            "embedding_parameters": embedding,
            "lm_head_parameters": embedding,
            "lm_head_tied": cfg.tie_word_embeddings,
            "lm_head_additional_parameters": lm_head_extra,
            "transformer_parameters": breakdown["total"] - embedding,
            "total_parameters": total,
            "delta_parameters_vs_baseline": total - baseline_total,
            "delta_fraction_vs_baseline": (total - baseline_total) / baseline_total,
            "reference_87m_parameters": 87_000_000,
            "delta_vs_87m_reference": total - 87_000_000,
            "exceeds_87m_reference": total > 87_000_000,
            "fp32_weight_bytes": weight_bytes,
            "fp32_weight_mib": weight_bytes / 1024 ** 2,
            "adamw_state_bytes": adamw_bytes,
            "gradient_bytes": grad_bytes,
            "training_state_bytes_weights_grads_adamw": optimizer_total,
            "conservative_training_bytes_20_per_param_plus_overhead": conservative,
            "logits_bytes_per_sequence": logits_bytes,
            "conservative_fits_t1200_budget": conservative <= hardware["budget_bytes"],
            "device_memory_bytes": hardware["memory_bytes"],
            "device_budget_bytes": hardware["budget_bytes"],
        }
    return {"schema_version": 1, "base_preset": "minillm-87m",
            "architecture_unchanged": True, "only_vocabulary_changed": ["vocab_size"],
            "baseline_vocab_size": base.vocab_size,
            "baseline_total_parameters": baseline_total,
            "architecture": {key: getattr(base, key) for key in (
                "hidden_size", "num_hidden_layers", "num_attention_heads",
                "num_key_value_heads", "head_dim", "intermediate_size",
                "max_position_embeddings", "tie_word_embeddings")},
            "formulas": {
                "embedding": "vocab_size * hidden_size",
                "lm_head": "tied to embedding (zero additional parameters); untied would add vocab_size * hidden_size",
                "total": "embedding + attention layers + SwiGLU layers + RMSNorm",
                "fp32_weights": "total_parameters * 4 bytes",
                "adamw": "weights (4) + gradients (4) + exp_avg (4) + exp_avg_sq (4) = 16 bytes/parameter",
                "conservative_training": "20 bytes/parameter + 768 MiB overhead (project pilot rule)",
                "logits": "batch * seq_len * vocab_size * 4 bytes",
            },
            "hardware": hardware, "seq_len": seq_len, "batch": batch,
            "vocab_sizes": rows}
