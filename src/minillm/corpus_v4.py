"""MiniLLM corpus v4: provenance, quality, dedup, split, tokenizer and freeze.

The v4 pipeline is a versioned orchestrator over the repository's existing
primitives: ``corpus_quality`` normalization/classification, ``corpus_utils``
disk-backed stages and identities, ``corpus.tokenize_split`` for the uint16 BIN
format, ``data`` streaming readers, and the frozen evaluation-asset checks.  It
adds what the earlier pipeline did not have: an explicit source/license
registry, per-record provenance, structured KEEP/REVIEW/REJECT findings, MinHash
LSH near-duplicate retrieval, family-aware splitting, contamination audit,
effective-token planning and a validated freeze manifest.

Stages are:

    INGESTED -> NORMALIZED -> QUALITY_REVIEWED -> DEDUPLICATED -> SPLIT
    -> TOKENIZER_EVALUATED -> TOKENIZED -> AUDITED -> FROZEN

Each stage marker is published last and records hashes of every artifact.  An
existing output directory is never modified with different inputs; callers must
choose a new ``--out`` for a new version.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import re
import shutil
import sqlite3
import subprocess
import tempfile
import time
import unicodedata
from pathlib import Path

from .checkpoint import atomic_write
from .corpus import tokenize_split
from .corpus_budget import plan as plan_budget
from .corpus_budget import read_token_index, supervision_by_category
from .corpus_contamination import audit as contamination_audit
from .corpus_dedup import MinHashLSHIndex, NearDupPolicy, chunk_features, chunk_text, exact_jaccard, shingles
from .corpus_quality import (QUALITY_RULES_VERSION, V4_CATEGORIES, aggregate_action,
                             content_hash, detect_language, evaluate_quality, infer_category,
                             merge_quality_v4, normalise)
from .corpus_utils import (CorpusError, Stages, database_output, file_info, histogram_add,
                           identity, json_bytes, sha256, verify_file, write_json)
from .data import stream_corpus_records
from .text_policy import IDENTITY, apply_text_policy, validate_policy
from .tokenizer import MiniLLMTokenizer

V4_SCHEMA_VERSION = 4
LICENSE_STATUSES = ("APPROVED", "REVIEW_REQUIRED", "REJECTED")
PROVENANCE_STATUSES = ("VERIFIED", "UPSTREAM_DECLARED", "UNVERIFIED", "DEMO_ONLY")
LICENSE_UNVERIFIED = "LICENSE_UNVERIFIED"
SPLITS = ("train", "val", "test")
STAGES = ("INGESTED", "NORMALIZED", "QUALITY_REVIEWED", "DEDUPLICATED", "SPLIT",
          "TOKENIZER_EVALUATED", "TOKENIZED", "AUDITED", "FROZEN")
STATUSES = ("READY", "NOT_READY", "NOT_ASSESSED")


def implementation_sha256() -> str:
    """Code identity: a changed reader/cleaner/tokenizer must not reuse old stages."""
    names = ("corpus_v4.py", "corpus.py", "corpus_quality.py", "corpus_dedup.py",
             "corpus_contamination.py", "corpus_budget.py", "tokenizer_v4.py",
             "data.py", "text_policy.py", "tokenizer.py")
    return identity({name: sha256(Path(__file__).with_name(name)) for name in names})

DEFAULT_RUNTIME = {"sqlite_cache_mib": 16, "commit_every": 1000, "buffer_bytes": 1048576}
DEFAULT_TARGETS = {"effective_train_tokens": 100_000_000, "seq_len": 1024,
                   "category_token_ratios": {"zh_general": 0.40, "zh_encyclopedic": 0.25,
                                             "zh_education": 0.15, "en_general": 0.10,
                                             "math_logic_code": 0.10}}
DEFAULT_LICENSE_POLICY = {"approved_only": True, "demo_allow_review_required": True}
DEFAULT_DEDUP = {"exact": True,
                 "near": {"method": "minhash_lsh", "threshold": 0.85, "num_bins": 128,
                          "bands": 32, "rows": 4, "seed": 42, "zh_ngram": 5,
                          "en_ngram": 3, "code_ngram": 8, "include_code": False,
                          "max_shingles_per_chunk": 20000, "max_candidates_per_chunk": 512,
                          "chunk_chars": 20000}}
DEFAULT_SPLIT = {"seed": 42, "ratios": {"train": 0.96, "val": 0.02, "test": 0.02},
                 "strategy": "family_cluster_hash", "verify_near_cross_split": True}
DEFAULT_TOKENIZER = {
    "candidates": [{"name": "v4-24576", "vocab_size": 24576},
                   {"name": "v4-32768", "vocab_size": 32768}],
    "selected": None,
    "output_root": "artifacts/tokenizers",
    "eval_asset": "tests/fixtures/corpus_v4/tokenizer-eval.jsonl",
    "baseline": None,
    "training": {"max_chars": 30_000_000, "seed": 42, "min_documents": 1,
                 "category_weights": {"zh_general": 0.40, "zh_encyclopedic": 0.25,
                                      "zh_education": 0.15, "en_general": 0.10,
                                      "math_logic_code": 0.10}},
    "evaluation": {"min_gain_fraction": 0.03, "max_param_increase_fraction": 0.02},
}
DEFAULT_BUDGET = {"chars_per_token": {"zh": 1.8, "en": 4.0, "code": 3.0,
                                      "mixed": 2.0, "unknown": 3.0},
                  "allow_repetition": False}
DEFAULT_CONTAMINATION = {
    "enabled": True, "min_prompt_chars": 12, "long_span_containment": 0.8,
    "max_hits": 5000, "decisions": None,
    "assets": [
        {"path": "data/evaluation/minillm-zh-benchmark.jsonl", "kind": "scored"},
        {"path": "data/evaluation/fixed-prompts.jsonl", "kind": "fixed_prompts"},
        {"path": "data/evaluation/pretrain-v1/candidates.jsonl", "kind": "candidates"},
        {"path": "data/evaluation/pretrain-v1/prompts.jsonl", "kind": "prompts"},
        {"path": "data/evaluation/chinese-foundation-v1/candidates.jsonl", "kind": "candidates"},
        {"path": "data/evaluation/chinese-foundation-v1/prompts.jsonl", "kind": "prompts"},
    ]}
DEFAULT_HARDWARE = {"device": "NVIDIA T1200 Laptop", "memory_bytes": 3_865_470_566,
                    "budget_bytes": 3_116_105_728, "adamw_bytes_per_parameter": 20,
                    "overhead_bytes": 805_306_368}
DEFAULT_AUDIT = {"retokenize": "full", "sample_documents": 2000}
DEFAULT_DEMO = {"sources_manifest": "tests/fixtures/corpus_v4/demo-sources.json",
                "effective_train_tokens": 2048, "seq_len": 128,
                "tokenizer_candidates": [{"name": "demo-320", "vocab_size": 320},
                                         {"name": "demo-384", "vocab_size": 384}],
                "eval_asset": "tests/fixtures/corpus_v4/tokenizer-eval.jsonl"}
TOP_LEVEL = {"schema_version", "corpus_version", "dataset_name", "seed", "text_policy",
             "targets", "sources", "license_policy", "quality", "deduplication", "split",
             "tokenizer", "budget", "contamination", "runtime", "hardware", "audit", "demo"}


def _deep_merge(defaults: dict, overrides: dict, name: str, maps: tuple[str, ...] = ()) -> dict:
    if not isinstance(overrides, dict):
        raise CorpusError(f"{name} must be a mapping")
    unknown = set(overrides) - set(defaults)
    if unknown:
        raise CorpusError(f"unknown {name} fields: {sorted(unknown)}")
    result = copy.deepcopy(defaults)
    for key, value in overrides.items():
        if isinstance(defaults[key], dict) and isinstance(value, dict) and key not in maps:
            result[key] = _deep_merge(defaults[key], value, f"{name}.{key}")
        elif key in maps:
            if not isinstance(value, dict):
                raise CorpusError(f"{name}.{key} must be a mapping")
            result[key] = dict(value)
        else:
            result[key] = value
    return result


def _load_raw(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in (".yaml", ".yml"):
        try:
            import yaml
        except ImportError as exc:  # pragma: no cover - dependency is declared
            raise CorpusError("YAML configuration requires PyYAML; install requirements.txt") from exc
        raw = yaml.safe_load(text)
    else:
        raw = json.loads(text)
    if not isinstance(raw, dict):
        raise CorpusError("corpus v4 configuration must be a mapping")
    return raw


def _split_ratios(value: dict) -> dict:
    if not isinstance(value, dict) or set(value) != set(SPLITS):
        raise CorpusError("split.ratios must contain exactly train/val/test")
    if any(not isinstance(item, (int, float)) or isinstance(item, bool) or not math.isfinite(item)
           or item < 0 for item in value.values()):
        raise CorpusError("split ratios must be finite and nonnegative")
    if not math.isclose(sum(value.values()), 1.0, abs_tol=1e-6) or value["train"] <= 0:
        raise CorpusError("split ratios must sum to one and include a positive train ratio")
    return {key: float(value[key]) for key in SPLITS}


def validate_config(raw: dict) -> dict:
    unknown = set(raw) - TOP_LEVEL
    if unknown:
        raise CorpusError(f"unknown corpus v4 configuration fields: {sorted(unknown)}")
    if raw.get("schema_version") != V4_SCHEMA_VERSION:
        raise CorpusError(f"corpus v4 requires schema_version={V4_SCHEMA_VERSION}")
    config = copy.deepcopy(raw)
    for key, value in (("corpus_version", "minillm-corpus-v4-draft"),
                       ("dataset_name", "minillm-corpus-v4"), ("seed", 42),
                       ("text_policy", IDENTITY)):
        config.setdefault(key, value)
    if not isinstance(config["corpus_version"], str) or not re.fullmatch(r"[A-Za-z0-9._-]{3,80}", config["corpus_version"]):
        raise CorpusError("corpus_version must be 3-80 filename-safe characters")
    if not isinstance(config["dataset_name"], str) or not config["dataset_name"].strip():
        raise CorpusError("dataset_name must be a nonempty string")
    if type(config["seed"]) is not int:
        raise CorpusError("seed must be an integer")
    config["text_policy"] = validate_policy(config["text_policy"])
    if not isinstance(config.get("sources", []), list):
        raise CorpusError("sources must be a list")
    config["targets"] = _deep_merge(DEFAULT_TARGETS, config.get("targets", {}), "targets",
                                    maps=("category_token_ratios",))
    targets = config["targets"]
    if type(targets["effective_train_tokens"]) is not int or targets["effective_train_tokens"] <= 0:
        raise CorpusError("targets.effective_train_tokens must be a positive integer")
    if type(targets["seq_len"]) is not int or not 2 <= targets["seq_len"] <= 2048:
        raise CorpusError("targets.seq_len must be an integer in [2, 2048]")
    ratios = targets["category_token_ratios"]
    if set(ratios) != set(V4_CATEGORIES):
        raise CorpusError(f"targets.category_token_ratios must name exactly {list(V4_CATEGORIES)}")
    if any(not isinstance(value, (int, float)) or isinstance(value, bool)
           or not math.isfinite(value) or value < 0 for value in ratios.values()):
        raise CorpusError("category token ratios must be finite and nonnegative")
    if not math.isclose(sum(ratios.values()), 1.0, abs_tol=1e-6):
        raise CorpusError("category token ratios must sum to one")
    config["license_policy"] = _deep_merge(DEFAULT_LICENSE_POLICY, config.get("license_policy", {}), "license_policy")
    if type(config["license_policy"]["approved_only"]) is not bool or type(
            config["license_policy"]["demo_allow_review_required"]) is not bool:
        raise CorpusError("license_policy flags must be booleans")
    config["quality"] = merge_quality_v4(config.get("quality", {}))
    config["deduplication"] = _deep_merge(DEFAULT_DEDUP, config.get("deduplication", {}),
                                          "deduplication", maps=("near",))
    if type(config["deduplication"]["exact"]) is not bool:
        raise CorpusError("deduplication.exact must be boolean")
    try:
        NearDupPolicy.from_config(config["deduplication"]["near"])
    except ValueError as exc:
        raise CorpusError(str(exc)) from exc
    config["split"] = _deep_merge(DEFAULT_SPLIT, config.get("split", {}), "split", maps=("ratios",))
    config["split"]["ratios"] = _split_ratios(config["split"]["ratios"])
    if type(config["split"]["seed"]) is not int:
        raise CorpusError("split.seed must be an integer")
    if config["split"]["strategy"] != "family_cluster_hash":
        raise CorpusError("only the family_cluster_hash split strategy is implemented")
    if type(config["split"]["verify_near_cross_split"]) is not bool:
        raise CorpusError("split.verify_near_cross_split must be boolean")
    config["tokenizer"] = _deep_merge(DEFAULT_TOKENIZER, config.get("tokenizer", {}), "tokenizer",
                                      maps=("training", "evaluation"))
    from .tokenizer_v4 import validate_candidates
    try:
        config["tokenizer"]["candidates"] = validate_candidates(config["tokenizer"]["candidates"])
        if config["tokenizer"].get("demo_candidates") is not None:
            config["tokenizer"]["demo_candidates"] = validate_candidates(config["tokenizer"]["demo_candidates"])
    except ValueError as exc:
        raise CorpusError(str(exc)) from exc
    if config["tokenizer"]["selected"] is not None:
        names = {candidate["name"] for candidate in config["tokenizer"]["candidates"]}
        if config["tokenizer"]["selected"] not in names:
            raise CorpusError("tokenizer.selected must name a configured candidate or be null")
    if not isinstance(config["tokenizer"]["output_root"], str) or not config["tokenizer"]["output_root"]:
        raise CorpusError("tokenizer.output_root must be a nonempty path")
    training = config["tokenizer"]["training"]
    if type(training.get("max_chars")) is not int or training["max_chars"] <= 0:
        raise CorpusError("tokenizer.training.max_chars must be a positive integer")
    if type(training.get("seed")) is not int:
        raise CorpusError("tokenizer.training.seed must be an integer")
    if set(training.get("category_weights", {})) != set(V4_CATEGORIES):
        raise CorpusError("tokenizer.training.category_weights must name every v4 category")
    if any(not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value)
           or value < 0 for value in training["category_weights"].values()) or not math.isclose(
               sum(training["category_weights"].values()), 1.0, abs_tol=1e-6):
        raise CorpusError("tokenizer category weights must be finite, nonnegative and sum to one")
    for key in ("min_gain_fraction", "max_param_increase_fraction"):
        value = config["tokenizer"]["evaluation"].get(key)
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not 0 <= value <= 1:
            raise CorpusError(f"tokenizer.evaluation.{key} must be in [0, 1]")
    config["budget"] = _deep_merge(DEFAULT_BUDGET, config.get("budget", {}), "budget",
                                   maps=("chars_per_token",))
    if type(config["budget"]["allow_repetition"]) is not bool:
        raise CorpusError("budget.allow_repetition must be boolean")
    for language, value in config["budget"]["chars_per_token"].items():
        if language not in ("zh", "en", "code", "mixed", "unknown") or not isinstance(value, (int, float)) \
                or isinstance(value, bool) or not math.isfinite(value) or value <= 0:
            raise CorpusError("budget.chars_per_token must be positive for each text language")
    config["contamination"] = _deep_merge(DEFAULT_CONTAMINATION, config.get("contamination", {}), "contamination")
    if type(config["contamination"]["enabled"]) is not bool:
        raise CorpusError("contamination.enabled must be boolean")
    if not isinstance(config["contamination"]["assets"], list):
        raise CorpusError("contamination.assets must be a list")
    if type(config["contamination"]["min_prompt_chars"]) is not int \
            or not 1 <= config["contamination"]["min_prompt_chars"] <= 200:
        raise CorpusError("contamination.min_prompt_chars must be in [1, 200]")
    if not 0.5 <= config["contamination"]["long_span_containment"] <= 1.0:
        raise CorpusError("contamination.long_span_containment must be in [0.5, 1.0]")
    config["runtime"] = _deep_merge(DEFAULT_RUNTIME, config.get("runtime", {}), "runtime")
    if not 1 <= config["runtime"]["sqlite_cache_mib"] <= 128 or config["runtime"]["commit_every"] <= 0 \
            or not 2 <= config["runtime"]["buffer_bytes"] <= 16 * 1048576:
        raise CorpusError("invalid runtime bounds")
    config["hardware"] = _deep_merge(DEFAULT_HARDWARE, config.get("hardware", {}), "hardware")
    config["audit"] = _deep_merge(DEFAULT_AUDIT, config.get("audit", {}), "audit")
    if config["audit"]["retokenize"] not in ("full", "sample", "none"):
        raise CorpusError("audit.retokenize must be full, sample or none")
    if type(config["audit"]["sample_documents"]) is not int or config["audit"]["sample_documents"] < 1:
        raise CorpusError("audit.sample_documents must be a positive integer")
    config["demo"] = _deep_merge(DEFAULT_DEMO, config.get("demo", {}), "demo")
    if not isinstance(config["demo"]["sources_manifest"], str) or not config["demo"]["sources_manifest"]:
        raise CorpusError("demo.sources_manifest must be a path")
    if type(config["demo"]["effective_train_tokens"]) is not int or config["demo"]["effective_train_tokens"] <= 0:
        raise CorpusError("demo.effective_train_tokens must be a positive integer")
    if type(config["demo"]["seq_len"]) is not int or not 2 <= config["demo"]["seq_len"] <= 2048:
        raise CorpusError("demo.seq_len must be in [2, 2048]")
    try:
        config["demo"]["tokenizer_candidates"] = validate_candidates(config["demo"]["tokenizer_candidates"])
    except ValueError as exc:
        raise CorpusError(str(exc)) from exc
    return config


def load_config(path: str | Path) -> dict:
    return validate_config(_load_raw(Path(path)))


def _iso_date(value: str) -> bool:
    return bool(re.fullmatch(r"\d{4}-\d{2}-\d{2}(?:[T ][0-9:+-]{4,20})?", value))


def build_registry(sources: list[dict], *, demo: bool, license_policy: dict) -> dict:
    """Resolve source status without inferring permission from a licence string."""
    included, excluded, seen = [], [], set()
    allowed_fields = {"source_id", "source_name", "source_uri", "license", "license_status",
                      "license_evidence", "license_reviewer", "license_reviewed_at",
                      "category", "language", "paths", "priority", "enabled",
                      "provenance_status", "acquisition_date", "author", "original_format",
                      "transformation_history", "family_policy", "notes"}
    for spec in sources:
        if not isinstance(spec, dict):
            raise CorpusError("each source must be a mapping")
        unknown = set(spec) - allowed_fields
        if unknown:
            raise CorpusError(f"unknown source fields: {sorted(unknown)}")
        source_id = spec.get("source_id")
        if not isinstance(source_id, str) or not re.fullmatch(r"[A-Za-z0-9._:-]{2,80}", source_id):
            raise CorpusError("source_id must be 2-80 filename-safe characters")
        if source_id in seen:
            raise CorpusError(f"duplicate source_id: {source_id}")
        seen.add(source_id)
        source_name = spec.get("source_name")
        if not isinstance(source_name, str) or not source_name.strip() or len(source_name) > 200:
            raise CorpusError(f"source {source_id}: source_name is required")
        source_uri = spec.get("source_uri")
        if source_uri is not None and (not isinstance(source_uri, str) or len(source_uri) > 2048):
            raise CorpusError(f"source {source_id}: invalid source_uri")
        license_value = spec.get("license")
        if not isinstance(license_value, str) or not license_value.strip():
            license_value = LICENSE_UNVERIFIED
        elif license_value.strip().lower() in ("unknown", "unverified", "license_unverified"):
            license_value = LICENSE_UNVERIFIED
        else:
            license_value = license_value.strip()
        status = spec.get("license_status", "REVIEW_REQUIRED")
        if status not in LICENSE_STATUSES:
            raise CorpusError(f"source {source_id}: license_status must be one of {LICENSE_STATUSES}")
        if license_value == LICENSE_UNVERIFIED and status == "APPROVED":
            raise CorpusError(f"source {source_id}: an unverified licence cannot be APPROVED")
        evidence = spec.get("license_evidence", "")
        reviewer = spec.get("license_reviewer", "")
        reviewed_at = spec.get("license_reviewed_at", "")
        for field, value in (("license_evidence", evidence), ("license_reviewer", reviewer),
                             ("license_reviewed_at", reviewed_at)):
            if value is None:
                value = ""
            if not isinstance(value, str) or len(value) > 4000:
                raise CorpusError(f"source {source_id}: {field} must be a string")
        if status == "APPROVED":
            if not evidence.strip() or not reviewer.strip() or not _iso_date(reviewed_at):
                raise CorpusError(f"source {source_id}: APPROVED requires evidence, reviewer and an ISO review date")
        category = spec.get("category", "auto")
        if category not in (*V4_CATEGORIES, "auto"):
            raise CorpusError(f"source {source_id}: invalid category")
        language = spec.get("language", "auto")
        if language not in ("zh", "en", "code", "mixed", "unknown", "auto"):
            raise CorpusError(f"source {source_id}: invalid language")
        provenance = spec.get("provenance_status", "UNVERIFIED")
        if provenance not in PROVENANCE_STATUSES:
            raise CorpusError(f"source {source_id}: invalid provenance_status")
        paths = spec.get("paths")
        if not isinstance(paths, list) or not paths or any(not isinstance(path, str) or not path for path in paths):
            raise CorpusError(f"source {source_id}: paths must be a nonempty list of strings")
        priority = spec.get("priority", 0)
        if type(priority) is not int:
            raise CorpusError(f"source {source_id}: priority must be an integer")
        if type(spec.get("enabled", True)) is not bool:
            raise CorpusError(f"source {source_id}: enabled must be boolean")
        family_policy = spec.get("family_policy", "document")
        if family_policy not in ("document", "file"):
            raise CorpusError(f"source {source_id}: family_policy must be document or file")
        entry = {
            "source_id": source_id, "source_name": source_name, "source_uri": source_uri,
            "license": license_value, "license_status": status,
            "license_evidence": evidence, "license_reviewer": reviewer,
            "license_reviewed_at": reviewed_at, "category": category, "language": language,
            "paths": list(paths), "priority": priority, "enabled": spec.get("enabled", True),
            "provenance_status": provenance,
            "acquisition_date": spec.get("acquisition_date"),
            "author": spec.get("author"), "original_format": spec.get("original_format"),
            "transformation_history": spec.get("transformation_history", []),
            "family_policy": family_policy, "notes": spec.get("notes", ""),
        }
        reason = None
        if not entry["enabled"]:
            reason = "disabled"
        elif status == "REJECTED":
            reason = "license_rejected"
        elif status != "APPROVED":
            if demo and license_policy.get("demo_allow_review_required", True):
                entry["demo_only"] = True
            else:
                reason = "license_review_required"
        if reason is None and not demo:
            if not license_policy.get("approved_only", True):
                raise CorpusError("formal v4 builds require license_policy.approved_only=true")
            if provenance == "UNVERIFIED":
                reason = "provenance_unverified"
            elif provenance == "DEMO_ONLY":
                reason = "demo_provenance_in_formal_build"
        if reason:
            excluded.append({**entry, "reason": reason})
        else:
            included.append(entry)
    counts = {status: sum(1 for entry in included + excluded if entry["license_status"] == status)
              for status in LICENSE_STATUSES}
    counts["included"] = len(included)
    counts["excluded"] = len(excluded)
    manifest = {"sources": included, "excluded": excluded, "counts": counts,
                "license_policy": license_policy}
    manifest["sha256"] = identity(manifest)
    return manifest


def inventory(registry: dict, base: Path) -> list[dict]:
    files = []
    for source in registry["sources"]:
        for pattern in source["paths"]:
            candidate = Path(pattern) if Path(pattern).is_absolute() else base / pattern
            paths = candidate.rglob("*") if candidate.is_dir() else (candidate,)
            for path in sorted(paths):
                if path.suffix.lower() not in (".jsonl", ".txt"):
                    continue
                if not path.is_file():
                    raise CorpusError(f"missing source file: {path}")
                if len(files) >= 10000:
                    raise CorpusError("at most 10000 input files; consolidate sources")
                files.append({"source_id": source["source_id"], "path": str(path.resolve()),
                              **file_info(path)})
    if not files:
        raise CorpusError("no JSONL/TXT source files found for the admitted sources")
    return sorted(files, key=lambda item: (item["source_id"], item["sha256"], item["path"]))


def status_report(config: dict, base: Path, *, demo: bool = False) -> dict:
    """Pre-build readiness: no corpus is fabricated when no source is approved."""
    original_base = base
    sources = config["sources"]
    if demo:
        manifest_path = (base / config["demo"]["sources_manifest"]).resolve()
        if not manifest_path.is_file():
            return {"status": "NOT_ASSESSED", "blockers": ["DEMO_SOURCES_MISSING"],
                    "sources_manifest": str(manifest_path)}
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
        sources = raw.get("sources", [])
        base = manifest_path.parent
    registry = build_registry(sources, demo=demo, license_policy=config["license_policy"])
    blockers = []
    if not registry["sources"]:
        blockers.append("NO_ADMISSIBLE_SOURCES")
    if not demo and registry["counts"].get("APPROVED", 0) == 0:
        blockers.append("NO_APPROVED_SOURCES")
    if not demo and config["tokenizer"]["selected"] is None:
        blockers.append("TOKENIZER_NOT_SELECTED")
    if not config["contamination"]["enabled"]:
        blockers.append("CONTAMINATION_AUDIT_DISABLED")
    if not all((original_base / asset["path"]).is_file() for asset in config["contamination"]["assets"]
               if asset.get("enabled", True)):
        blockers.append("CONTAMINATION_ASSET_MISSING")
    status = "NOT_READY" if blockers else ("DEMO_READY" if demo else "READY_FOR_BUILD")
    return {"schema_version": 1, "status": status, "blockers": sorted(set(blockers)),
            "demo": demo, "counts": registry["counts"],
            "sources": [{key: source[key] for key in (
                "source_id", "source_name", "license", "license_status", "license_reviewer",
                "license_reviewed_at", "category", "provenance_status")}
                for source in registry["sources"]],
            "excluded": [{key: source[key] for key in (
                "source_id", "license", "license_status", "reason")}
                for source in registry["excluded"]],
            "source_manifest_sha256": registry["sha256"],
            "target_effective_train_tokens": (config["demo"]["effective_train_tokens"] if demo
                                              else config["targets"]["effective_train_tokens"]),
            "limitations": "No download, no network licence inference; APPROVED is an explicit reviewer decision."}


class _DSU:
    def __init__(self):
        self.parent: dict[str, str] = {}

    def add(self, item: str):
        self.parent.setdefault(item, item)

    def find(self, item: str) -> str:
        self.add(item)
        root = item
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[item] != item:
            self.parent[item], item = root, self.parent[item]
        return root

    def union(self, a: str, b: str):
        a, b = self.find(a), self.find(b)
        if a != b:
            lower, upper = sorted((a, b))
            self.parent[upper] = lower


def _rep_key(row) -> tuple:
    return (-int(row["priority"]), row["source_id"], row["document_id"])


def _count(db, table: str, source: str, reason: str, n: int = 1):
    db.execute(f"INSERT INTO {table} VALUES(?,?,?) ON CONFLICT(source_id,reason) "
               "DO UPDATE SET n=n+excluded.n", (source, reason, n))


def _safe_json(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)


class CorpusV4Pipeline:
    def __init__(self, out: Path, config: dict, registry: dict, inputs: list[dict], *,
                 demo: bool = False, log=print):
        self.out = Path(out).resolve()
        self.config = config
        self.registry = registry
        self.inputs = inputs
        self.demo = demo
        self.log = log
        self.runtime = config["runtime"]
        self.seq_len = config["demo"]["seq_len"] if demo else config["targets"]["seq_len"]
        self.target_tokens = (config["demo"]["effective_train_tokens"] if demo
                              else config["targets"]["effective_train_tokens"])
        self.category_ratios = config["targets"]["category_token_ratios"]
        self.external_selection: dict | None = None
        self._done: set[str] = set()
        self.out.mkdir(parents=True, exist_ok=True)
        plan = {
            "schema_version": V4_SCHEMA_VERSION, "corpus_version": config["corpus_version"],
            "dataset_name": config["dataset_name"], "demo": demo,
            "config_sha256": identity(config), "implementation_sha256": implementation_sha256(),
            "registry": registry, "inputs": inputs,
            "policies": {
                "text_policy": config["text_policy"],
                "quality_policy_sha256": identity({"rules_version": QUALITY_RULES_VERSION,
                                                   "quality": config["quality"]}),
                "dedup_policy_sha256": identity(config["deduplication"]),
                "split_policy_sha256": identity(config["split"]),
                "tokenizer_config_sha256": identity(config["tokenizer"]),
            },
        }
        plan["plan_key"] = identity(plan)
        owner = self.out / ".v4" / "plan.json"
        if owner.exists():
            previous = json.loads(owner.read_text(encoding="utf-8"))
            if previous.get("plan_key") != plan["plan_key"]:
                raise CorpusError("corpus v4 inputs/policies changed; select a new --out (frozen inputs are never overwritten)")
            self.plan = previous
        else:
            if any(item.name != ".v4" for item in self.out.iterdir()) or (self.out / ".v4").exists():
                raise CorpusError("nonempty output has no corpus v4 plan; choose a new empty --out")
            write_json(owner, plan)
            self.plan = plan
        self.work = self.out / ".v4" / "work"
        self.stages = Stages(self.out, self.work, self.runtime, log, control=".v4")
        self.keys: dict[str, str] = {}
        self.stats: dict[str, dict] = {}

    # ------------------------------------------------------------------ keys

    def _key(self, name: str, extra=None) -> str:
        return identity([V4_SCHEMA_VERSION, self.plan["plan_key"], name, extra or {}])

    def _assert_writable(self):
        if (self.out / "FROZEN").exists():
            raise CorpusError("dataset is already frozen; create a new corpus version instead of modifying it")

    def _run(self, name: str, key: str, producer):
        self._assert_writable()
        record = self.stages.run(name, key, producer)
        self.keys[name] = record["key"]
        self.stats[name] = record["stats"]
        self._done.add(name)
        return record

    # ------------------------------------------------------------------ ingest

    def ingest(self):
        if "ingested" in self._done:
            return self.stats["ingested"]
        path = self.out / "ingested.sqlite"
        key = self._key("ingested")
        self._run("ingested", key, lambda: self._build_ingest(path))
        return self.stats["ingested"]

    def _build_ingest(self, path: Path):
        quality = self.config["quality"]
        sources = {source["source_id"]: source for source in self.registry["sources"]}
        documents = 0
        records_seen = 0
        with database_output(path, cache_mib=self.runtime["sqlite_cache_mib"]) as db:
            db.executescript("""
                CREATE TABLE docs(
                    document_id TEXT PRIMARY KEY, source_id TEXT NOT NULL, source_name TEXT NOT NULL,
                    source_uri TEXT, license TEXT NOT NULL, license_evidence TEXT,
                    license_status TEXT NOT NULL, provenance_status TEXT NOT NULL,
                    collection_version TEXT NOT NULL, declared_category TEXT NOT NULL,
                    declared_language TEXT, document_family_id TEXT NOT NULL, priority INTEGER NOT NULL,
                    file_sha256 TEXT NOT NULL, line_number INTEGER NOT NULL,
                    content_sha256 TEXT NOT NULL, raw_text TEXT NOT NULL, raw_bytes INTEGER NOT NULL,
                    metadata TEXT NOT NULL);
                CREATE TABLE counts(source_id TEXT, reason TEXT, n INTEGER,
                    PRIMARY KEY(source_id, reason));
            """)
            for item in self.inputs:
                source = sources[item["source_id"]]
                try:
                    records = stream_corpus_records(item["path"], max_record_bytes=quality["max_record_bytes"])
                except (OSError, ValueError) as exc:
                    raise CorpusError(f"cannot read {item['path']}: {exc}") from exc
                try:
                    for ordinal, record in enumerate(records, 1):
                        records_seen += 1
                        _count(db, "counts", source["source_id"], "records_seen")
                        if "_issue" in record:
                            _count(db, "counts", source["source_id"], record["_issue"])
                            continue
                        raw = record.get("text")
                        if not isinstance(raw, str) or not raw:
                            _count(db, "counts", source["source_id"], "missing_text")
                            continue
                        content_sha = hashlib.sha256(raw.encode("utf-8", errors="surrogatepass")).hexdigest()
                        line_number = record.get("_line", ordinal)
                        document_id = identity({
                            "source_id": source["source_id"], "file_sha256": item["sha256"],
                            "line": line_number, "content_sha256": content_sha})
                        parent = record.get("parent_id")
                        if parent is not None:
                            if not isinstance(parent, (str, int)) or len(str(parent)) > 512:
                                _count(db, "counts", source["source_id"], "invalid_parent_id")
                                continue
                            family = f"{source['source_id']}:parent:{parent}"
                        elif source["family_policy"] == "file" or Path(item["path"]).suffix.lower() == ".txt":
                            family = f"{source['source_id']}:file:{item['sha256']}"
                        else:
                            family = document_id
                        declared = source["category"]
                        if declared == "auto":
                            language = source["language"] if source["language"] != "auto" else detect_language(raw)
                            declared = infer_category(language, raw)
                        provenance = "DEMO_ONLY" if self.demo else source["provenance_status"]
                        metadata = {
                            "parent_id": parent, "author": record.get("author", source.get("author")),
                            "publication_date": record.get("publication_date"),
                            "acquisition_date": record.get("acquisition_date", source.get("acquisition_date")),
                            "original_format": record.get("original_format", source.get("original_format")),
                            "transformation_history": record.get("transformation_history",
                                                                 source.get("transformation_history", [])),
                            "source_notes": source.get("notes", ""),
                            "family_policy": source["family_policy"],
                        }
                        db.execute(
                            "INSERT INTO docs VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                            (document_id, source["source_id"], source["source_name"],
                             source["source_uri"], source["license"], source["license_evidence"],
                             source["license_status"], provenance, self.config["corpus_version"],
                             declared, source["language"], family, source["priority"],
                             item["sha256"], line_number, content_sha, raw,
                             len(raw.encode("utf-8", errors="surrogatepass")), _safe_json(metadata)))
                        documents += 1
                        _count(db, "counts", source["source_id"], "accepted_unique")
                        if records_seen % self.runtime["commit_every"] == 0:
                            db.commit()
                finally:
                    records.close()
                verify_file(Path(item["path"]), {key: item[key] for key in ("bytes", "sha256")})
            if not documents:
                raise CorpusError("no documents survived ingestion")
            stats = {"documents": documents, "records_seen": records_seen,
                     "by_source_reason": [dict(row) for row in db.execute(
                         "SELECT * FROM counts ORDER BY source_id, reason")]}
        return stats, [path]

    # ------------------------------------------------------------------ quality

    def quality(self):
        if "quality" in self._done:
            return self.stats["quality"]
        self.ingest()
        path = self.out / "reviewed.sqlite"
        queue = self.out / "review-queue.jsonl"
        report_path = self.out / "quality-report.json"
        key = self._key("quality", {"ingested": self.keys["ingested"],
                                   "quality": self.plan["policies"]["quality_policy_sha256"],
                                   "text_policy": self.config["text_policy"]})
        self._run("quality", key, lambda: self._build_quality(path, queue, report_path))
        return self.stats["quality"]

    def _build_quality(self, path: Path, queue: Path, report_path: Path):
        policy = self.config["quality"]
        policy_sha = self.plan["policies"]["quality_policy_sha256"]
        text_policy = self.config["text_policy"]
        with database_output(path, source=self.out / "ingested.sqlite",
                             cache_mib=self.runtime["sqlite_cache_mib"]) as db:
            db.executescript("""
                ALTER TABLE docs ADD COLUMN normalized_text TEXT;
                ALTER TABLE docs ADD COLUMN normalized_bytes INTEGER;
                ALTER TABLE docs ADD COLUMN normalized_sha256 TEXT;
                ALTER TABLE docs ADD COLUMN detected_language TEXT;
                ALTER TABLE docs ADD COLUMN category TEXT;
                ALTER TABLE docs ADD COLUMN quality_action TEXT;
                ALTER TABLE docs ADD COLUMN quality_flags TEXT;
                ALTER TABLE docs ADD COLUMN quality_rules_version TEXT;
                ALTER TABLE docs ADD COLUMN quality_policy_sha256 TEXT;
                CREATE TABLE quality_findings(document_id TEXT, source_id TEXT, rule_id TEXT,
                    severity TEXT, observed_value TEXT, threshold_value TEXT, action TEXT,
                    explanation TEXT);
                CREATE TABLE quality_counts(source_id TEXT, reason TEXT, action TEXT, n INTEGER,
                    PRIMARY KEY(source_id, reason, action));
            """)
            processed = 0
            cursor = db.execute("SELECT document_id FROM docs ORDER BY document_id")
            while batch := cursor.fetchmany(1000):
                for (document_id,) in batch:
                    row = db.execute("SELECT * FROM docs WHERE document_id=?", (document_id,)).fetchone()
                    self._review_one(db, row, policy, policy_sha, text_policy)
                    processed += 1
                    if processed % self.runtime["commit_every"] == 0:
                        db.commit()
            db.execute("CREATE INDEX docs_action ON docs(quality_action)")
            db.execute("CREATE INDEX docs_normalized ON docs(quality_action, normalized_sha256)")

            def write_queue(handle):
                for row in db.execute("SELECT document_id, source_id, category, detected_language, "
                                      "normalized_text, quality_flags FROM docs WHERE quality_action='REVIEW' "
                                      "ORDER BY source_id, document_id"):
                    handle.write(json_bytes({"document_id": row["document_id"], "source_id": row["source_id"],
                                             "category": row["category"], "language": row["detected_language"],
                                             "text": row["normalized_text"],
                                             "quality_flags": json.loads(row["quality_flags"])}))
            atomic_write(queue, write_queue)
            report = self._quality_report(db)
            write_json(report_path, report)
            stats = {key: report["totals"][key] for key in ("documents", "kept", "review", "rejected",
                                                            "raw_bytes", "normalized_bytes")}
            stats["by_source"] = report["by_source"]
            stats["rules_version"] = QUALITY_RULES_VERSION
        return stats, [path, queue, report_path]

    def _review_one(self, db, row, policy, policy_sha, text_policy):
        document_id = row["document_id"]
        raw = row["raw_text"]
        language = detect_language(raw)
        category = row["declared_category"]
        if category == "auto":
            category = infer_category(language, raw)
        normalized = normalise(raw, language, strip_html=policy["strip_html"])
        normalized = apply_text_policy(normalized, text_policy, category=language)
        normalized_language = detect_language(normalized)
        if normalized_language != language:
            language = normalized_language
            normalized = normalise(raw, language, strip_html=policy["strip_html"])
            normalized = apply_text_policy(normalized, text_policy, category=language)
        findings = evaluate_quality(normalized, language=language, category=category, policy=policy)
        action = aggregate_action(findings)
        flags = [finding.to_dict() for finding in findings]
        db.execute("""UPDATE docs SET normalized_text=?, normalized_bytes=?, normalized_sha256=?,
                      detected_language=?, category=?, quality_action=?, quality_flags=?,
                      quality_rules_version=?, quality_policy_sha256=? WHERE document_id=?""",
                   (normalized, len(normalized.encode("utf-8")),
                    content_hash(normalized, language), language, category, action,
                    _safe_json(flags), QUALITY_RULES_VERSION, policy_sha, document_id))
        for finding in findings:
            if finding.action == "KEEP":
                continue
            db.execute("INSERT INTO quality_findings VALUES(?,?,?,?,?,?,?,?)",
                       (document_id, row["source_id"], finding.rule_id, finding.severity,
                        _safe_json(finding.observed_value), _safe_json(finding.threshold),
                        finding.action, finding.explanation))
            db.execute(
                "INSERT INTO quality_counts VALUES(?,?,?,1) ON CONFLICT(source_id,reason,action) "
                "DO UPDATE SET n=n+1", (row["source_id"], finding.rule_id, finding.action))

    def _quality_report(self, db) -> dict:
        totals = dict(db.execute("SELECT COUNT(*) AS documents, SUM(quality_action='KEEP') AS kept, "
                                 "SUM(quality_action='REVIEW') AS review, SUM(quality_action='REJECT') AS rejected, "
                                 "COALESCE(SUM(raw_bytes),0) AS raw_bytes, "
                                 "COALESCE(SUM(normalized_bytes),0) AS normalized_bytes, "
                                 "COALESCE(SUM(LENGTH(raw_text)),0) AS raw_characters, "
                                 "COALESCE(SUM(LENGTH(normalized_text)),0) AS normalized_characters "
                                 "FROM docs").fetchone())
        names = ("documents", "kept", "review", "rejected", "raw_bytes", "normalized_bytes",
                 "raw_characters", "normalized_characters")
        totals = {name: int(totals[name] or 0) for name in names}
        totals["reject_ratio"] = totals["rejected"] / max(totals["documents"], 1)
        totals["review_ratio"] = totals["review"] / max(totals["documents"], 1)
        totals["document_ratio"] = 1.0
        totals["raw_character_ratio"] = 1.0
        totals["normalized_character_ratio"] = 1.0
        totals["tokens"] = None
        totals["token_ratio"] = None
        totals["token_basis"] = "not estimated before tokenizer selection; see tokenization-report.json"
        totals["language_distribution"] = {row[0]: row[1] for row in db.execute(
            "SELECT detected_language, COUNT(*) FROM docs GROUP BY detected_language ORDER BY detected_language")}
        totals["category_distribution"] = {row[0]: row[1] for row in db.execute(
            "SELECT category, COUNT(*) FROM docs GROUP BY category ORDER BY category")}
        reasons = [{"rule_id": row[0], "action": row[1], "findings": row[2]} for row in db.execute(
            "SELECT reason AS rule_id, action, SUM(n) AS findings FROM quality_counts GROUP BY reason, action "
            "ORDER BY SUM(n) DESC, reason LIMIT 20")]
        totals["top_reasons"] = reasons
        by_source = {}
        for (source_id,) in db.execute("SELECT DISTINCT source_id FROM docs ORDER BY source_id"):
            row = db.execute("SELECT COUNT(*), SUM(quality_action='KEEP'), SUM(quality_action='REVIEW'), "
                             "SUM(quality_action='REJECT'), COALESCE(SUM(raw_bytes),0), "
                             "COALESCE(SUM(normalized_bytes),0), COALESCE(SUM(LENGTH(raw_text)),0), "
                             "COALESCE(SUM(LENGTH(normalized_text)),0) FROM docs WHERE source_id=?",
                             (source_id,)).fetchone()
            data = {name: int(value or 0) for name, value in zip(names, row)}
            data["reject_ratio"] = data["rejected"] / max(data["documents"], 1)
            data["review_ratio"] = data["review"] / max(data["documents"], 1)
            data["document_ratio"] = data["documents"] / max(totals["documents"], 1)
            data["raw_character_ratio"] = data["raw_characters"] / max(totals["raw_characters"], 1)
            data["normalized_character_ratio"] = (
                data["normalized_characters"] / max(totals["normalized_characters"], 1))
            data["tokens"] = None
            data["token_ratio"] = None
            data["token_basis"] = "not estimated before tokenizer selection; see tokenization-report.json"
            data["language_distribution"] = {item[0]: item[1] for item in db.execute(
                "SELECT detected_language, COUNT(*) FROM docs WHERE source_id=? GROUP BY detected_language "
                "ORDER BY detected_language", (source_id,))}
            data["category_distribution"] = {item[0]: item[1] for item in db.execute(
                "SELECT category, COUNT(*) FROM docs WHERE source_id=? GROUP BY category ORDER BY category",
                (source_id,))}
            data["top_reasons"] = [{"rule_id": item[0], "action": item[1], "findings": item[2]}
                                   for item in db.execute(
                                       "SELECT reason AS rule_id, action, SUM(n) AS findings FROM quality_counts WHERE source_id=? "
                                       "GROUP BY reason, action ORDER BY SUM(n) DESC, reason LIMIT 10",
                                       (source_id,))]
            by_source[source_id] = data
        return {"schema_version": 1, "rules_version": QUALITY_RULES_VERSION,
                "policy_sha256": self.plan["policies"]["quality_policy_sha256"],
                "basis": ("documents and characters use separate denominators; real token ratios are "
                          "reported only after tokenization, never estimated here"),
                "totals": totals, "by_source": by_source,
                "limitations": "Rules are explainable heuristics; REVIEW is not an approval and absence of findings is not semantic truth."}

    # ------------------------------------------------------------------ dedup

    def deduplicate(self):
        if "dedup" in self._done:
            return self.stats["dedup"]
        self.quality()
        path = self.out / "deduplicated.sqlite"
        report = self.out / "dedup-report.json"
        key = self._key("dedup", {"quality": self.keys["quality"],
                                  "dedup": self.plan["policies"]["dedup_policy_sha256"]})
        self._run("dedup", key, lambda: self._build_dedup(path, report))
        return self.stats["dedup"]

    def _build_dedup(self, path: Path, report_path: Path):
        if not self.config["deduplication"]["exact"]:
            raise CorpusError("exact normalized-hash dedup is mandatory for corpus v4")
        policy = NearDupPolicy.from_config(self.config["deduplication"]["near"])
        with database_output(path, source=self.out / "reviewed.sqlite",
                             cache_mib=self.runtime["sqlite_cache_mib"]) as db:
            db.executescript("""
                ALTER TABLE docs ADD COLUMN dedup_cluster TEXT;
                ALTER TABLE docs ADD COLUMN dedup_kept INTEGER;
                ALTER TABLE docs ADD COLUMN dedup_duplicate_of TEXT;
                ALTER TABLE docs ADD COLUMN dedup_relation TEXT;
                ALTER TABLE docs ADD COLUMN dedup_similarity REAL;
                CREATE TABLE dedup_relations(kept_document_id TEXT, duplicate_document_id TEXT,
                    relation TEXT, similarity REAL, kept_source_id TEXT, duplicate_source_id TEXT);
            """)
            kept_rows = [dict(row) for row in db.execute(
                "SELECT document_id, normalized_sha256, source_id, priority FROM docs "
                "WHERE quality_action='KEEP' ORDER BY document_id")]
            dsu = _DSU()
            for row in kept_rows:
                dsu.add(row["document_id"])
            exact_groups = [row for row in db.execute(
                "SELECT normalized_sha256, COUNT(*) n FROM docs WHERE quality_action='KEEP' "
                "GROUP BY normalized_sha256 HAVING COUNT(*)>1 ORDER BY normalized_sha256")]
            exact_duplicates = 0
            exact_rep: dict[str, str] = {}
            for group in exact_groups:
                members = [dict(row) for row in db.execute(
                    "SELECT document_id, source_id, priority FROM docs WHERE quality_action='KEEP' "
                    "AND normalized_sha256=? ORDER BY document_id", (group["normalized_sha256"],))]
                representative = min(members, key=_rep_key)
                for member in members:
                    dsu.union(representative["document_id"], member["document_id"])
                    exact_rep[member["document_id"]] = representative["document_id"]
                    if member["document_id"] != representative["document_id"]:
                        exact_duplicates += 1
            for row in kept_rows:
                exact_rep.setdefault(row["document_id"], row["document_id"])
            exact_representatives = sorted({row for row in exact_rep.values()})
            near_stats = {"enabled": policy.method != "exact", "candidate_comparisons": 0,
                          "candidate_truncations": 0, "max_candidates_observed": 0,
                          "truncated_chunks": 0, "clusters": 0, "duplicates": 0}
            best_near: dict[tuple[str, str], float] = {}
            index = None
            if policy.method != "exact" and exact_representatives:
                index = MinHashLSHIndex(db, policy)
                for document_id in exact_representatives:
                    row = db.execute("SELECT normalized_text, detected_language FROM docs WHERE document_id=?",
                                     (document_id,)).fetchone()
                    index.add(document_id, row["normalized_text"], row["detected_language"])
                near_stats.update(index.annotated())
                cache: dict[str, tuple[str, str]] = {}
                def load(document_id):
                    if document_id not in cache:
                        if len(cache) >= 2048:
                            cache.clear()
                        row = db.execute("SELECT normalized_text, detected_language FROM docs WHERE document_id=?",
                                         (document_id,)).fetchone()
                        cache[document_id] = (row["normalized_text"], row["detected_language"])
                    return cache[document_id]
                for document_id in exact_representatives:
                    text, language = load(document_id)
                    for chunk_index, _, own_values, signature, _ in chunk_features(text, language, policy):
                        for candidate_id, candidate_chunk in index.candidates(document_id, chunk_index, signature):
                            near_stats["candidate_comparisons"] += 1
                            if dsu.find(candidate_id) == dsu.find(document_id):
                                continue
                            candidate_text, candidate_language = load(candidate_id)
                            candidate_chunk_text = chunk_text(candidate_text, candidate_chunk, policy)
                            if candidate_chunk_text is None:
                                continue
                            candidate_values, _ = shingles(candidate_chunk_text, candidate_language, policy)
                            similarity = exact_jaccard(own_values, candidate_values)
                            if similarity < policy.threshold:
                                continue
                            dsu.union(document_id, candidate_id)
                            pair = tuple(sorted((document_id, candidate_id)))
                            best_near[pair] = max(best_near.get(pair, 0.0), similarity)
                near_stats["candidate_truncations"] = index.stats["candidate_truncations"]
                near_stats["max_candidates_observed"] = index.stats["max_candidates_observed"]
                near_stats["truncated_chunks"] = index.stats["truncated_chunks"]
            clusters: dict[str, list[str]] = {}
            for row in kept_rows:
                representative = exact_rep[row["document_id"]]
                clusters.setdefault(dsu.find(representative), []).append(row["document_id"])
            cluster_rows = []
            doc_meta = {row["document_id"]: row for row in kept_rows}
            for root, members in clusters.items():
                members = sorted(members)
                representative = min(members, key=lambda item: _rep_key(
                    {"priority": doc_meta[item]["priority"], "source_id": doc_meta[item]["source_id"],
                     "document_id": item}))
                cluster_id = "cluster:" + representative
                cluster_rows.append((cluster_id, representative, members))
            cluster_size_histogram: dict[str, int] = {}
            for cluster_id, representative, members in cluster_rows:
                histogram_add(cluster_size_histogram, len(members))
                for member in members:
                    if member == representative:
                        relation, similarity = None, None
                    elif exact_rep[member] != member:
                        relation, similarity = "exact", 1.0
                    else:
                        relation = "near"
                        similarity = max((value for (a, b), value in best_near.items()
                                          if {a, b} == {member, representative}), default=None)
                    db.execute("UPDATE docs SET dedup_cluster=?, dedup_kept=?, dedup_duplicate_of=?, "
                               "dedup_relation=?, dedup_similarity=? WHERE document_id=?",
                               (cluster_id, int(member == representative),
                                None if member == representative else representative,
                                relation, similarity, member))
                    if relation is not None:
                        db.execute("INSERT INTO dedup_relations VALUES(?,?,?,?,?,?)",
                                   (representative, member, relation, similarity,
                                    doc_meta[representative]["source_id"], doc_meta[member]["source_id"]))
            db.execute("UPDATE docs SET dedup_kept=0, dedup_relation='review_held' WHERE quality_action='REVIEW'")
            db.execute("UPDATE docs SET dedup_kept=0, dedup_relation='rejected' WHERE quality_action='REJECT'")
            counts = dict(db.execute(
                "SELECT COUNT(*) AS total, SUM(dedup_kept=1) AS kept, "
                "SUM(dedup_relation='exact') AS exact, SUM(dedup_relation='near') AS near "
                "FROM docs WHERE quality_action='KEEP'").fetchone())
            chars = dict(db.execute(
                "SELECT COALESCE(SUM(LENGTH(normalized_text)),0) AS before, "
                "COALESCE(SUM(CASE WHEN dedup_kept=1 THEN LENGTH(normalized_text) ELSE 0 END),0) AS after "
                "FROM docs WHERE quality_action='KEEP'").fetchone())
            estimated = {"before": 0, "after": 0}
            chars_per_token = self.config["budget"]["chars_per_token"]
            for row in db.execute("SELECT LENGTH(normalized_text), detected_language, dedup_kept "
                                  "FROM docs WHERE quality_action='KEEP'"):
                tokens = max(1, math.ceil(row[0] / chars_per_token.get(row[1], 3.0)))
                estimated["before"] += tokens
                if row[2]:
                    estimated["after"] += tokens
            near_duplicates = int(counts["near"] or 0)
            near_clusters = db.execute(
                "SELECT COUNT(DISTINCT dedup_cluster) FROM docs WHERE dedup_relation='near'").fetchone()[0]
            near_stats.update({"clusters": int(near_clusters or 0),
                               "duplicates": near_duplicates,
                               "candidate_truncated": bool(near_stats["candidate_truncations"])})
            report = {
                "schema_version": 1, "policy": policy.to_dict(),
                "policy_sha256": self.plan["policies"]["dedup_policy_sha256"],
                "near_policy_sha256": policy.policy_sha256,
                "exact": {"enabled": bool(self.config["deduplication"]["exact"]),
                          "groups": len(exact_groups), "duplicates": exact_duplicates,
                          "method": "normalized text SHA256"},
                "near": near_stats,
                "clusters": {"total": len(cluster_rows), "size_histogram": cluster_size_histogram,
                             "deleted_documents": int(counts["total"] or 0) - int(counts["kept"] or 0)},
                "documents": {"kept": int(counts["kept"] or 0), "deleted": int(counts["total"] or 0) - int(counts["kept"] or 0)},
                "chars": {"before": int(chars["before"] or 0), "after": int(chars["after"] or 0)},
                "tokens_estimated": {"basis": "chars_per_token budget estimate before tokenizer selection",
                                     **estimated, "deleted": estimated["before"] - estimated["after"]},
                "relations": {"exact": exact_duplicates, "near": near_duplicates,
                              "review_held": db.execute("SELECT COUNT(*) FROM docs WHERE quality_action='REVIEW'").fetchone()[0],
                              "rejected": db.execute("SELECT COUNT(*) FROM docs WHERE quality_action='REJECT'").fetchone()[0]},
                "determinism": "documents ordered by id; representatives by (-priority, source_id, document_id); union roots canonicalised by member set",
                "limitations": "Near-duplicate similarity is exact Jaccard on shingles for retrieved candidates; LSH can miss candidates below the documented probability curve."}
            write_json(report_path, report)
        return {**report, "documents": int(report["documents"]["kept"])}, [path, report_path]

    # ------------------------------------------------------------------ split

    def split(self):
        if "split" in self._done:
            return self.stats["split"]
        self.deduplicate()
        key = self._key("split", {"dedup": self.keys["dedup"],
                                  "split": self.plan["policies"]["split_policy_sha256"],
                                  "seed": self.config["split"]["seed"]})
        self._run("split", key, lambda: self._build_split())
        return self.stats["split"]

    def _build_split(self):
        split_policy = self.config["split"]
        assignment_path = self.out / "split-assignment.jsonl"
        manifest_path = self.out / "split-manifest.json"
        with sqlite3.connect(self.out / "deduplicated.sqlite") as db:
            db.row_factory = sqlite3.Row
            rows = [dict(row) for row in db.execute(
                "SELECT document_id, document_family_id, dedup_cluster, category, detected_language, "
                "source_id, normalized_sha256, content_sha256, LENGTH(normalized_text) chars "
                "FROM docs WHERE dedup_kept=1 ORDER BY document_id")]
        if not rows:
            raise CorpusError("no kept documents to split")
        by_id = {row["document_id"]: row for row in rows}
        dsu = _DSU()
        for row in rows:
            dsu.add(row["document_id"])
        for column in ("dedup_cluster", "document_family_id"):
            groups: dict[str, list[str]] = {}
            for row in rows:
                groups.setdefault(row[column], []).append(row["document_id"])
            for group in groups.values():
                for member in group[1:]:
                    dsu.union(group[0], member)
        raw_groups: dict[str, list[str]] = {}
        for row in rows:
            raw_groups.setdefault(dsu.find(row["document_id"]), []).append(row["document_id"])
        groups = []
        for members in raw_groups.values():
            members = sorted(members)
            group_id = "group:" + hashlib.sha256("\n".join(members).encode("utf-8")).hexdigest()[:24]
            score = int(hashlib.sha256(f"{split_policy['seed']}:split:{group_id}".encode("utf-8")).hexdigest()[:16], 16) / 2 ** 64
            groups.append({"group_id": group_id, "members": members, "score": score,
                           "documents": len(members),
                           "characters": sum(by_id[member]["chars"] for member in members)})
        groups.sort(key=lambda group: (group["score"], group["group_id"]))
        total_documents = len(rows)
        ratios = split_policy["ratios"]
        if len(groups) < 3:
            raise CorpusError("at least three independent family/cluster groups are required for train/val/test")
        test_groups = max(1, round(len(groups) * ratios["test"])) if ratios["test"] > 0 else 0
        val_groups = max(1, round(len(groups) * ratios["val"])) if ratios["val"] > 0 else 0
        while test_groups + val_groups >= len(groups):
            if test_groups >= val_groups and test_groups > 1:
                test_groups -= 1
            elif val_groups > 1:
                val_groups -= 1
            else:
                raise CorpusError("split ratios leave no room for independent train/val/test groups")
        assigned: dict[str, str] = {}
        doc_group: dict[str, str] = {}
        for position, group in enumerate(groups):
            if position < test_groups:
                target = "test"
            elif position < test_groups + val_groups:
                target = "val"
            else:
                target = "train"
            for member in group["members"]:
                assigned[member] = target
                doc_group[member] = group["group_id"]
        group_splits = {group["group_id"]: next(
            split for split in SPLITS if assigned[group["members"][0]] == split) for group in groups}
        for group in groups:
            splits = {assigned[member] for member in group["members"]}
            if len(splits) != 1:
                raise CorpusError(f"family/cluster group crosses splits: {group['group_id']}")
        split_stats = {name: {"documents": 0, "characters": 0, "by_category": {}, "by_language": {},
                              "groups": 0} for name in SPLITS}
        for group in groups:
            split = group_splits[group["group_id"]]
            split_stats[split]["groups"] += 1
        handles = {}
        for name in SPLITS:
            handle = (self.out / f".{name}.text.jsonl.tmp").open("w", encoding="utf-8")
            handles[name] = handle
        try:
            with sqlite3.connect(self.out / "deduplicated.sqlite") as source:
                source.row_factory = sqlite3.Row
                for split in SPLITS:
                    for row in source.execute(
                            "SELECT * FROM docs WHERE dedup_kept=1 ORDER BY category, document_id"):
                        if assigned[row["document_id"]] != split:
                            continue
                        record = {
                            "document_id": row["document_id"], "source_id": row["source_id"],
                            "source_name": row["source_name"], "source_uri": row["source_uri"],
                            "license": row["license"], "license_evidence": row["license_evidence"],
                            "language": row["detected_language"], "category": row["category"],
                            "text": row["normalized_text"], "content_sha256": row["content_sha256"],
                            "normalized_sha256": row["normalized_sha256"],
                            "document_family_id": row["document_family_id"],
                            "collection_version": row["collection_version"],
                            "quality_flags": json.loads(row["quality_flags"]),
                            "provenance_status": row["provenance_status"],
                            "author": json.loads(row["metadata"]).get("author"),
                            "publication_date": json.loads(row["metadata"]).get("publication_date"),
                            "acquisition_date": json.loads(row["metadata"]).get("acquisition_date"),
                            "original_format": json.loads(row["metadata"]).get("original_format"),
                            "transformation_history": json.loads(row["metadata"]).get("transformation_history", []),
                            "split": split, "group_id": doc_group[row["document_id"]],
                            "dedup_cluster": row["dedup_cluster"],
                            "quality_action": row["quality_action"],
                        }
                        handles[split].write(json_bytes(record).decode())
                        stats = split_stats[split]
                        stats["documents"] += 1
                        stats["characters"] += len(row["normalized_text"])
                        for field, label in (("by_category", row["category"]),
                                             ("by_language", row["detected_language"])):
                            stats[field][label] = stats[field].get(label, 0) + 1
        finally:
            for handle in handles.values():
                handle.close()
        for name in SPLITS:
            os.replace(self.out / f".{name}.text.jsonl.tmp", self.out / f"{name}.text.jsonl")
        assignment_rows = sorted(
            ({"document_id": member, "group_id": doc_group[member],
              "split": assigned[member], "category": by_id[member]["category"],
              "language": by_id[member]["detected_language"], "source_id": by_id[member]["source_id"]}
             for member in assigned), key=lambda row: (row["split"], row["group_id"], row["document_id"]))
        def write_assignment(handle):
            for row in assignment_rows:
                handle.write(json_bytes(row))
        atomic_write(assignment_path, write_assignment)
        leakage = self._split_leakage(assigned, by_id)
        file_hashes = {f"{name}.text.jsonl": file_info(self.out / f"{name}.text.jsonl") for name in SPLITS}
        file_hashes["split-assignment.jsonl"] = file_info(assignment_path)
        manifest = {
            "schema_version": 1, "strategy": split_policy["strategy"],
            "seed": split_policy["seed"], "ratios": ratios,
            "group_count": len(groups), "document_count": total_documents,
            "splits": split_stats, "files": file_hashes,
            "assignment_sha256": file_hashes["split-assignment.jsonl"]["sha256"],
            "leakage": leakage,
            "known_limitations": "Group assignment is deterministic but not size-optimal; ratios are approximate at group granularity.",
        }
        if not split_stats["train"]["documents"]:
            raise CorpusError("split produced no train documents")
        if not split_stats["val"]["documents"] or not split_stats["test"]["documents"]:
            raise CorpusError("split produced an empty validation/test split; add independent groups")
        write_json(manifest_path, manifest)
        stats = {"documents": total_documents, "splits": split_stats, "groups": len(groups),
                 "leakage": leakage, "split_manifest_sha256": sha256(manifest_path)}
        return stats, [*(self.out / f"{name}.text.jsonl" for name in SPLITS), assignment_path, manifest_path]

    def _split_leakage(self, assigned: dict, by_id: dict) -> dict:
        exact = {}
        for name in SPLITS:
            hashes = {row["normalized_sha256"] for row in by_id.values() if assigned[row["document_id"]] == name}
            exact[name] = hashes
        overlaps = {}
        for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
            overlaps[f"{a}/{b}"] = len(exact[a] & exact[b])
        raw = {}
        with sqlite3.connect(self.out / "deduplicated.sqlite") as db:
            for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
                ids_a = [doc_id for doc_id, name in assigned.items() if name == a]
                ids_b = [doc_id for doc_id, name in assigned.items() if name == b]
                def grouped(ids):
                    if not ids:
                        return {}
                    placeholders = ",".join("?" for _ in ids)
                    return {row[0]: row[1] for row in db.execute(
                        f"SELECT content_sha256, COUNT(*) FROM docs WHERE document_id IN ({placeholders}) "
                        "GROUP BY content_sha256", ids)}
                raw[f"{a}/{b}"] = len(set(grouped(ids_a)) & set(grouped(ids_b)))
        near = {"status": "NOT_ASSESSED", "hits": 0}
        if self.config["split"]["verify_near_cross_split"]:
            near = self._cross_split_near(assigned)
        return {"exact_normalized": overlaps, "exact_raw": raw,
                "family_cluster_violations": 0,
                "near_duplicate": near,
                "scope": "group isolation is enforced; exact raw/normalized hashes are recomputed across splits"}

    def _cross_split_near(self, assigned: dict) -> dict:
        policy = NearDupPolicy.from_config(self.config["deduplication"]["near"])
        temporary = self.work / "cross-split-near.sqlite"
        temporary.unlink(missing_ok=True)
        conn = sqlite3.connect(temporary)
        conn.row_factory = sqlite3.Row
        hits = 0
        maximum = 0.0
        try:
            with sqlite3.connect(self.out / "deduplicated.sqlite") as source:
                source.row_factory = sqlite3.Row
                pairs = (("train", ("val", "test")), ("val", ("test",)))
                for pair_index, (left, right_names) in enumerate(pairs):
                    index = MinHashLSHIndex(conn, policy, prefix=f"cross{pair_index}")
                    for row in source.execute(
                            "SELECT document_id, normalized_text, detected_language FROM docs "
                            "WHERE dedup_kept=1 ORDER BY document_id"):
                        if assigned[row["document_id"]] == left:
                            index.add(row["document_id"], row["normalized_text"], row["detected_language"])
                    for row in source.execute(
                            "SELECT document_id, normalized_text, detected_language FROM docs "
                            "WHERE dedup_kept=1 ORDER BY document_id"):
                        if assigned[row["document_id"]] not in right_names:
                            continue
                        for chunk_index, _, own_values, signature, _ in chunk_features(
                                row["normalized_text"], row["detected_language"], policy):
                            for candidate_id, candidate_chunk in index.candidates(
                                    row["document_id"], chunk_index, signature):
                                candidate = source.execute(
                                    "SELECT normalized_text, detected_language FROM docs WHERE document_id=?",
                                    (candidate_id,)).fetchone()
                                candidate_text = chunk_text(candidate["normalized_text"], candidate_chunk, policy)
                                if candidate_text is None:
                                    continue
                                values, _ = shingles(candidate_text, candidate["detected_language"], policy)
                                similarity = exact_jaccard(own_values, values)
                                maximum = max(maximum, similarity)
                                if similarity >= policy.threshold:
                                    hits += 1
            return {"status": "CHECKED", "hits": hits, "max_similarity": round(maximum, 6),
                    "threshold": policy.threshold,
                    "method": "MinHash LSH candidates for train/val/test pairs; exact Jaccard confirmation"}
        finally:
            conn.close()
            temporary.unlink(missing_ok=True)

    # ------------------------------------------------------------------ tokenizers

    def train_tokenizers(self):
        if "tokenizers" in self._done:
            return self.stats["tokenizers"]
        self.split()
        if self.demo:
            candidates = self.config["demo"]["tokenizer_candidates"]
            output_root = self.out / "tokenizers"
        else:
            candidates = self.config["tokenizer"]["candidates"]
            output_root = (Path(self.config["tokenizer"]["output_root"])).resolve()
        from .tokenizer_v4 import (build_selection, guard_train_only, train_candidate)
        train_path = self.out / "train.text.jsonl"
        try:
            manifest = guard_train_only(train_path, self.out / "split-manifest.json")
        except ValueError as exc:
            raise CorpusError(str(exc)) from exc
        training = self.config["tokenizer"]["training"]
        selection = build_selection(train_path, category_weights=training["category_weights"],
                                    max_chars=training["max_chars"], seed=training["seed"])
        key = self._key("tokenizers", {"split": self.keys["split"],
                                       "candidates": candidates,
                                       "selection": {k: selection[k] for k in (
                                           "train_sha256", "seed", "max_chars", "category_weights")}})
        def producer():
            results = []
            for candidate in candidates:
                results.append(train_candidate(
                    selection, train_path, name=candidate["name"], vocab_size=candidate["vocab_size"],
                    out_root=output_root, text_policy=self.config["text_policy"],
                    split_manifest_sha256=sha256(self.out / "split-manifest.json"),
                    training_config=training, demo=self.demo))
            report = {"schema_version": 1, "demo_only": self.demo, "candidates": results,
                      "selection": {key: selection[key] for key in (
                          "seed", "max_chars", "category_weights", "category_budgets",
                          "selected_documents", "selected_chars", "selected_by_category")},
                      "train_only_guard": "split-manifest hash verified; val/test file hashes rechecked",
                      "training_split_sha256": sha256(train_path),
                      "split_manifest_sha256": sha256(self.out / "split-manifest.json"),
                      "limitations": "Fitting on a bounded 30M-character sample unless configured otherwise; never on val/test."}
            report_path = self.out / "tokenizer-candidates.json"
            write_json(report_path, report)
            files = []
            for result in results:
                directory = Path(result["directory"])
                files.extend(directory / name for name in result["files"])
                files.append(directory / "candidate-manifest.json")
            files.append(report_path)
            return {"documents": selection["selected_documents"], "candidates": results, "report": report}, files
        self._run("tokenizers", key, producer)
        return self.stats["tokenizers"]

    def evaluate_tokenizers(self, *, baseline: str | Path | None = None):
        if "tokenizer_eval" in self._done:
            return self.stats["tokenizer_eval"]
        self.train_tokenizers()
        from .tokenizer_v4 import compare_candidates, load_eval_texts, parameter_budget
        candidates = self.stats["tokenizers"]["candidates"]
        if self.demo:
            eval_asset = (Path.cwd() / self.config["demo"]["eval_asset"]).resolve()
        else:
            eval_asset = self._resolve_path(self.config["tokenizer"]["eval_asset"])
        try:
            texts = load_eval_texts(eval_asset)
        except (OSError, ValueError) as exc:
            raise CorpusError(f"invalid fixed tokenizer evaluation text: {exc}") from exc
        baseline = baseline or self.config["tokenizer"].get("baseline")
        baseline_path = self._resolve_path(baseline) if baseline else None
        key = self._key("tokenizer_eval", {"tokenizers": self.keys["tokenizers"],
                                           "eval": file_info(eval_asset),
                                           "baseline": str(baseline_path) if baseline_path else None})
        def producer():
            comparison = compare_candidates(
                candidates, texts, baseline_path=baseline_path, text_policy=self.config["text_policy"],
                evaluation_config=self.config["tokenizer"]["evaluation"])
            comparison["parameter_budget"] = parameter_budget(
                [candidate["vocab_size"] for candidate in candidates],
                seq_len=self.seq_len,
                hardware=self.config["hardware"])
            comparison["demo_only"] = self.demo
            comparison["selection"] = self._select_candidate(candidates, comparison)
            report_path = self.out / "tokenizer-evaluation.json"
            write_json(report_path, comparison)
            return {**comparison, "documents": len(texts)}, [report_path]
        self._run("tokenizer_eval", key, producer)
        return self.stats["tokenizer_eval"]

    def _resolve_path(self, value) -> Path:
        path = Path(value)
        return path if path.is_absolute() else (Path.cwd() / path).resolve()

    def _select_candidate(self, candidates: list[dict], comparison: dict) -> dict:
        if self.demo:
            selected = candidates[0]
            return {"name": selected["name"], "path": selected["path"],
                    "fingerprint": selected["fingerprint"], "vocab_size": selected["vocab_size"],
                    "mode": "AUTO_DEMO_ONLY", "reason": "demo uses the smaller candidate; not a formal selection"}
        configured = self.config["tokenizer"]["selected"]
        if configured:
            match = next((candidate for candidate in candidates if candidate["name"] == configured), None)
            if match is None:
                raise CorpusError("selected tokenizer candidate was not trained")
            return {"name": match["name"], "path": match["path"],
                    "fingerprint": match["fingerprint"], "vocab_size": match["vocab_size"],
                    "mode": "EXPLICIT_CONFIG", "reason": "config tokenizer.selected"}
        raise CorpusError("formal v4 build requires tokenizer.selected (or pass --tokenizer); "
                          "the comparison is evidence for a human decision, not an automatic choice")

    # ------------------------------------------------------------------ tokenize

    def tokenize(self, tokenizer_path: str | Path | None = None, *, selected: dict | None = None):
        if "tokenized" in self._done:
            return self.stats["tokenized"]
        self.split()
        if tokenizer_path is None:
            comparison = self.evaluate_tokenizers()
            if selected is None:
                selected = comparison.get("selection")
            if not selected:
                raise CorpusError("no tokenizer selected")
            tokenizer_path = selected["path"]
        tokenizer_path = self._resolve_path(tokenizer_path)
        tokenizer = MiniLLMTokenizer.load(tokenizer_path)
        tokenizer.text_policy = validate_policy(self.config["text_policy"])
        if not 264 <= tokenizer.vocab_size <= 65536:
            raise CorpusError("tokenizer vocabulary must fit the uint16 corpus format")
        if selected is None:
            selected = {"name": tokenizer_path.parent.name, "path": str(tokenizer_path),
                        "fingerprint": tokenizer.fingerprint, "vocab_size": tokenizer.vocab_size,
                        "mode": "EXTERNAL_PATH", "reason": "tokenizer supplied explicitly"}
        self.external_selection = selected
        target = self.out / "tokenizer.json"
        def copy_tokenizer():
            with tokenizer_path.open("rb") as source:
                atomic_write(target, lambda handle: shutil.copyfileobj(source, handle, length=1048576))
            return {"fingerprint": tokenizer.fingerprint, "vocab_size": tokenizer.vocab_size,
                    "source_path": str(tokenizer_path)}, [target]
        token_key = self._key("tokenizer-file", {"fingerprint": tokenizer.fingerprint,
                                                 "split": self.keys["split"]})
        token_record = self._run("tokenizer-file", token_key, copy_tokenizer)
        seq_len = self.seq_len
        key = self._key("tokenized", {"split": self.keys["split"],
                                      "tokenizer": tokenizer.fingerprint,
                                      "seq_len": seq_len,
                                      "text_policy": self.config["text_policy"]})
        def producer():
            from .corpus import read_records
            results = {}
            index_paths = []
            for name in SPLITS:
                text_path = self.out / f"{name}.text.jsonl"
                bin_path = self.out / f"{name}.bin"
                index_path = self.out / f"{name}.token-index.jsonl"
                records = read_records(text_path)
                try:
                    stats = tokenize_split(
                        records, tokenizer, bin_path,
                        buffer_bytes=self.runtime["buffer_bytes"],
                        category_key="category", source_key="source_id",
                        text_category_key="language", index_path=index_path,
                        index_fields=("document_id", "category", "language", "source_id"))
                finally:
                    records.close()
                results[name] = stats
                index_paths.append(index_path)
            packing = supervision_by_category(
                ((row["category"], row["tokens"]) for row in read_token_index([self.out / "train.token-index.jsonl"])["records"]),
                seq_len)
            source_totals = {}
            for split in SPLITS:
                for source, data in results[split]["by_source"].items():
                    totals = source_totals.setdefault(source, {"documents": 0, "characters": 0, "tokens": 0})
                    totals["documents"] += data["documents"]
                    totals["characters"] += data["chars"]
                    totals["tokens"] += data["tokens"]
            total_characters = sum(item["characters"] for item in source_totals.values())
            total_tokens = sum(item["tokens"] for item in source_totals.values())
            total_documents = sum(item["documents"] for item in source_totals.values())
            for totals in source_totals.values():
                totals["document_ratio"] = totals["documents"] / max(total_documents, 1)
                totals["character_ratio"] = totals["characters"] / max(total_characters, 1)
                totals["token_ratio"] = totals["tokens"] / max(total_tokens, 1)
            deleted = self._measure_deleted_tokens(tokenizer)
            kept_tokens = sum(results[name]["tokens"] for name in SPLITS)
            dedup_delta = {"basis": "exact selected-tokenizer encoding; EOS included",
                           "deleted_documents": deleted["documents"],
                           "deleted_tokens": deleted["tokens"],
                           "kept_tokens": kept_tokens,
                           "before_tokens": kept_tokens + deleted["tokens"]}
            report = {
                "schema_version": 1, "documents": total_documents,
                "tokenizer": {"fingerprint": tokenizer.fingerprint, "vocab_size": tokenizer.vocab_size,
                              "source_path": str(tokenizer_path), "file": "tokenizer.json"},
                "text_policy": self.config["text_policy"], "seq_len": seq_len,
                "splits": results, "packing": packing,
                "effective_train_tokens": packing["effective_tokens"],
                "dedup_token_delta": dedup_delta,
                "category_token_distribution": {
                    "raw": results["train"]["by_category"],
                    "effective": packing["by_category"],
                    "basis": "exact supervised positions in the packed train stream"},
                "source_distribution": {
                    "basis": "documents, Unicode characters and exact token counts; ratios are separate denominators",
                    "sources": source_totals,
                    "totals": {"documents": total_documents, "characters": total_characters,
                               "tokens": total_tokens}},
                "files": {name: file_info(self.out / f"{name}.bin") for name in SPLITS},
            }
            report_path = self.out / "tokenization-report.json"
            write_json(report_path, report)
            files = [*(self.out / f"{name}.bin" for name in SPLITS),
                     *(self.out / f"{name}.token-index.jsonl" for name in SPLITS), report_path]
            return report, files
        self._run("tokenized", key, producer)
        return self.stats["tokenized"]

    def _measure_deleted_tokens(self, tokenizer: MiniLLMTokenizer) -> dict:
        tokens, documents = 0, 0
        with sqlite3.connect(self.out / "deduplicated.sqlite") as db:
            for row in db.execute(
                    "SELECT normalized_text, detected_language FROM docs WHERE dedup_kept=0 "
                    "AND dedup_relation IN ('exact','near') ORDER BY document_id"):
                tokens += len(tokenizer.encode_document(row[0], category=row[1]))
                documents += 1
        return {"documents": documents, "tokens": tokens}

    # ------------------------------------------------------------------ reports

    def contamination(self):
        if "contamination" in self._done:
            return self.stats["contamination"]
        self.split()
        from .corpus import read_records
        records = read_records(self.out / "train.text.jsonl")
        try:
            report = contamination_audit(
                self._contamination_assets(),
                ({"document_id": row["document_id"], "text": row["text"],
                  "language": row["language"], "category": row["category"],
                  "document_family_id": row["document_family_id"]} for row in records),
                decisions=self._contamination_decisions(),
                min_prompt_chars=self.config["contamination"]["min_prompt_chars"],
                long_span_containment=self.config["contamination"]["long_span_containment"],
                max_hits=self.config["contamination"]["max_hits"])
        finally:
            records.close()
        report["demo_only"] = self.demo
        write_json(self.out / "contamination-report.json", report)
        self.stats["contamination"] = report
        self._done.add("contamination")
        return report

    def _contamination_assets(self) -> list[dict]:
        assets = []
        for asset in self.config["contamination"]["assets"]:
            path = self._resolve_path(asset["path"])
            assets.append({**asset, "path": str(path)})
        return assets

    def _contamination_decisions(self) -> Path | None:
        value = self.config["contamination"].get("decisions")
        return self._resolve_path(value) if value else None

    def budget(self):
        if "budget" in self._done:
            return self.stats["budget"]
        self.tokenize()
        tokenization = self.stats["tokenized"]
        packing = tokenization["packing"]
        available = {
            category: {"documents": packing["documents_by_category"].get(category, 0),
                       "effective_tokens": packing["by_category"].get(category, 0)}
            for category in V4_CATEGORIES}
        split_summary = {name: {"documents": tokenization["splits"][name]["documents"],
                                "raw_tokens": tokenization["splits"][name]["tokens"]}
                         for name in SPLITS}
        target = self.target_tokens
        ratios = self.category_ratios
        seq_len = self.seq_len
        license_status = self._license_status()
        contamination_status = (self.stats.get("contamination") or {}).get("status")
        if contamination_status is None and (self.out / "contamination-report.json").is_file():
            contamination_status = json.loads((self.out / "contamination-report.json").read_text(
                encoding="utf-8")).get("status")
        contamination_status = contamination_status or "NOT_ASSESSED"
        report = plan_budget(available, target, ratios, seq_len=seq_len,
                             tokenizer_fingerprint=tokenization["tokenizer"]["fingerprint"],
                             split_summary=split_summary,
                             license_review_status=license_status,
                             contamination_status=contamination_status,
                             allow_repetition=self.config["budget"]["allow_repetition"])
        report["demo_only"] = self.demo
        report["category_token_distribution_raw"] = tokenization["category_token_distribution"]["raw"]
        write_json(self.out / "budget-report.json", report)
        self.stats["budget"] = report
        self._done.add("budget")
        return report

    # ------------------------------------------------------------------ manifest

    def _license_status(self) -> str:
        included = self.registry["sources"]
        if not included:
            return "NOT_ASSESSED"
        if self.demo:
            return "DEMO_ONLY"
        if all(source["license_status"] == "APPROVED" and source["license_evidence"].strip()
               and source["license_reviewer"].strip() and source["license_reviewed_at"]
               for source in included):
            return "APPROVED"
        return "REVIEW_REQUIRED"

    def _published_files(self) -> dict[str, dict]:
        names = [f"{name}.bin" for name in SPLITS] + [f"{name}.text.jsonl" for name in SPLITS] + [
            f"{name}.token-index.jsonl" for name in SPLITS] + [
            "tokenizer.json", "tokenization-report.json", "tokenizer-evaluation.json",
            "tokenizer-candidates.json", "quality-report.json", "dedup-report.json",
            "split-manifest.json", "split-assignment.jsonl", "contamination-report.json",
            "budget-report.json"]
        files = {}
        for name in names:
            path = self.out / name
            if path.is_file():
                files[name] = file_info(path)
        return files

    def _stage_hashes(self) -> dict:
        result = {}
        for marker in sorted((self.out / ".v4").glob("*.stage.json")):
            data = json.loads(marker.read_text(encoding="utf-8"))
            result[marker.name[:-len(".stage.json")]] = {
                "key": data.get("key"),
                "files": {Path(path).name: info for path, info in data.get("files", {}).items()},
            }
        return result

    def _manifest_identity(self, files: dict) -> dict:
        tokenization = self.stats["tokenized"]
        budget = self.stats["budget"]
        contamination = self.stats.get("contamination") or {}
        return {
            "stage_hashes": self._stage_hashes(),
            "v4": {
                "corpus_version": self.config["corpus_version"],
                "source_manifest_sha256": self.registry["sha256"],
                "implementation_sha256": self.plan["implementation_sha256"],
                "cleaning_policy_sha256": self.plan["policies"]["quality_policy_sha256"],
                "dedup_policy_sha256": self.plan["policies"]["dedup_policy_sha256"],
                "split_manifest_sha256": sha256(self.out / "split-manifest.json"),
                "tokenizer_fingerprint": tokenization["tokenizer"]["fingerprint"],
                "text_policy": self.config["text_policy"],
                "effective_train_tokens": tokenization["effective_train_tokens"],
                "category_token_distribution": tokenization["category_token_distribution"],
                "budget_status": budget["status"],
                "contamination_audit_status": contamination.get("status", "NOT_ASSESSED"),
                "split_fractions": self.config["split"]["ratios"],
                "demo_only": self.demo,
                "stage_hashes": self._stage_hashes(),
            },
            "files": files,
        }

    def _manifest_files(self) -> dict[str, dict]:
        files = {}
        required = [f"{name}.bin" for name in SPLITS] + [f"{name}.text.jsonl" for name in SPLITS] + [
            f"{name}.token-index.jsonl" for name in SPLITS] + [
            "tokenizer.json", "tokenization-report.json", "quality-report.json",
            "dedup-report.json", "split-manifest.json", "split-assignment.jsonl",
            "contamination-report.json", "budget-report.json"]
        for name in required:
            path = self.out / name
            if not path.is_file():
                raise CorpusError(f"missing published artifact: {name}")
            files[name] = file_info(path)
        for name in ("tokenizer-evaluation.json", "tokenizer-candidates.json"):
            if (self.out / name).is_file():
                files[name] = file_info(self.out / name)
        return files

    def build_manifest(self, *, freeze_status: str = "NOT_ASSESSED",
                       dataset_status: str | None = None) -> dict:
        if "tokenized" not in self._done:
            self.tokenize()
        self.contamination()
        self.budget()
        files = self._manifest_files()
        identity_payload = self._manifest_identity(files)
        tokenization = self.stats["tokenized"]
        contamination = self.stats["contamination"]
        budget = self.stats["budget"]
        dataset_status = dataset_status or ("DEMO_ONLY" if self.demo else "FORMAL")
        binaries = {name: files[f"{name}.bin"]["sha256"] for name in SPLITS}
        manifest = {
            "schema_version": 1,
            "dataset_name": self.config["dataset_name"],
            "corpus_version": self.config["corpus_version"],
            "dataset_status": dataset_status,
            "content_id": identity(identity_payload),
            "identity": identity_payload,
            "files": files,
            "source_manifest_sha256": self.registry["sha256"],
            "implementation_sha256": self.plan["implementation_sha256"],
            "cleaning_policy_sha256": self.plan["policies"]["quality_policy_sha256"],
            "dedup_policy_sha256": self.plan["policies"]["dedup_policy_sha256"],
            "split_manifest_sha256": identity_payload["v4"]["split_manifest_sha256"],
            "tokenizer_fingerprint": tokenization["tokenizer"]["fingerprint"],
            "text_policy": self.config["text_policy"],
            "train_bin_sha256": binaries["train"], "val_bin_sha256": binaries["val"],
            "test_bin_sha256": binaries["test"],
            "effective_train_tokens": tokenization["effective_train_tokens"],
            "category_token_distribution": tokenization["category_token_distribution"],
            "license_review_status": self._license_status(),
            "contamination_audit_status": contamination["status"],
            "freeze_status": freeze_status,
            "budget_status": budget["status"],
            "split_fractions": self.config["split"]["ratios"],
            "tokenizer": {"fingerprint": tokenization["tokenizer"]["fingerprint"],
                          "vocab_size": tokenization["tokenizer"]["vocab_size"],
                          "file": "tokenizer.json"},
            "content_config": {"text_policy": self.config["text_policy"],
                               "corpus_version": self.config["corpus_version"]},
            "license_sources": self.registry,
            "contamination": {"status": contamination["status"],
                              "report": "contamination-report.json",
                              "high_risk_hits": contamination["stats"]["high_risk_hits"],
                              "unreviewed_high_risk": contamination.get("decisions", {}).get(
                                  "unreviewed_high_risk")},
            "budget": {"status": budget["status"],
                       "target_effective_train_tokens": budget["target_effective_train_tokens"],
                       "total_gap_effective_tokens": budget["total_gap_effective_tokens"]},
            "dataset_v4_stages": list(STAGES),
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }
        return manifest

    # ------------------------------------------------------------------ audit

    def audit(self, *, retokenize: str | None = None) -> dict:
        self._assert_writable()
        if "tokenized" not in self._done:
            self.tokenize()
        self.contamination()
        self.budget()
        retokenize = retokenize or self.config["audit"]["retokenize"]
        checks = {}
        blockers = []

        def check(name, status, detail, *, blocking=True):
            checks[name] = {"status": status, "detail": detail}
            if status == "FAIL" and blocking:
                blockers.append(name)
            if status == "NOT_ASSESSED" and blocking:
                blockers.append(name)

        license_status = self._license_status()
        if self.demo:
            check("license_review", "NOT_ASSESSED", {"status": license_status,
                                                     "reason": "DEMO_ONLY never asserts a formal licence review"})
        elif license_status == "APPROVED":
            check("license_review", "PASS", {"included_sources": len(self.registry["sources"])})
        else:
            check("license_review", "FAIL", {"status": license_status,
                                             "excluded": self.registry["excluded"]})

        quality_report = json.loads((self.out / "quality-report.json").read_text(encoding="utf-8"))
        quality_ok = (quality_report["rules_version"] == QUALITY_RULES_VERSION
                      and quality_report["policy_sha256"] == self.plan["policies"]["quality_policy_sha256"])
        check("cleaning_policy", "PASS" if quality_ok else "FAIL",
              {"rules_version": quality_report["rules_version"],
               "kept": quality_report["totals"]["kept"],
               "review": quality_report["totals"]["review"],
               "rejected": quality_report["totals"]["rejected"],
               "review_queue": "review-queue.jsonl"})

        dedup_report = json.loads((self.out / "dedup-report.json").read_text(encoding="utf-8"))
        dedup_ok = (dedup_report["policy_sha256"] == self.plan["policies"]["dedup_policy_sha256"]
                    and not dedup_report["near"].get("candidate_truncated"))
        check("dedup_policy", "PASS" if dedup_ok else "FAIL",
              {"policy_sha256": dedup_report["policy_sha256"],
               "exact_duplicates": dedup_report["exact"]["duplicates"],
               "near_duplicates": dedup_report["near"]["duplicates"],
               "candidate_truncations": dedup_report["near"].get("candidate_truncations", 0)})

        split_manifest = json.loads((self.out / "split-manifest.json").read_text(encoding="utf-8"))
        split_files_ok = all((self.out / name).is_file() and file_info(self.out / name) == info
                             for name, info in split_manifest["files"].items())
        leakage = split_manifest["leakage"]
        blocked = any(leakage["exact_normalized"].values()) or any(leakage["exact_raw"].values()) \
            or bool(leakage["family_cluster_violations"]) \
            or (leakage["near_duplicate"].get("hits", 0) > 0)
        check("split_isolation", "PASS" if split_files_ok and not blocked else "FAIL",
              {"files_verified": split_files_ok, "leakage": leakage})

        selection = self.stats.get("tokenizer_eval", {}).get("selection") or self.external_selection
        tokenization = self.stats["tokenized"]
        candidate_ok = bool(selection) and selection.get("fingerprint") == tokenization["tokenizer"]["fingerprint"]
        tokenizer_file = self.out / "tokenizer.json"
        loaded = MiniLLMTokenizer.load(tokenizer_file)
        tokenizer_ok = (candidate_ok and loaded.fingerprint == tokenization["tokenizer"]["fingerprint"]
                        and loaded.text_policy == self.config["text_policy"])
        check("tokenizer_selection", "PASS" if tokenizer_ok else "FAIL",
              {"selection": selection, "tokenizer_fingerprint": loaded.fingerprint,
               "vocab_size": loaded.vocab_size, "text_policy": loaded.text_policy})

        encoding = self._verify_encoding(loaded, retokenize)
        check("bin_reproducibility", encoding["status"], encoding)

        budget = self.stats["budget"]
        budget_ok = (budget["status"] == "READY_FOR_TRAINING"
                     and budget["target_effective_train_tokens"] == self.target_tokens
                     and budget["total_gap_effective_tokens"] == 0
                     and tokenization["effective_train_tokens"] > 0)
        check("token_budget", "PASS" if budget_ok else "FAIL",
              {"status": budget["status"], "target": budget["target_effective_train_tokens"],
               "available": budget["total_available_effective_tokens"],
               "gap": budget["total_gap_effective_tokens"],
               "effective_train_tokens": tokenization["effective_train_tokens"]})

        contamination = self.stats["contamination"]
        contamination_ok = contamination["status"] in ("CHECKED_NO_HITS", "REVIEWED")
        check("contamination_audit", "PASS" if contamination_ok else "FAIL",
              {"status": contamination["status"],
               "high_risk_hits": contamination["stats"]["high_risk_hits"],
               "allowed_knowledge_hits": contamination["stats"]["allowed_knowledge_hits"],
               "unreviewed_high_risk": contamination.get("decisions", {}).get("unreviewed_high_risk")})

        try:
            manifest = self.build_manifest(freeze_status="NOT_ASSESSED")
            files_ok = all((self.out / name).is_file() and file_info(self.out / name) == info
                           for name, info in manifest["files"].items())
            required = ("corpus_version", "source_manifest_sha256", "cleaning_policy_sha256",
                        "dedup_policy_sha256", "split_manifest_sha256", "tokenizer_fingerprint",
                        "text_policy", "train_bin_sha256", "val_bin_sha256", "test_bin_sha256",
                        "effective_train_tokens", "category_token_distribution",
                        "license_review_status", "contamination_audit_status", "freeze_status")
            complete = all(field in manifest for field in required) and files_ok
            check("manifest_completeness", "PASS" if complete else "FAIL",
                  {"content_id": manifest["content_id"], "files": len(manifest["files"]),
                   "missing_fields": [field for field in required if field not in manifest]})
        except (OSError, ValueError, KeyError, CorpusError) as exc:
            check("manifest_completeness", "FAIL", {"error": str(exc)})

        if self.demo:
            status = "NOT_READY"
            blockers.append("DEMO_ONLY")
        elif any(value["status"] == "FAIL" for value in checks.values()):
            status = "NOT_READY"
        elif any(value["status"] == "NOT_ASSESSED" for value in checks.values()):
            status = "NOT_ASSESSED"
        else:
            status = "READY"
        report = {"schema_version": 1, "status": status, "checks": checks,
                  "blockers": sorted(set(blockers)), "demo_only": self.demo,
                  "limitations": ("A PASS is evidence for the recorded policy and hashes, not a semantic "
                                  "guarantee; licence review, contamination dispositions and world-knowledge "
                                  "quality remain human responsibilities.")}
        write_json(self.out / "audit-report.json", report)
        self.stats["audit"] = report
        self._done.add("audit")
        return report

    def _verify_encoding(self, tokenizer: MiniLLMTokenizer, mode: str) -> dict:
        if mode == "none":
            return {"status": "NOT_ASSESSED", "detail": "retokenization disabled by config"}
        from .corpus import read_records
        import numpy as np
        verified = 0
        sample = self.config["audit"]["sample_documents"]
        for name in SPLITS:
            text_path, bin_path = self.out / f"{name}.text.jsonl", self.out / f"{name}.bin"
            if not bin_path.is_file() or not text_path.is_file():
                return {"status": "FAIL", "detail": f"missing split artifacts: {name}"}
            in_split = 0
            with bin_path.open("rb") as handle:
                for record in read_records(text_path):
                    if mode == "sample" and in_split >= sample:
                        break
                    ids = tokenizer.encode_document(record["text"], category=record["language"])
                    data = np.asarray(ids, dtype="<u2").tobytes()
                    if handle.read(len(data)) != data:
                        return {"status": "FAIL", "detail": f"{name}: retokenization differs",
                                "document_id": record["document_id"]}
                    in_split += 1
                    verified += 1
                if mode == "full" and handle.read(1):
                    return {"status": "FAIL", "detail": f"{name}: trailing tokens after last document"}
        return {"status": "PASS", "mode": mode, "documents_verified": verified,
                "scope": "all documents" if mode == "full" else f"first {sample} documents per split"}

    # ------------------------------------------------------------------ freeze

    def finalize(self, *, audit: dict | None = None) -> dict:
        self._assert_writable()
        audit = audit or self.audit()
        if self.demo:
            freeze_status, dataset_status = "NOT_READY", "DEMO_ONLY"
        elif audit["status"] == "READY":
            freeze_status, dataset_status = "READY", "FORMAL"
        elif audit["status"] == "NOT_ASSESSED":
            freeze_status, dataset_status = "NOT_ASSESSED", "FORMAL"
        else:
            freeze_status, dataset_status = "NOT_READY", "FORMAL"
        marker = self.out / "FROZEN"
        manifest = self.build_manifest(freeze_status=freeze_status, dataset_status=dataset_status)
        write_json(self.out / "dataset-manifest.json", manifest)
        if self.demo:
            atomic_write(self.out / "DEMO_ONLY", lambda handle: handle.write(json_bytes({
                "dataset_status": "DEMO_ONLY", "freeze_status": "NOT_READY",
                "reason": "synthetic fixtures exercise the pipeline but are not a trainable corpus"})))
        if freeze_status == "READY":
            freeze = {"schema_version": 1, "content_id": manifest["content_id"],
                      "freeze_status": "READY", "dataset_status": "FORMAL",
                      "manifest_sha256": sha256(self.out / "dataset-manifest.json"),
                      "corpus_version": self.config["corpus_version"],
                      "effective_train_tokens": manifest["effective_train_tokens"]}
            write_json(self.out / "freeze.json", freeze)
            atomic_write(marker, lambda handle: handle.write(
                json_bytes({"content_id": manifest["content_id"]})))
        self.log(f"[v4] freeze_status={freeze_status} dataset_status={dataset_status} "
                 f"content_id={manifest['content_id'][:16]}…")
        return manifest


def verify_frozen(directory: str | Path) -> dict:
    """Refuse anything but a formally READY, hash-verified, frozen dataset."""
    directory = Path(directory)
    manifest = json.loads((directory / "dataset-manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1 or manifest.get("freeze_status") != "READY" \
            or manifest.get("dataset_status") != "FORMAL":
        raise CorpusError("dataset is not formally frozen (freeze_status/dataset_status)")
    if identity(manifest["identity"]) != manifest["content_id"] or manifest["identity"]["files"] != manifest["files"]:
        raise CorpusError("dataset identity/files mismatch")
    for name, info in manifest["files"].items():
        if Path(name).name != name:
            raise CorpusError("manifest file names must be local")
        verify_file(directory / name, info)
    # Stage markers bind intermediate artifacts (including tokenizer candidate
    # metadata outside the dataset directory); verify them too, not just final bins.
    recomputed = {}
    for marker in sorted((directory / ".v4").glob("*.stage.json")):
        data = json.loads(marker.read_text(encoding="utf-8"))
        stage = marker.name[:-len(".stage.json")]
        files = {}
        for filename, info in data.get("files", {}).items():
            path = Path(filename)
            verify_file(path, info)
            files[path.name] = info
        recomputed[stage] = {"key": data.get("key"), "files": files}
    if recomputed != manifest["identity"].get("stage_hashes"):
        raise CorpusError("stage hashes do not match the frozen manifest")
    if not (directory / "FROZEN").is_file():
        raise CorpusError("missing FROZEN marker")
    if manifest["license_review_status"] != "APPROVED":
        raise CorpusError("frozen dataset lacks approved licence review")
    if manifest["contamination_audit_status"] not in ("CHECKED_NO_HITS", "REVIEWED"):
        raise CorpusError("frozen dataset lacks cleared contamination audit")
    if manifest["budget_status"] != "READY_FOR_TRAINING":
        raise CorpusError("frozen dataset has an insufficient token budget")
    return manifest
