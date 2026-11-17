"""Read-only source, tokenizer and independent corpus-budget assessments."""
from __future__ import annotations

import json
import resource
import time
import sqlite3
from pathlib import Path

from .corpus_budget import plan
from .corpus_quality import V4_CATEGORIES
from .corpus_utils import CorpusError, file_info, identity
from .corpus_v4 import load_config, status_report, verify_frozen
from .data import stream_corpus_records
from .tokenizer_v4 import compare_candidates, load_eval_texts, parameter_budget
from .v4_sources import load_source_review

TOPICS = {"geography": ("地理", "河流", "山脉"), "history": ("历史", "朝代", "史料"),
          "physics": ("物理", "力学", "光速"), "chemistry": ("化学", "元素", "分子"),
          "biology": ("生物", "细胞", "植物"), "math": ("数学", "方程", "定理"),
          "computing": ("计算机", "算法", "程序"), "daily_life": ("生活", "食物", "交通"),
          "culture_literature": ("文化", "文学", "诗歌")}


def inspect_local(paths):
    files = sorted({p.resolve() for pattern in paths for p in
                    ([Path(pattern)] if Path(pattern).is_file() else Path(pattern).rglob("*"))
                    if p.is_file() and p.suffix.lower() in (".txt", ".jsonl")})
    result = {"files": {}, "documents": 0, "characters": 0, "invalid_records": 0,
              "estimated_tokens": None, "raw_tokens": None, "effective_tokens": None,
              "topics_estimated": {key: 0 for key in TOPICS}, "topic_unknown_documents": 0}
    for path in files:
        info = file_info(path)
        for record in stream_corpus_records(path):
            if "_issue" in record:
                result["invalid_records"] += 1
                continue
            result["documents"] += 1
            text = record["text"]
            result["characters"] += len(text)
            matched = False
            for topic, words in TOPICS.items():
                if any(word in text for word in words):
                    result["topics_estimated"][topic] += 1
                    matched = True
            result["topic_unknown_documents"] += not matched
        if file_info(path) != info:
            raise CorpusError(f"source changed during read-only assessment: {path}")
        result["files"][str(path)] = info
    return result


def corpus_readiness(config_paths, *, dataset_dirs=None, review_path="configs/corpus_v4_sources.review.json",
                     excluded_local=()):
    started = time.monotonic()
    review = load_source_review(Path(review_path))
    sources = []
    for entry in review["sources"]:
        source = entry["v4_source"]
        sources.append({"source_id": source["source_id"], "category": source["category"],
                        "license_status": source["license_status"],
                        "provenance_status": source["provenance_status"],
                        "local": inspect_local(source["paths"])})
    versions = {}
    dataset_dirs = dataset_dirs or {}
    for label, path in config_paths.items():
        config = load_config(path)
        status = status_report(config, Path.cwd())
        directory = Path(dataset_dirs.get(label, f"artifacts/corpus/{config['corpus_version']}"))
        available = {c: {"documents": 0, "effective_tokens": 0} for c in V4_CATEGORIES}
        blockers = list(status["blockers"])
        frozen = None
        admission = {"status": "BLOCKED_DATA_NOT_READY", "ok": False}
        try:
            frozen = verify_frozen(directory)
            if frozen["corpus_version"] != config["corpus_version"]:
                raise CorpusError("dataset version differs from requested independent version")
            # Policy identity is checked for both scales using the same existing
            # validator and each explicit profile; the Pilot default stays pinned.
            from .v4_admission import validate_v4_pilot_policy
            admission = validate_v4_pilot_policy(directory, path)
            if not admission["ok"]:
                raise CorpusError(admission["status"])
            frozen_budget = json.loads((directory / "budget-report.json").read_text(encoding="utf-8"))
            for c, row in frozen_budget["categories"].items():
                available[c] = {"documents": row["available_documents"],
                                "effective_tokens": row["available_effective_tokens"]}
        except (OSError, ValueError, KeyError, CorpusError) as exc:
            frozen = None
            available = {c: {"documents": 0, "effective_tokens": 0} for c in V4_CATEGORIES}
            blockers.append(f"FROZEN_DATA_NOT_READY: {exc}")
        counts = {"cleaning_documents": {}, "deduplicated_documents": {}}
        if frozen:
            for field, filename, predicate in (("cleaning_documents", "reviewed.sqlite", "quality_action='KEEP'"),
                                               ("deduplicated_documents", "deduplicated.sqlite", "dedup_kept=1")):
                # Intermediate SQLite files are already verified by verify_frozen.
                db = sqlite3.connect(f"file:{directory.resolve() / filename}?mode=ro", uri=True)
                try:
                    counts[field] = dict(db.execute(f"SELECT category, COUNT(*) FROM docs WHERE {predicate} GROUP BY category"))
                finally:
                    db.close()
        budget = plan(available, config["targets"]["effective_train_tokens"],
                      config["targets"]["category_token_ratios"], seq_len=config["targets"]["seq_len"],
                      tokenizer_fingerprint=frozen["tokenizer_fingerprint"] if frozen else None,
                      license_review_status="APPROVED" if frozen else "NOT_ASSESSED",
                      contamination_status=frozen["contamination_audit_status"] if frozen else "NOT_ASSESSED")
        # The planner keeps NOT_ASSESSED when no tokenizer exists. The supply
        # assessment separately reports the known zero qualified-token shortfall.
        for c, row in budget["categories"].items():
            matches = [s for s in sources if s["category"] == c]
            row.update(candidate_sources=len(matches), approved_sources=sum(s["license_status"] == "APPROVED" for s in matches),
                       local_raw_documents=sum(s["local"]["documents"] for s in matches),
                       approved_raw_documents=sum(s["local"]["documents"] for s in matches if s["license_status"] == "APPROVED"),
                       cleaning_documents=counts["cleaning_documents"].get(c, 0) if frozen else None,
                       deduplicated_documents=counts["deduplicated_documents"].get(c, 0) if frozen else None,
                       raw_tokens=frozen["category_token_distribution"]["raw"].get(c, {}).get("tokens", 0) if frozen else None,
                       stage_statistics="NOT_RUN" if not frozen else "SEE_FROZEN_REPORTS",
                       missing_sources=[s["source_id"] for s in matches if s["license_status"] != "APPROVED" or not s["local"]["files"]])
        versions[label] = {"corpus_version": config["corpus_version"], "dataset_directory": str(directory),
                           "status": "DATA_READY" if frozen else "INSUFFICIENT_CORPUS",
                           "frozen_content_id": frozen["content_id"] if frozen else None,
                           "independent_effective_tokens": sum(r["effective_tokens"] for r in available.values()),
                           "budget": budget, "preparation": status, "admission": admission,
                           "blockers": [] if frozen else blockers}
    dirs = [v["dataset_directory"] for v in versions.values()]
    if len(set(map(lambda d: str(Path(d).resolve()), dirs))) != len(dirs):
        raise CorpusError("independent corpus versions cannot share an output directory")
    excluded = [{"reason": "Historical or unverified material; never counted as approved V4 supply",
                 "inventory": inspect_local([p])} for p in excluded_local]
    measured_documents = sum(s["local"]["documents"] for s in sources) + sum(s["inventory"]["documents"] for s in excluded)
    elapsed = time.monotonic() - started
    return {"schema_version": 1, "sources": sources, "source_review_sha256": file_info(Path(review_path))["sha256"],
            "versions": versions, "data_status": "DATA_READY" if all(v["frozen_content_id"] for v in versions.values()) else "DATA_NOT_READY",
            "excluded_local": excluded,
            "inventory_performance": {"seconds": elapsed, "documents_per_second": measured_documents / max(elapsed, 1e-9),
                "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                "scope": "read-only inventory including excluded local material; not V4 scale benchmark"},
            "knowledge_coverage": {"status": "ESTIMATED_KEYWORD_PRESENCE", "rules": TOPICS,
                "limitations": "Overlapping keyword counts, no semantic coverage claim; unknown documents remain unknown."},
            "scale_benchmark": "NOT_RUN", "gpu_training": "NOT_RUN"}


def tokenizer_readiness(dataset_dirs, *, eval_asset, baseline_path=None):
    texts = load_eval_texts(eval_asset)
    versions = {}
    for label, directory in dataset_dirs.items():
        directory = Path(directory)
        # Only hash-bound pipeline reports count as V4 candidate evidence.
        report = {"status": "NOT_ASSESSED", "reason": "TOKENIZER_TRAINING_BLOCKED: no approved Train candidates",
                  "candidates": {name: {"status": "NOT_ASSESSED"} for name in ("v4-24576", "v4-32768")}}
        if (directory / "tokenizer-evaluation.json").is_file():
            from .tokenizer_v4 import candidate_record, guard_train_only
            from .corpus_utils import verify_file
            guard_train_only(directory / "train.text.jsonl", directory / "split-manifest.json")
            for stage in ("tokenizers", "tokenizer_eval"):
                marker = json.loads((directory / f".v4/{stage}.stage.json").read_text(encoding="utf-8"))
                for name, info in marker["files"].items():
                    verify_file(Path(name), info)
            published = json.loads((directory / "tokenizer-evaluation.json").read_text(encoding="utf-8"))
            if published["fixed_texts"]["sha256"] != identity(texts) or published["demo_only"]:
                raise CorpusError("tokenizer evaluation assets differ or candidate is DEMO_ONLY")
            candidates = json.loads((directory / "tokenizer-candidates.json").read_text(encoding="utf-8"))["candidates"]
            for candidate in candidates:
                if candidate != candidate_record(Path(candidate["directory"])):
                    raise CorpusError("candidate identity mismatch")
            report = {"status": "ASSESSED", "report": published, "files": {
                name: file_info(directory / name) for name in ("tokenizer-candidates.json", "tokenizer-evaluation.json")}}
        report["tokenizer_status"] = "TOKENIZER_NOT_FINALIZED"
        if (directory / "tokenizer-selection.json").is_file():
            from .v4_tokenizer_decision import verify_selection
            report["selection"] = verify_selection(directory)
            report["tokenizer_status"] = "TOKENIZER_FINALIZED"
        versions[label] = report
    reference = compare_candidates([], texts, baseline_path=baseline_path)
    return {"schema_version": 2, "status": "NOT_ASSESSED" if all(v["status"] == "NOT_ASSESSED" for v in versions.values()) else "ASSESSED",
            "versions": versions, "historical_reference_only": reference["baseline"],
            "fixed_texts": {**reference["fixed_texts"], "file": file_info(Path(eval_asset)),
                            "purpose": "tokenization efficiency and reversibility; never model knowledge scores or fitting data"},
            "parameter_budget": parameter_budget([24576, 32768], seq_len=2048),
            "tokenizer_status": "TOKENIZER_FINALIZED" if versions and all(v["tokenizer_status"] == "TOKENIZER_FINALIZED" for v in versions.values()) else "TOKENIZER_NOT_FINALIZED",
            "human_decision": "SEE_VERSION_SELECTION_EVIDENCE" if any("selection" in v for v in versions.values()) else "NOT_RECORDED"}
