"""Read-only Pilot source inventory and readiness evidence."""
from __future__ import annotations

import json
import math
import sqlite3
from collections import Counter
from contextlib import closing
from pathlib import Path

from .corpus_budget import plan as plan_budget
from .corpus_budget import read_token_index, supervision_by_category, supervision_by_field
from .corpus_quality import V4_CATEGORIES
from .corpus_utils import CorpusError, file_info, identity, verify_file
from .corpus_v4 import (V4_SCHEMA_VERSION, _split_isolation_check, build_registry,
                        implementation_sha256, status_report, verify_frozen)
from .data import stream_corpus_records
from .v4_pilot_sampling import (PILOT_VERSION, SAMPLE_FILE, load_sample_manifest, scan_jsonl,
                                sample_origin, verify_sample, verify_sample_binding)
from .v4_language_pilot import (LANGUAGE_TOLERANCE, LANGUAGE_VERSION, SELECTION_REPORT, SELECTED_TRAIN,
                                is_language, tokenized_extra, verify_train_selection)
from .v4_sources import PLACEHOLDERS, validate_reviewed_source, validate_source_plan

_READ_ERRORS = (CorpusError, OSError, ValueError, KeyError, TypeError, sqlite3.Error)
_TEXT_FIELDS = ("publisher", "release", "commercial_use", "redistribution", "model_training", "license_obligations")
_SOURCE_FIELDS = ("source_uri", "license", "license_evidence", "license_reviewer", "license_reviewed_at")


def _read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _relative(path, base):
    return str(path.relative_to(base)) if path.is_relative_to(base) else path.name


def _paths(source, base):
    files = set()
    for configured in source["paths"]:
        path = base / configured
        candidates = path.rglob("*") if path.is_dir() else [path]
        files.update(p.resolve() for p in candidates if p.is_file() and p.suffix.lower() in (".jsonl", ".txt"))
    if not files and source["license_status"] == "REVIEW_REQUIRED" and all(
            Path(f"data/raw/{source['source_id']}") in (Path(path), *Path(path).parents)
            for path in source["paths"]):
        staged = base / "data/staged"
        for path in (staged / source["source_id"], *staged.glob(source["source_id"] + "-*")):
            candidate = path / "documents.jsonl"
            if candidate.is_file():
                files.add(candidate.resolve())
    return sorted(files)


def _missing_evidence(entry, version):
    source = entry["v4_source"]

    def missing(value):
        return not isinstance(value, str) or not value.strip() or value.strip().upper() in PLACEHOLDERS

    fields = [key for key in _TEXT_FIELDS if missing(entry.get(key))]
    fields += [key for key in _SOURCE_FIELDS if missing(source.get(key))]
    if not entry.get("license_references"):
        fields.append("license_references")
    if source["provenance_status"] != "VERIFIED":
        fields.append("provenance_status")
    if version not in entry.get("corpus_versions", []):
        fields.append("corpus_versions")
    if not entry["local_inventory"]["files"]:
        fields.append("local_inventory.files")
    return fields


def _wiki_identity(path, info, base, source_id):
    sidecar = path.with_name(path.name + ".manifest.json")
    if not sidecar.is_file():
        return None
    saved = _read(sidecar)
    if saved.get("schema") != "wiki-dump-extraction/1":
        return None
    if (saved.get("source_id") != source_id
            or saved["sha256"] != identity({k: v for k, v in saved.items() if k != "sha256"})
            or {k: saved["output"][k] for k in ("sha256", "bytes")} != info):
        raise CorpusError("Wiki extraction manifest differs from the observed V4 input")
    dump = path.parent / Path(saved["input"]["path"]).name
    expected = {k: saved["input"][k] for k in ("sha256", "bytes")}
    verify_file(dump, expected)
    lineage = {"dump": {"path": _relative(dump, base), **expected},
               "extraction_manifest_sha256": file_info(sidecar)["sha256"],
               "extraction_status": saved.get("extraction_status", "LEGACY_MANIFEST"),
               "pages_seen": saved["pages_seen"], "pages_kept": saved["pages_kept"]}
    if "multistream-sampled" in dump.name:
        sample = path.parent / (path.name.removesuffix("-pages.jsonl") + "-sample-manifest.json")
        lineage["upstream_scope"] = "MULTISTREAM_SLICE"
        if sample.is_file():
            lineage["upstream_sampling_manifest_sha256"] = file_info(sample)["sha256"]
    return lineage


def _inspect(source, base, max_record_bytes, preview, version):
    source_id = source["source_id"]
    entry = source["source_review"]["entry"]
    result = {
        "source_id": source_id, "category": source["category"], "language": source["language"],
        "publisher": entry["publisher"], "release": entry["release"], "source_uri": source["source_uri"],
        "upstream_snapshot_uri": entry.get("upstream_snapshot_uri"),
        "license_status": source["license_status"], "available_files": [], "available_documents": 0,
        "available_characters": 0, "missing_evidence": _missing_evidence(entry, version),
        "source_evidence_status": "REVIEW_REQUIRED", "admitted": False,
    }
    paths = _paths(source, base)
    if not paths:
        result["file_status"] = "SOURCE_FILES_MISSING"
        return result
    result["file_status"] = "SOURCE_FILES_VERIFIED"
    approved = source["license_status"] == "APPROVED"
    sample_output = None
    preview_inputs = {}
    try:
        if preview:
            manifest_path, manifest = preview
            if (manifest["source_id"] != source_id or manifest["release"] != entry["release"]
                    or manifest["category"] != source["category"] or manifest["corpus_version"] != sample_origin(entry, version, manifest_path)
                    or manifest["max_record_bytes"] != max_record_bytes):
                raise CorpusError("preview differs from the configured source scope or record bound")
            configured = {str(path) for path in paths}
            output = str((manifest_path.parent / SAMPLE_FILE).resolve())
            if output in configured:
                if configured != {output}:
                    raise CorpusError("sample preview requires an exact output file set")
                verify_sample_binding(manifest_path, manifest, entry, version)
                sample_output = output
                result.update(available_documents=manifest["counts"]["selected_documents"],
                              available_characters=manifest["counts"]["selected_characters"])
            elif set(manifest["inputs"]) == configured:
                preview_inputs = manifest["inputs"]
                result.update(available_documents=manifest["counts"]["documents"],
                              available_characters=manifest["counts"]["characters"])
            else:
                raise CorpusError("preview differs from the configured source input set")
            # The approved-sample branch delegates to the strict source verifier below.
            if not (sample_output and approved):
                if verify_sample(manifest_path) != manifest:
                    raise CorpusError("Pilot sample manifest changed during preview verification")
            result["preview_sample"] = {
                k: manifest[k] for k in ("sha256", "algorithm", "seed", "basis_points", "counts",
                                         "output", "complete_input_scan", "partial_extraction")}
            result["preview_sample"].update(
                scope="SAMPLE_OUTPUT" if sample_output else "RAW_INPUTS",
                verification="PENDING" if sample_output and approved else "INTEGRITY_CHECKED")
        observed = []
        for path in paths:
            if str(path) == sample_output:
                info = manifest["output"]
            elif str(path) in preview_inputs:
                info = preview_inputs[str(path)]
            else:
                info = file_info(path)
                counts = Counter()
                if path.suffix.lower() == ".jsonl":
                    for _ in scan_jsonl(path, source_id, max_record_bytes, counts):
                        pass
                else:
                    for row in stream_corpus_records(path, max_record_bytes=max_record_bytes):
                        if "_issue" not in row and row["text"].strip():
                            counts["documents"] += 1
                            counts["characters"] += len(row["text"])
                verify_file(path, info)
                if not counts["documents"]:
                    raise CorpusError(f"source file has no readable nonempty records: {path.name}")
                result["available_documents"] += counts["documents"]
                result["available_characters"] += counts["characters"]
            observed.append({"source_id": source_id, "path": str(path), **info})
            public = {"path": _relative(path, base), **info}
            lineage = _wiki_identity(path, info, base, source_id)
            if lineage:
                public["wiki_conversion"] = lineage
            result["available_files"].append(public)
        if not result["available_documents"]:
            raise CorpusError("source has no nonempty documents")
    except _READ_ERRORS as exc:
        result.update(file_status="SOURCE_FILES_INVALID", input_error=str(exc))
        return result

    if approved:
        try:
            validate_reviewed_source(source, observed, corpus_version=version, verify_inputs=True)
            result["source_evidence_status"] = "VERIFIED"
            result["admitted"] = source.get("enabled", True)
            if sample_output:
                result["preview_sample"]["verification"] = "REPRODUCED"
        except _READ_ERRORS as exc:
            result.update(source_evidence_status="INVALID", evidence_error=str(exc))
            if sample_output:
                result["file_status"] = "SOURCE_FILES_INVALID"
                result["preview_sample"]["verification"] = "FAILED"
    return result


def _stage(directory, plan, name, required, extra):
    path = directory / f".v4/{name}.stage.json"
    marker = _read(path)
    if marker["key"] != identity([V4_SCHEMA_VERSION, plan["plan_key"], name, extra]):
        raise CorpusError(f"{name} stage key differs from the generation plan")
    if not {str(directory / filename) for filename in required} <= marker["files"].keys():
        raise CorpusError(f"{name} stage lacks artifact evidence")
    for filename, info in marker["files"].items():
        if Path(filename).parent != directory:
            raise CorpusError(f"{name} contains an unexpected artifact path")
        verify_file(Path(filename), info)
    return marker["key"]


def _stages(config, registry, directory, result, by_source, coverage):
    plan_path = directory / ".v4/plan.json"
    if not plan_path.is_file():
        return None
    generation = _read(plan_path)
    if (generation.get("demo") is not False or generation["config_sha256"] != identity(config)
            or generation["implementation_sha256"] != implementation_sha256()):
        raise CorpusError("generation plan differs from the active formal Pilot configuration")
    validate_source_plan(generation, registry, require=True, verify_inputs=True)
    policies = generation["policies"]
    if not (directory / ".v4/quality.stage.json").is_file():
        return None
    ingest = _stage(directory, generation, "ingested", ["ingested.sqlite"], {})
    quality = _stage(directory, generation, "quality", ["quality-report.json"],
                     {"ingested": ingest, "quality": policies["quality_policy_sha256"],
                      "text_policy": config["text_policy"]})
    stats = _read(directory / "quality-report.json")
    result["quality_status"] = "VERIFIED"
    reasons = {cat: Counter() for cat in V4_CATEGORIES}
    for source, row in stats["by_source"].items():
        row_stats = {field: row[field] for field in ("documents", "kept", "rejected", "review", "top_reasons")}
        by_source[source]["quality"] = row_stats
        category = by_source[source]["category"]
        for field in ("documents", "kept", "rejected", "review"):
            coverage[category][field] += row[field]
        for reason in row["top_reasons"]:
            reasons[category][reason["rule_id"], reason["action"]] += reason["findings"]
    for category, counts in reasons.items():
        coverage[category]["top_reasons"] = [
            {"rule_id": rule, "action": action, "findings": number}
            for (rule, action), number in counts.most_common(10)]

    if not (directory / ".v4/dedup.stage.json").is_file():
        return None
    dedup = _stage(directory, generation, "dedup", ["dedup-report.json", "deduplicated.sqlite"],
                   {"quality": quality, "dedup": policies["dedup_policy_sha256"]})
    report = _read(directory / "dedup-report.json")
    result["dedup_status"] = "FAIL" if report["near"]["candidate_truncated"] else "VERIFIED"
    result["dedup_counts"] = report["documents"]
    with closing(sqlite3.connect((directory / "deduplicated.sqlite").as_uri() + "?mode=ro", uri=True)) as db:
        rows = db.execute("SELECT source_id, COUNT(*), SUM(dedup_kept=1), "
                          "COALESCE(SUM(CASE WHEN dedup_kept=1 THEN LENGTH(normalized_text) ELSE 0 END), 0) "
                          "FROM docs WHERE quality_action='KEEP' GROUP BY source_id").fetchall()
    for source, before, after, chars in rows:
        row = {"dedup_deleted": before - after, "kept_documents": after, "kept_characters": chars}
        by_source[source]["dedup"] = row
        category = by_source[source]["category"]
        for field, number in row.items():
            coverage[category][field] += number

    if not (directory / ".v4/split.stage.json").is_file():
        return None
    split = _stage(directory, generation, "split", ["split-manifest.json"],
                  {"dedup": dedup, "split": policies["split_policy_sha256"], "seed": config["split"]["seed"]})
    isolation = _split_isolation_check(directory)
    result["split_status"] = isolation["status"]
    result["split_isolation"] = isolation["detail"]
    result["split_counts"] = _read(directory / "split-manifest.json")["splits"]
    if isolation["status"] != "PASS":
        return None
    estimates = Counter()
    for row in stream_corpus_records(directory / "train.text.jsonl", max_record_bytes=16 * 1024 * 1024,
                                     preserve_json_fields=True):
        category = row["category"]
        source = by_source[row["source_id"]]
        estimate = math.ceil(len(row["text"]) / config["budget"]["chars_per_token"].get(row["language"], 3.0))
        estimates[category] += estimate
        coverage[category]["train_documents"] += 1
        source["estimated_train_tokens"] = source.get("estimated_train_tokens", 0) + estimate
    for category in coverage:
        coverage[category]["estimated_train_tokens"] = estimates[category]
    result["estimated_train_tokens"] = sum(estimates.values())
    return generation, split


def _token_budget(config, directory, stages, result, coverage):
    from .v4_tokenizer_decision import verify_selection

    evidence = result.setdefault("tokenization_evidence", {
        "status": "TOKENIZER_NOT_SELECTED", "measured": False,
        "reason": "no human tokenizer decision"})
    if not (directory / "tokenizer-selection.json").is_file():
        return
    generation, split = stages
    try:
        selection = verify_selection(directory)
    except _READ_ERRORS as exc:
        evidence.update(status="INVALID", measured=False, reason=str(exc))
        raise
    result["tokenizer_status"] = "HUMAN_DECISION_VERIFIED"
    if not (directory / ".v4/tokenized.stage.json").is_file():
        evidence.update(status="SELECTED_NOT_TOKENIZED", measured=False,
                        reason="human decision verified but no tokenized Stage")
        return
    seq_len = config["targets"]["seq_len"]
    fingerprint = selection["fingerprint"]
    try:
        _stage(directory, generation, "tokenizer-file", ["tokenizer.json"],
               {"fingerprint": fingerprint, "split": split})
        verify_file(directory / "tokenizer.json", file_info(Path(selection["path"])))
        selection_key, train_selection = None, None
        if is_language(config):
            from .tokenizer import MiniLLMTokenizer

            selection_key = _stage(directory, generation, "train_selection", [SELECTED_TRAIN, SELECTION_REPORT],
                                   {"split": split, "tokenizer": fingerprint})
            train_selection = verify_train_selection(directory, config, MiniLLMTokenizer.load(directory / "tokenizer.json"))
        tokenized_key = _stage(
            directory, generation, "tokenized",
            ["tokenization-report.json", "train.bin", "train.token-index.jsonl"],
            tokenized_extra(config, split=split, fingerprint=fingerprint, seq_len=seq_len,
                            selection_key=selection_key))
        tokenized = _read(directory / "tokenization-report.json")
        index_path = directory / "train.token-index.jsonl"
        index = read_token_index([index_path])
        packing = supervision_by_category(
            ((row["category"], row["tokens"]) for row in index["records"]), seq_len)
        if (tokenized["tokenizer"]["fingerprint"] != fingerprint or tokenized["seq_len"] != seq_len
                or tokenized["packing"] != packing
                or (directory / "train.bin").stat().st_size != 2 * packing["raw_tokens"]):
            raise CorpusError("tokenization differs from verified decision or Pilot sequence length")
        by_source = supervision_by_field(index["records"], seq_len, "source_id")
        if sum(by_source["by_field"].values()) != packing["effective_tokens"]:
            raise CorpusError("tokenization source supervision differs from category totals")
        available = {cat: {"documents": packing["documents_by_category"].get(cat, 0),
                           "effective_tokens": packing["by_category"].get(cat, 0)} for cat in coverage}
        budget = plan_budget(available, config["targets"]["effective_train_tokens"],
                             config["targets"]["category_token_ratios"], seq_len=seq_len,
                             tokenizer_fingerprint=fingerprint, license_review_status="APPROVED",
                             fraction_tolerance=LANGUAGE_TOLERANCE if is_language(config) else None)
        if budget["total_available_effective_tokens"] != tokenized["effective_train_tokens"]:
            raise CorpusError("tokenization Train supervision differs from category totals")
    except _READ_ERRORS as exc:
        evidence.update(status="INVALID", measured=False, reason=str(exc))
        raise
    if train_selection is not None:
        result["train_selection"] = {k: v for k, v in train_selection.items() if k != "selected_documents"}
        result["train_selection"]["category_budget"] = budget["categories"]
        result["train_selection"]["budget_status"] = budget["status"]
    result["effective_train_tokens"] = tokenized["effective_train_tokens"]
    result["budget_gap"]["measured_category_gap_total"] = budget["total_gap_effective_tokens"]
    for cat, row in budget["categories"].items():
        coverage[cat]["effective_train_tokens"] = row["available_effective_tokens"]
        coverage[cat]["effective_token_gap"] = row["gap_effective_tokens"]
    evidence.update({
        "status": "VERIFIED", "measured": True,
        "reason": "human decision and tokenized Stage verified",
        "tokenizer_fingerprint": fingerprint, "seq_len": seq_len,
        "tokenized_stage_key": tokenized_key,
        "token_index": file_info(index_path),
        "tokenization_report": file_info(directory / "tokenization-report.json"),
        "train_bin": file_info(directory / "train.bin"),
        "by_category": packing["by_category"],
        "documents_by_category": packing["documents_by_category"],
        "by_source": by_source["by_field"],
        "documents_by_source": by_source["documents_by_field"],
        "raw_tokens": packing["raw_tokens"],
        "effective_train_tokens": tokenized["effective_train_tokens"],
    })


def _contamination(config, base, directory):
    from .corpus import read_records
    from .corpus_contamination import audit

    policy = config["contamination"]
    assets = [{**asset, "path": str(base / asset["path"])} for asset in policy["assets"]]
    with closing(read_records(directory / "train.text.jsonl")) as records:
        result = audit(assets, records, decisions=base / policy["decisions"] if policy["decisions"] else None,
                       min_prompt_chars=policy["min_prompt_chars"], max_hits=policy["max_hits"],
                       long_span_containment=policy["long_span_containment"])
    return result["status"]


def pilot_readiness(config, base, directory, *, sample_manifests=()):
    """Inspect source files and verified V4 artifacts without building a dataset."""
    version = config["corpus_version"]
    if version not in (PILOT_VERSION, LANGUAGE_VERSION):
        raise CorpusError("requires the V4 1M Pilot configuration")
    base, directory = Path(base).resolve(), Path(directory).resolve()
    previews = {}
    for path in sample_manifests:
        path = Path(path).resolve()
        preview = load_sample_manifest(path)
        source_id = preview["source_id"]
        if source_id in previews:
            raise CorpusError(f"duplicate sample for source {source_id}")
        previews[source_id] = (path, preview)
    if previews.keys() - {source["source_id"] for source in config["sources"]}:
        raise CorpusError("sample source absent from the review registry")
    by_source = {source["source_id"]: _inspect(source, base, config["quality"]["max_record_bytes"],
                                               previews.get(source["source_id"]), version)
                 for source in config["sources"]}
    targets = config["targets"]
    target = targets["effective_train_tokens"]
    coverage = {cat: {"target_ratio": ratio, "target_effective_train_tokens": round(target * ratio),
                      "available_documents": 0, "available_characters": 0,
                      "admitted_documents": 0, "documents": 0, "kept": 0, "rejected": 0, "review": 0,
                      "dedup_deleted": 0, "kept_documents": 0, "kept_characters": 0,
                      "train_documents": 0, "estimated_train_tokens": None,
                      "effective_train_tokens": None, "effective_token_gap": None}
                for cat, ratio in targets["category_token_ratios"].items()}
    for source in by_source.values():
        if source["category"] not in coverage:
            continue
        row = coverage[source["category"]]
        if source["file_status"] == "SOURCE_FILES_VERIFIED":
            row["available_documents"] += source["available_documents"]
            row["available_characters"] += source["available_characters"]
        if source["admitted"]:
            row["admitted_documents"] += source["available_documents"]

    registry = build_registry(config["sources"], demo=False, license_policy=config["license_policy"])
    policy = status_report(config, base, dataset_dir=directory)
    blockers = set(policy["blockers"])
    if any(source["file_status"] == "SOURCE_FILES_MISSING"
           and (source["category"] not in coverage or coverage[source["category"]]["target_ratio"])
           for source in by_source.values()):
        blockers.add("SOURCE_FILES_MISSING")
    evidence = "VERIFIED" if registry["sources"] and all(by_source[s["source_id"]]["admitted"]
                                                         for s in registry["sources"]) else "REVIEW_REQUIRED"
    result = {"schema_version": 1, "corpus_version": version, "status": "DATA_NOT_READY",
              "source_status": by_source, "category_coverage": coverage,
              "approved_source_count": sum(s["license_status"] == "APPROVED" for s in by_source.values()),
              "admitted_source_count": sum(s["admitted"] for s in by_source.values()),
              "available_source_count": sum(s["file_status"] == "SOURCE_FILES_VERIFIED"
                                            for s in by_source.values()),
              "source_evidence_status": evidence, "quality_status": "NOT_RUN", "dedup_status": "NOT_RUN",
              "split_status": "NOT_RUN", "tokenizer_status": "TOKENIZER_NOT_FINALIZED",
              "estimated_train_tokens": None, "effective_train_tokens": None,
              "target_effective_train_tokens": target,
              "budget_gap": {"unverified_target_tokens": target, "measured_category_gap_total": None},
              "tokenization_evidence": {"status": "TOKENIZER_NOT_SELECTED", "measured": False,
                                        "reason": "no human tokenizer decision"},
              "contamination_status": "NOT_RUN", "pilot_preflight_status": "BLOCKED_DATA_NOT_READY",
              "frozen_integrity_status": "NOT_RUN", "frozen_budget_match": None,
              "pilot_policy_validation": None}
    if evidence != "VERIFIED":
        blockers.add("SOURCE_EVIDENCE_INCOMPLETE")
    try:
        stages = _stages(config, registry, directory, result, by_source, coverage)
        if stages and result["split_status"] == "PASS":
            _token_budget(config, directory, stages, result, coverage)
            if (directory / "contamination-report.json").is_file():
                result["contamination_status"] = _contamination(config, base, directory)
    except _READ_ERRORS as exc:
        blockers.add("STAGE_EVIDENCE_INVALID")
        result["stage_error"] = str(exc)

    missing_categories = []
    for cat, row in coverage.items():
        if not row["target_ratio"]:
            row.update(status="NOT_REQUIRED", effective_token_gap=0)
            continue
        measured = row["effective_token_gap"]
        if (not row["admitted_documents"] or
                (result["split_status"] == "PASS" and not row["train_documents"]) or
                (measured is not None and measured > 0)):
            missing_categories.append(cat)
    if missing_categories:
        blockers.add("CATEGORY_DATA_INSUFFICIENT")
    result["insufficient_categories"] = sorted(missing_categories)
    if result.get("train_selection", {}).get("budget_status") == "CATEGORY_MIX_OUT_OF_TOLERANCE":
        blockers.add("CATEGORY_MIX_OUT_OF_TOLERANCE")
    if result["split_status"] != "PASS":
        blockers.add("SPLIT_NOT_VERIFIED")
    if result["dedup_status"] == "FAIL":
        blockers.add("DEDUP_CANDIDATE_TRUNCATION")
    effective = result["effective_train_tokens"]
    if effective is not None:
        result["budget_gap"]["unverified_target_tokens"] = max(0, target - effective)
    if result["tokenizer_status"] == "TOKENIZER_NOT_FINALIZED":
        blockers.add("TOKENIZER_NOT_FINALIZED")
    if (result["estimated_train_tokens"] is not None and effective is None
            and any(row["estimated_train_tokens"] < row["target_effective_train_tokens"]
                    for row in coverage.values())):
        blockers.add("ESTIMATED_TRAIN_SUPPLY_SHORTFALL")
    optional_sources = {"SOURCE_FILES_MISSING", "TOKENIZER_NOT_FINALIZED"}
    text_ready = not (blockers - optional_sources)
    if text_ready:
        if result["tokenizer_status"] == "TOKENIZER_NOT_FINALIZED":
            result["status"] = "READY_FOR_TOKENIZER_EVALUATION"
        elif effective is None:
            blockers.add("EFFECTIVE_BUDGET_NOT_ASSESSED")
        elif result["contamination_status"] not in ("CHECKED_NO_HITS", "REVIEWED"):
            blockers.add("CONTAMINATION_NOT_CLEARED")
        else:
            result["status"] = "READY_FOR_FREEZE"
            blockers.add("FINAL_AUDIT_REQUIRED")
    if result["status"] == "READY_FOR_FREEZE" and (directory / "FROZEN").is_file():
        _check_frozen_readiness(directory, effective, result, blockers)
    result["blockers"] = sorted(blockers)
    return result


def _check_frozen_readiness(directory, effective, result, blockers):
    from .v4_admission import validate_v4_pilot_policy

    result["status"] = "DATA_NOT_READY"
    try:
        manifest = verify_frozen(directory)
        frozen_tokens = manifest["effective_train_tokens"]
    except _READ_ERRORS as exc:
        result["frozen_integrity_status"] = "FAIL"
        result["frozen_error"] = str(exc)
        blockers.add("FROZEN_VERIFICATION_FAILED")
        return
    result["frozen_integrity_status"] = "VERIFIED"
    result["frozen_budget_match"] = type(frozen_tokens) is int and frozen_tokens == effective
    if not result["frozen_budget_match"]:
        result["frozen_error"] = "frozen dataset differs from measured Pilot budget"
        blockers.add("FROZEN_VERIFICATION_FAILED")
        return

    try:
        policy = validate_v4_pilot_policy(directory)
    except _READ_ERRORS as exc:
        policy = {"ok": False, "status": "BLOCKED_POLICY_EVIDENCE_MISSING", "errors": [str(exc)]}
    result["pilot_policy_validation"] = policy
    result["pilot_preflight_status"] = policy.get("status", "BLOCKED_POLICY_EVIDENCE_MISSING")
    if policy.get("ok") is True and policy.get("status") == "POLICY_MATCH":
        result["status"] = "FROZEN_VERIFIED"
        blockers.discard("FINAL_AUDIT_REQUIRED")
    else:
        blockers.add("FROZEN_POLICY_INVALID")
