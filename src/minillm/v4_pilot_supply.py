"""Read-only V4 Pilot supply accounting, token-gap analysis and blocker reporting.

The analysis intentionally consumes the existing readiness, quality, dedup,
split and budget artifacts instead of re-implementing their policies.  It never
builds, trains, selects a tokenizer or writes into the dataset directory.

Two different quantities are kept separate:

* ``estimated_train_tokens`` uses the configured ``chars_per_token`` budget
  estimate from the Train split only.  It is explicitly not a measurement.
* ``effective_train_tokens`` uses the exact packed supervision rule from
  :mod:`minillm.corpus_budget` and is available only after a human-selected
  tokenizer has tokenized the dataset.  Until then every measured field is
  ``null`` with a reason.

The report also proves that source-level and category-level accounting do not
double count: every published total is checked against the sum of its parts.
"""
from __future__ import annotations

import json
import math
import sqlite3
from collections import Counter
from contextlib import closing
from pathlib import Path

from .corpus_budget import supervision_by_category, supervision_by_field
from .corpus_quality import V4_CATEGORIES
from .corpus_utils import CorpusError, file_info, identity, sha256, verify_file
from .data import stream_corpus_records
from .v4_pilot_readiness import pilot_readiness

SPLITS = ("train", "val", "test")
MAX_TEXT_RECORD_BYTES = 16 * 1024 * 1024


def _read(path):
    return _json_loads(Path(path).read_text(encoding="utf-8"), path)


def _json_loads(text, path):
    try:
        return json.loads(text)
    except (TypeError, ValueError) as exc:
        raise CorpusError(f"invalid JSON evidence: {path}") from exc


def _maybe_read(path):
    path = Path(path)
    return _read(path) if path.is_file() else None


def _relative(path, base):
    path, base = Path(path), Path(base)
    try:
        return str(path.resolve().relative_to(base.resolve()))
    except ValueError:
        return path.name


def _stage_stats(dataset, name):
    path = Path(dataset) / ".v4" / f"{name}.stage.json"
    if not path.is_file():
        return None
    return _read(path).get("stats")


def _canonical_source_row(source):
    preview = source.get("preview_sample") or {}
    counts = preview.get("counts") or {}
    return {
        "source_id": source["source_id"],
        "category": source["category"],
        "language": source["language"],
        "license_status": source["license_status"],
        "provenance_status": source.get("provenance_status"),
        "source_evidence_status": source["source_evidence_status"],
        "admitted": bool(source["admitted"]),
        "file_status": source["file_status"],
        "release": source.get("release"),
        "source_uri": source.get("source_uri"),
        "upstream_snapshot_uri": source.get("upstream_snapshot_uri"),
        "input_scope": preview.get("scope"),
        "approved_input_documents": source["available_documents"] if source["admitted"] else 0,
        "approved_input_characters": source["available_characters"] if source["admitted"] else 0,
        "available_input_documents": source["available_documents"],
        "available_input_characters": source["available_characters"],
        "upstream_sample_documents": counts.get("documents"),
        "upstream_sample_characters": counts.get("characters"),
        "sample_sha256": preview.get("sha256"),
        "sample_verification": preview.get("verification"),
        "missing_evidence": list(source.get("missing_evidence") or []),
    }


def _ingest_by_source(stats):
    if not stats:
        return None, None
    accepted, seen = Counter(), Counter()
    for row in stats.get("by_source_reason", []):
        if not isinstance(row, dict):
            raise CorpusError("invalid ingest source accounting")
        if row.get("reason") == "accepted_unique":
            accepted[row["source_id"]] += row["n"]
        elif row.get("reason") == "records_seen":
            seen[row["source_id"]] += row["n"]
    return dict(accepted), dict(seen)


def _dedup_by_source(dataset):
    """Exact/near deletions per source from the stage-verified SQLite copy."""
    path = Path(dataset) / "deduplicated.sqlite"
    if not path.is_file():
        return None, None
    with closing(sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)) as db:
        rows = db.execute(
            "SELECT duplicate_source_id, relation, COUNT(*) FROM dedup_relations "
            "GROUP BY duplicate_source_id, relation ORDER BY duplicate_source_id, relation").fetchall()
    by_source = {}
    exact_total = near_total = 0
    for source_id, relation, count in rows:
        row = by_source.setdefault(source_id, {"exact": 0, "near": 0})
        if relation not in row:
            raise CorpusError(f"unknown dedup relation: {relation}")
        row[relation] += count
        if relation == "exact":
            exact_total += count
        else:
            near_total += count
    for row in by_source.values():
        row["deleted"] = row["exact"] + row["near"]
    return by_source, {"exact": exact_total, "near": near_total}


def _split_documents(dataset):
    """Verified assignment counts per source and per split."""
    dataset = Path(dataset)
    manifest_path = dataset / "split-manifest.json"
    if not manifest_path.is_file():
        return None, None
    manifest = _read(manifest_path)
    files = manifest.get("files") or {}
    assignment = dataset / "split-assignment.jsonl"
    info = files.get("split-assignment.jsonl")
    if info is None:
        raise CorpusError("split manifest lacks assignment evidence")
    if info != file_info(assignment):
        raise CorpusError("split assignment differs from the manifest identity")
    if sha256(assignment) != manifest.get("assignment_sha256"):
        raise CorpusError("split assignment SHA256 differs from the manifest")
    by_source = {name: Counter() for name in SPLITS}
    totals = Counter()
    for number, line in enumerate(assignment.open(encoding="utf-8"), 1):
        if not line.strip():
            continue
        row = _json_loads(line, f"{assignment}:{number}")
        if not isinstance(row, dict) or row.get("split") not in SPLITS \
                or not isinstance(row.get("source_id"), str) or not row["source_id"]:
            raise CorpusError(f"{assignment}:{number}: invalid split assignment")
        by_source[row["split"]][row["source_id"]] += 1
        totals[row["split"]] += 1
    return by_source, dict(totals)


def _train_rows(dataset, chars_per_token):
    """Train-only character/token estimates with explicit source and category."""
    dataset = Path(dataset)
    path = dataset / "train.text.jsonl"
    if not path.is_file():
        return None, None
    by_source, by_category = {}, {}
    records = stream_corpus_records(path, max_record_bytes=MAX_TEXT_RECORD_BYTES,
                                    preserve_json_fields=True)
    try:
        for number, row in enumerate(records, 1):
            if "_issue" in row:
                raise CorpusError(f"{path}:{number}: unreadable Train record")
            if row.get("split") != "train":
                raise CorpusError(f"{path}:{number}: non-Train record in Train split")
            source_id, category = row.get("source_id"), row.get("category")
            language, text = row.get("language"), row.get("text")
            if category not in V4_CATEGORIES or not isinstance(source_id, str) \
                    or not isinstance(language, str) or not isinstance(text, str):
                raise CorpusError(f"{path}:{number}: invalid Train record metadata")
            estimate = math.ceil(len(text) / chars_per_token.get(language, 3.0))
            for target, key in ((by_source, source_id), (by_category, category)):
                item = target.setdefault(key, {"documents": 0, "characters": 0,
                                               "estimated_train_tokens": 0})
                item["documents"] += 1
                item["characters"] += len(text)
                item["estimated_train_tokens"] += estimate
    finally:
        records.close()
    return by_source, by_category


def _split_estimates(dataset, chars_per_token):
    """Character/token estimates for every split; only Train enters the budget."""
    dataset = Path(dataset)
    result = {}
    for name in SPLITS:
        path = dataset / f"{name}.text.jsonl"
        if not path.is_file():
            return None
        characters = tokens = documents = 0
        records = stream_corpus_records(path, max_record_bytes=MAX_TEXT_RECORD_BYTES,
                                        preserve_json_fields=True)
        try:
            for number, row in enumerate(records, 1):
                if "_issue" in row:
                    raise CorpusError(f"{path}:{number}: unreadable {name} record")
                language, text = row.get("language"), row.get("text")
                if not isinstance(language, str) or not isinstance(text, str):
                    raise CorpusError(f"{path}:{number}: invalid {name} record")
                documents += 1
                characters += len(text)
                tokens += math.ceil(len(text) / chars_per_token.get(language, 3.0))
        finally:
            records.close()
        result[name] = {"documents": documents, "characters": characters,
                        "estimated_tokens": tokens,
                        "included_in_train_budget": name == "train"}
    return result


def _stage_artifact(dataset, stage, filename):
    """Verify one published artifact against its Stage manifest.

    This proves the artifact hash/size binding; the formal generation identity
    still comes from ``verify_selection``/readiness.  It never treats mere file
    existence as stage success.
    """
    marker_path = Path(dataset) / ".v4" / f"{stage}.stage.json"
    if not marker_path.is_file():
        return {"status": "NOT_RUN", "reason": f"{stage} Stage is absent"}
    try:
        marker = _read(marker_path)
        path = (Path(dataset) / filename).resolve()
        info = (marker.get("files") or {}).get(str(path))
        if info is None:
            return {"status": "INVALID", "reason": f"{filename} absent from {stage} Stage evidence"}
        verify_file(path, info)
    except (CorpusError, OSError, ValueError, KeyError, TypeError):
        return {"status": "INVALID", "reason": f"artifact evidence invalid for {filename}"}
    return {"status": "VERIFIED", "stage_key": marker.get("key"),
            "sha256": info["sha256"], "bytes": info["bytes"]}


def _candidate_evidence(dataset, report, config):
    """Reuse the tokenizer candidate-record contract for source binding."""
    if report is None:
        return {"status": "NOT_RUN", "reason": "no tokenizer candidate report"}
    from .tokenizer_v4 import candidate_record

    dataset = Path(dataset)
    try:
        split_sha = file_info(dataset / "split-manifest.json")["sha256"]
        train_sha = file_info(dataset / "train.text.jsonl")["sha256"]
        if (report.get("split_manifest_sha256") != split_sha
                or report.get("training_split_sha256") != train_sha):
            raise CorpusError("candidate report is not bound to the current Split")
        for candidate in report.get("candidates", []):
            verified = candidate_record(Path(candidate["directory"]))
            if verified != candidate:
                raise CorpusError("candidate report differs from verified candidate artifact")
            metadata = verified.get("metadata", {})
            if (metadata.get("split_manifest_sha256") != split_sha
                    or metadata.get("train_split_sha256") != train_sha
                    or metadata.get("text_policy") != config["text_policy"]
                    or metadata.get("demo_only") is not False):
                raise CorpusError("candidate metadata is not bound to this formal Pilot")
    except (CorpusError, OSError, ValueError, KeyError, TypeError):
        return {"status": "INVALID", "reason": "candidate evidence is not bound to this formal Pilot"}
    return {"status": "VERIFIED", "reason": "candidate records and Split/train binding verified"}


def _tokenizer_report(dataset, base, readiness, config):
    dataset, base = Path(dataset), Path(base)

    def safe_read(path):
        try:
            return _maybe_read(path), None
        except (CorpusError, OSError, ValueError, KeyError, TypeError):
            return None, f"unreadable evidence: {Path(path).name}"

    evaluation, evaluation_error = safe_read(dataset / "tokenizer-evaluation.json")
    candidates, candidates_error = safe_read(dataset / "tokenizer-candidates.json")
    selection, selection_error = safe_read(dataset / "tokenizer-selection.json")
    evidence = readiness.get("tokenization_evidence") or {}
    evidence_status = evidence.get("status", "TOKENIZER_NOT_SELECTED")
    evaluation_artifact = _stage_artifact(dataset, "tokenizer_eval", "tokenizer-evaluation.json")
    candidate_artifact = _stage_artifact(dataset, "tokenizers", "tokenizer-candidates.json")
    candidate_check = _candidate_evidence(dataset, candidates, config)
    if evaluation_artifact["status"] == "VERIFIED":
        from .v4_tokenizer_decision import verify_evaluation_stage

        try:
            verify_evaluation_stage(dataset, config, base=base)
        except (CorpusError, OSError, ValueError, KeyError, TypeError) as exc:
            evaluation_artifact.update(status="INVALID", reason=str(exc))
    unverified_tokenized = bool(
        (dataset / "train.token-index.jsonl").is_file()
        or (dataset / ".v4/tokenized.stage.json").is_file()) and evidence_status != "VERIFIED"
    invalid = bool(evaluation_error or candidates_error or selection_error
                   or evidence_status == "INVALID"
                   or evaluation_artifact["status"] == "INVALID"
                   or candidate_artifact["status"] == "INVALID"
                   or candidate_check["status"] == "INVALID"
                   or unverified_tokenized)
    selection_verified = evidence_status in ("VERIFIED", "SELECTED_NOT_TOKENIZED")
    if invalid:
        evaluation_status = "INVALID"
    elif evaluation is None:
        evaluation_status = "NOT_RUN"
    elif evaluation_artifact["status"] == "VERIFIED" and candidate_check["status"] == "VERIFIED" and candidate_artifact["status"] == "VERIFIED":
        evaluation_status = evaluation.get("status", "COMPARED")
    else:
        evaluation_status = "UNVERIFIED"
    if invalid:
        selection_status = "TOKENIZER_EVIDENCE_INVALID"
        run_status = "TOKENIZER_EVIDENCE_INVALID"
    elif selection_verified:
        selection_status = "HUMAN_DECISION_VERIFIED"
        run_status = "TOKENIZER_SELECTED"
    elif evaluation_status == "COMPARED":
        selection_status = "HUMAN_TOKENIZER_DECISION_REQUIRED"
        run_status = "TOKENIZER_EVALUATED"
    elif readiness.get("split_status") == "PASS":
        selection_status = "TOKENIZER_EVALUATION_REQUIRED"
        run_status = "TOKENIZER_EVALUATION_REQUIRED"
    else:
        selection_status = "TOKENIZER_NOT_READY"
        run_status = "TOKENIZER_NOT_FINALIZED"
    if evidence_status == "VERIFIED":
        tokenization_status = "VERIFIED"
    elif evidence_status == "SELECTED_NOT_TOKENIZED" and not unverified_tokenized:
        tokenization_status = "NOT_RUN"
    elif evidence_status == "INVALID" or unverified_tokenized:
        tokenization_status = "INVALID"
    else:
        tokenization_status = "BLOCKED"
    result = {
        "status": readiness.get("tokenizer_status", "TOKENIZER_NOT_FINALIZED"),
        "run_status": run_status,
        "evaluation_status": evaluation_status,
        "selection_status": selection_status,
        "tokenization_status": tokenization_status,
        "evidence_status": "INVALID" if invalid else ("VERIFIED" if selection_verified else "UNVERIFIED"),
        "evaluation_evidence_status": evaluation_artifact["status"],
        "candidate_evidence_status": candidate_check["status"],
        "candidate_artifact_status": candidate_artifact["status"],
        "eval_file": None if evaluation is None else {
            "path": _relative(dataset / "tokenizer-evaluation.json", base),
            "sha256": file_info(dataset / "tokenizer-evaluation.json")["sha256"],
            "artifact_status": evaluation_artifact["status"],
            "stage_key": evaluation_artifact.get("stage_key")},
        "selection_file": None,
        "candidates": [],
        "baseline": None,
        "recommendation": None,
        "parameter_budget": None,
        "fixed_eval_categories": [],
        "selection_error": selection_error,
        "evaluation_error": evaluation_error,
        "candidate_error": candidate_check.get("reason") if candidate_check["status"] == "INVALID" else None,
        "limitation": ("candidate comparison does not select a tokenizer and is not a formal decision; "
                       "effective Train supervision remains null until a human decision and a verified "
                       "tokenized Stage exist"),
    }
    if selection is not None:
        result["selection_file"] = {
            "path": _relative(dataset / "tokenizer-selection.json", base),
            "sha256": file_info(dataset / "tokenizer-selection.json")["sha256"]}
        if selection_verified:
            result["selection_file"].update({
                "candidate": selection.get("name"),
                "fingerprint": selection.get("fingerprint"),
                "mode": selection.get("mode")})
        elif invalid:
            result["selection_file"]["error"] = "formal tokenizer decision verification failed"
    if invalid or evaluation_status != "COMPARED":
        return result
    if candidates is not None and candidate_check["status"] != "INVALID":
        for candidate in candidates.get("candidates", []):
            metadata = candidate.get("metadata", {})
            result["candidates"].append({
                "name": candidate.get("name"),
                "vocab_size": candidate.get("vocab_size"),
                "fingerprint": candidate.get("fingerprint"),
                "train_split_sha256": metadata.get("train_split_sha256"),
                "split_manifest_sha256": metadata.get("split_manifest_sha256"),
                "training_corpus_sha256": metadata.get("training_corpus_sha256"),
                "selected_documents": metadata.get("training_selection", {}).get("selected_documents"),
                "selected_chars": metadata.get("training_selection", {}).get("selected_chars"),
                "demo_only": metadata.get("demo_only"),
            })
    comparison = evaluation.get("comparison", {})
    for candidate in result["candidates"]:
        row = comparison.get(candidate["name"])
        if row is None:
            continue
        candidate.update({
            "zh_characters_per_token": row.get("zh_characters_per_token"),
            "zh_tokens_per_character": row.get("zh_tokens_per_character"),
            "en_tokens_per_word": row.get("en_tokens_per_word"),
            "code_characters_per_token": row.get("code_characters_per_token"),
            "roundtrip_ok": row.get("roundtrip_ok"),
            "document_boundary_ok": row.get("document_boundary_ok"),
            "relative_to_baseline": row.get("relative_to_baseline"),
        })
        metrics = evaluation.get("candidates", {}).get(candidate["name"]) or {}
        candidate["eval_categories"] = {
            category: {key: values.get(key) for key in ("documents", "characters", "tokens",
                                                        "characters_per_token")}
            for category, values in (metrics.get("categories") or {}).items()}
        result["fixed_eval_categories"] = sorted((metrics.get("categories") or {}).keys())
    baseline = evaluation.get("baseline")
    if baseline:
        result["baseline"] = {"status": baseline.get("status"),
                              "sha256": baseline.get("sha256"),
                              "vocab_size": baseline.get("vocab_size")}
    result["recommendation"] = evaluation.get("recommendation")
    budget = evaluation.get("parameter_budget") or {}
    result["parameter_budget"] = {
        "architecture_unchanged": budget.get("architecture_unchanged"),
        "baseline_total_parameters": budget.get("baseline_total_parameters"),
        "vocab_sizes": {
            vocab: {key: row.get(key) for key in (
                "total_parameters", "delta_parameters_vs_baseline", "delta_fraction_vs_baseline",
                "int4_tensor_payload_bytes_estimated", "int4_under_50m_tensor_payload",
                "conservative_fits_t1200_budget")}
            for vocab, row in (budget.get("vocab_sizes") or {}).items()},
    }
    return result


class _Accounting:
    def __init__(self):
        self.checks = []

    def check(self, name, expected, actual, explanation):
        if expected is None or actual is None:
            return
        self.checks.append({"check": name, "expected": expected, "actual": actual,
                            "status": "PASS" if expected == actual else "FAIL",
                            "explanation": explanation})

    def summary(self):
        failures = [row["check"] for row in self.checks if row["status"] == "FAIL"]
        return {"status": "FAIL" if failures else "PASS",
                "checks": self.checks, "failed_checks": failures}


def _sum(values):
    values = [value for value in values if value is not None]
    return sum(values) if values else None


def _display_name(path, dataset):
    """Keep evidence readable without copying absolute workstation paths."""
    resolved = Path(path).resolve()
    try:
        return str(resolved.relative_to(dataset.resolve()))
    except ValueError:
        return "/".join(resolved.parts[-3:])


def _plan_identity(dataset):
    path = dataset / ".v4" / "plan.json"
    if not path.is_file():
        return None
    plan = json.loads(path.read_text(encoding="utf-8"))
    inputs = [{"name": Path(item["path"]).name, "source_id": item.get("source_id"),
               "sha256": item.get("sha256"), "bytes": item.get("bytes")}
              for item in plan.get("inputs", [])]
    return {"plan_key": plan.get("plan_key"),
            "implementation_sha256": plan.get("implementation_sha256"),
            "config_sha256": plan.get("config_sha256"),
            "policies": plan.get("policies"), "inputs": inputs}


def _stage_identity(dataset):
    stages = {}
    for name in ("ingested", "quality", "dedup", "split", "tokenizers", "tokenizer_eval"):
        path = dataset / ".v4" / f"{name}.stage.json"
        if not path.is_file():
            stages[name] = None
            continue
        stage = json.loads(path.read_text(encoding="utf-8"))
        outputs = {}
        for filename, info in sorted((stage.get("files") or {}).items()):
            outputs[_display_name(filename, dataset)] = {
                "sha256": info.get("sha256"), "bytes": info.get("bytes")}
        stages[name] = {"key": stage.get("key"), "outputs": outputs}
    return stages


def _measurement(readiness, dataset):
    """Only the readiness-verified tokenized Stage may publish measured tokens."""
    evidence = readiness.get("tokenization_evidence") or {}
    verified = (evidence.get("status") == "VERIFIED"
                and evidence.get("measured") is True
                and readiness.get("tokenizer_status") == "HUMAN_DECISION_VERIFIED"
                and readiness.get("effective_train_tokens") is not None)
    if verified:
        return evidence, "MEASURED_VERIFIED"
    unverified_artifact = ((Path(dataset) / "train.token-index.jsonl").is_file()
                           or (Path(dataset) / ".v4/tokenized.stage.json").is_file())
    if evidence.get("status") == "INVALID" or unverified_artifact:
        return evidence, "TOKENIZED_EVIDENCE_INVALID"
    if evidence.get("status") == "SELECTED_NOT_TOKENIZED":
        return evidence, "NOT_MEASURED_TOKENIZED_STAGE_NOT_RUN"
    return evidence, "NOT_MEASURED_TOKENIZER_NOT_FINALIZED"


def _math_supply_status(admitted, split_assessed, estimated, measured, target):
    if not target:
        return "NOT_REQUIRED"
    if not admitted:
        return "MATH_CODE_SOURCE_REQUIRED"
    if not split_assessed or estimated is None:
        return "MATH_CODE_SUPPLY_NOT_ASSESSED"
    if measured is not None:
        return "MATH_CODE_SUPPLY_VERIFIED" if measured >= target else "MATH_CODE_MEASURED_SHORTFALL"
    if estimated < target:
        return "MATH_CODE_ESTIMATED_SHORTFALL"
    return "MATH_CODE_MEASUREMENT_REQUIRED"


def _pipeline_status(readiness, evidence, dataset, blockers=()):
    blockers = set(blockers or readiness.get("blockers", []))
    stage_error = "STAGE_EVIDENCE_INVALID" in blockers
    unverified_artifact = ((Path(dataset) / "train.token-index.jsonl").is_file()
                           or (Path(dataset) / ".v4/tokenized.stage.json").is_file())         and evidence.get("status") != "VERIFIED"
    if evidence.get("status") == "VERIFIED":
        selection = "VERIFIED"
        tokenized = "VERIFIED"
    elif evidence.get("status") == "SELECTED_NOT_TOKENIZED":
        selection = "VERIFIED"
        tokenized = "NOT_RUN"
    elif evidence.get("status") == "INVALID" or stage_error or unverified_artifact:
        selection = "INVALID"
        tokenized = "INVALID"
    elif "HUMAN_TOKENIZER_DECISION_REQUIRED" in blockers:
        selection = "BLOCKED"
        tokenized = "BLOCKED"
    else:
        selection = "NOT_RUN"
        tokenized = "BLOCKED"
    budget_artifact = _stage_artifact(dataset, "budget", "budget-report.json")
    if tokenized == "VERIFIED" and readiness.get("budget_gap", {}).get("measured_category_gap_total") is not None:
        budget = ("VERIFIED" if budget_artifact["status"] == "VERIFIED" else
                  "INVALID" if budget_artifact["status"] == "INVALID" else "NOT_RUN")
    elif tokenized in ("INVALID", "BLOCKED"):
        budget = tokenized
    else:
        budget = "NOT_RUN"
    contamination = readiness.get("contamination_status", "NOT_RUN")
    if contamination not in ("CHECKED_NO_HITS", "REVIEWED"):
        contamination = "INVALID" if stage_error else contamination
    frozen = (Path(dataset) / "FROZEN").is_file()
    if tokenized == "INVALID" and frozen:
        freeze, audit = "INVALID", "INVALID"
    elif readiness.get("status") == "FROZEN_VERIFIED":
        freeze, audit = "VERIFIED", "VERIFIED"
    elif frozen:
        freeze = "INVALID"
        audit = "INVALID" if readiness.get("frozen_integrity_status") == "FAIL" else "NOT_RUN"
    else:
        freeze = "NOT_RUN"
        audit = "INVALID" if stage_error else "NOT_RUN"
        from .v4_language_pilot import AUDIT_RECEIPT, LANGUAGE_VERSION, verify_audit_receipt

        if readiness.get("corpus_version") == LANGUAGE_VERSION and (Path(dataset) / AUDIT_RECEIPT).is_file():
            try:
                audited = verify_audit_receipt(Path(dataset), _read(Path(dataset) / ".v4/plan.json"))
                audit = ("VERIFIED" if audited["status"] == "READY" else "BLOCKED") if tokenized == "VERIFIED" else "INVALID"
            except (CorpusError, OSError, ValueError, KeyError, TypeError):
                audit = "INVALID"
    return {"tokenizer_selection": selection, "tokenization": tokenized,
            "contamination": contamination, "budget": budget,
            "audit": audit, "freeze": freeze}


def analyze_supply(config, base, dataset, *, sample_manifests=(), run_id=None):
    """Return a deterministic, sanitized supply report for one built dataset."""
    base, dataset = Path(base).resolve(), Path(dataset).resolve()
    readiness = pilot_readiness(config, base, dataset, sample_manifests=sample_manifests)
    target = config["targets"]["effective_train_tokens"]
    seq_len = config["targets"]["seq_len"]
    chars_per_token = config["budget"]["chars_per_token"]
    quality_verified = readiness.get("quality_status") == "VERIFIED"
    dedup_verified = readiness.get("dedup_status") == "VERIFIED"
    split_pass = readiness.get("split_status") == "PASS"

    ingest_stats = _stage_stats(dataset, "ingested") if quality_verified else None
    ingest_by_source, ingest_seen_by_source = _ingest_by_source(ingest_stats)
    quality_report = _maybe_read(dataset / "quality-report.json") if quality_verified else None
    dedup_report = _maybe_read(dataset / "dedup-report.json") if dedup_verified else None
    split_manifest = _maybe_read(dataset / "split-manifest.json") if split_pass else None
    budget_report = _maybe_read(dataset / "budget-report.json")
    split_by_source, split_totals = _split_documents(dataset) if split_pass else (None, None)
    train_by_source, train_by_category = _train_rows(dataset, chars_per_token) if split_pass else (None, None)
    split_estimates = _split_estimates(dataset, chars_per_token) if split_pass else None
    dedup_source, dedup_exact_near = _dedup_by_source(dataset) if dedup_verified else (None, None)
    measured_evidence, measurement_status = _measurement(readiness, dataset)
    measured_category = (measured_evidence.get("by_category")
                         if measurement_status == "MEASURED_VERIFIED" else None)
    measured_source = measured_evidence if measurement_status == "MEASURED_VERIFIED" else None

    sources = {}
    for source_id, original in readiness["source_status"].items():
        row = _canonical_source_row(original)
        row["ingest_success_documents"] = None if ingest_by_source is None else ingest_by_source.get(source_id, 0)
        row["ingest_records_seen"] = None if ingest_seen_by_source is None else ingest_seen_by_source.get(source_id, 0)
        quality = original.get("quality") if quality_verified else None
        row["quality"] = None if quality is None else {
            "documents": quality["documents"], "keep": quality["kept"],
            "reject": quality["rejected"], "review": quality["review"],
            "keep_ratio": quality["kept"] / max(quality["documents"], 1),
            "reject_ratio": quality.get("reject_ratio", quality["rejected"] / max(quality["documents"], 1)),
            "review_ratio": quality.get("review_ratio", quality["review"] / max(quality["documents"], 1)),
            "top_reasons": quality.get("top_reasons", [])}
        dedup = original.get("dedup") if dedup_verified else None
        exact_near = (dedup_source or {}).get(source_id, {"exact": 0, "near": 0})
        row["dedup"] = None if dedup is None else {
            "deleted": dedup["dedup_deleted"], "exact_deleted": exact_near["exact"],
            "near_deleted": exact_near["near"], "kept_documents": dedup["kept_documents"],
            "kept_characters": dedup["kept_characters"]}
        row["split"] = None if split_by_source is None else {
            name: split_by_source[name].get(source_id, 0) for name in SPLITS}
        train = (train_by_source or {}).get(source_id)
        row["train"] = None if train is None else {
            **train,
            "effective_train_tokens": None if measured_source is None else
            measured_source.get("by_source", {}).get(source_id, 0),
            "effective_basis": ("exact packed supervision" if measured_source is not None
                                else measurement_status)}
        sources[source_id] = row

    categories = {}
    for category in V4_CATEGORIES:
        coverage = readiness["category_coverage"][category]
        estimated = coverage.get("estimated_train_tokens") if split_pass else None
        effective = coverage.get("effective_train_tokens") if measured_category is not None else None
        target_tokens = round(target * config["targets"]["category_token_ratios"][category])
        quality = None
        if quality_verified:
            quality = {"documents": coverage["documents"], "keep": coverage["kept"],
                       "reject": coverage["rejected"], "review": coverage["review"],
                       "top_reasons": coverage.get("top_reasons", [])}
        dedup = None if not dedup_verified else {
            "deleted": coverage.get("dedup_deleted", 0),
            "kept_documents": coverage.get("kept_documents", 0),
            "kept_characters": coverage.get("kept_characters", 0)}
        split_by_category = None
        if split_manifest is not None:
            split_by_category = {
                name: split_manifest["splits"][name]["by_category"].get(category, 0)
                for name in SPLITS}
        train_category = (train_by_category or {}).get(category)
        categories[category] = {
            "target_ratio": config["targets"]["category_token_ratios"][category],
            "target_effective_train_tokens": target_tokens,
            "approved_source_documents": coverage.get("admitted_documents"),
            "available_source_documents": coverage.get("available_documents"),
            "ingest_success_documents": None if ingest_by_source is None else sum(
                row["ingest_success_documents"] for row in sources.values()
                if row["category"] == category and row["ingest_success_documents"] is not None),
            "quality": quality,
            "dedup": dedup,
            "split": split_by_category,
            "train": None if train_category is None else {
                **train_category,
                "effective_train_tokens": None if effective is None else effective,
                "effective_basis": ("exact packed supervision" if effective is not None
                                    else measurement_status),
            },
            "estimated_train_tokens": estimated,
            "estimated_gap_tokens": None if estimated is None else max(0, target_tokens - estimated),
            "estimated_coverage": None if estimated is None or not target_tokens else estimated / target_tokens,
            "effective_train_tokens": effective,
            "effective_gap_tokens": None if effective is None else max(0, target_tokens - effective),
            "effective_coverage": None if effective is None or not target_tokens else effective / target_tokens,
            "measurement_status": measurement_status if target_tokens else "NOT_REQUIRED",
        }

    accounting = _Accounting()
    approved_source_total = _sum(row["approved_input_documents"] for row in sources.values())
    approved_category_total = _sum(row["approved_source_documents"] for row in categories.values())
    accounting.check("source/category approved input documents", approved_source_total,
                     approved_category_total, "approved inputs are counted once")
    category_target_sum = _sum(row["target_effective_train_tokens"] for row in categories.values())
    accounting.check("category targets equal Pilot target", target, category_target_sum,
                     "configured category targets sum to the Pilot target")

    if ingest_stats is not None:
        published_ingest = ingest_stats.get("documents")
        accounting.check("ingest total vs source rows", published_ingest,
                         _sum(row["ingest_success_documents"] for row in sources.values()),
                         "every ingested document has one source")
        accounting.check("ingest total vs category rows", published_ingest,
                         _sum(row["ingest_success_documents"] for row in categories.values()),
                         "every ingested document has one category")
    if quality_report is not None:
        totals = quality_report["totals"]
        accounting.check("quality documents vs source rows", totals["documents"],
                         _sum(row["quality"]["documents"] for row in sources.values() if row["quality"]),
                         "quality totals are source sums")
        for field in ("keep", "reject", "review"):
            key = {"keep": "kept", "reject": "rejected", "review": "review"}[field]
            accounting.check(f"quality {field} vs source rows", totals[key],
                             _sum(row["quality"][field] for row in sources.values() if row["quality"]),
                             "quality actions are not double counted")
            accounting.check(f"quality {field} vs category rows", totals[key],
                             _sum(row["quality"][field] for row in categories.values() if row["quality"]),
                             "category quality is an aggregation")
    if dedup_report is not None:
        accounting.check("dedup deleted vs source rows", dedup_report["documents"]["deleted"],
                         _sum(row["dedup"]["deleted"] for row in sources.values() if row["dedup"]),
                         "each duplicate is deleted once")
        accounting.check("dedup deleted vs category rows", dedup_report["documents"]["deleted"],
                         _sum(row["dedup"]["deleted"] for row in categories.values() if row["dedup"]),
                         "category dedup is an aggregation")
        if dedup_exact_near is not None:
            accounting.check("dedup exact total", dedup_report["exact"]["duplicates"],
                             dedup_exact_near["exact"], "exact duplicates once")
            accounting.check("dedup near total", dedup_report["near"]["duplicates"],
                             dedup_exact_near["near"], "near duplicates once")
        accounting.check("dedup kept + deleted == quality keep", quality_report["totals"]["kept"] if quality_report else None,
                         dedup_report["documents"]["kept"] + dedup_report["documents"]["deleted"],
                         "dedup deletes only KEEP documents")
    if split_manifest is not None:
        published_split = {name: split_manifest["splits"][name]["documents"] for name in SPLITS}
        for name in SPLITS:
            accounting.check(f"split {name} documents vs source rows", published_split[name],
                             _sum(row["split"][name] for row in sources.values() if row["split"]),
                             "each split document has one source")
            accounting.check(f"split {name} documents vs category rows", published_split[name],
                             _sum(row["split"][name] for row in categories.values() if row["split"]),
                             "each split document has one category")
            accounting.check(f"split {name} total vs assignment", published_split[name],
                             (split_totals or {}).get(name), "assignment and manifest agree")
            if split_estimates is not None:
                accounting.check(f"split {name} assignment characters",
                                 split_manifest["splits"][name]["characters"],
                                 split_estimates[name]["characters"],
                                 "split text files preserve manifest character counts")
        train_chars = split_manifest["splits"]["train"]["characters"]
        accounting.check("Train characters vs source rows", train_chars,
                         _sum(row["train"]["characters"] for row in sources.values() if row["train"]),
                         "Train text is counted once per source")
        accounting.check("Train characters vs category rows", train_chars,
                         _sum(row["train"]["characters"] for row in categories.values() if row["train"]),
                         "Train text is counted once per category")
        accounting.check("Train estimated tokens vs readiness",
                         readiness.get("estimated_train_tokens"),
                         _sum(row["estimated_train_tokens"] for row in categories.values()),
                         "category estimates sum to the readiness total")
        accounting.check("Train estimated tokens vs Train split",
                         readiness.get("estimated_train_tokens"),
                         split_estimates["train"]["estimated_tokens"] if split_estimates else None,
                         "only Train is included in the training budget")
        accounting.check("Val/Test excluded from Train budget",
                         readiness.get("estimated_train_tokens"),
                         split_estimates["train"]["estimated_tokens"] if split_estimates else None,
                         "Val/Test estimates are reported separately and never added")
        accounting.check("Train documents are a subset of dedup KEEP",
                         split_manifest["splits"]["train"]["documents"] <= dedup_report["documents"]["kept"]
                         if dedup_report else None, True,
                         "Train can only contain deduplicated KEEP documents")
    if measured_category is not None:
        measured_total = sum(measured_category.values())
        accounting.check("effective tokens vs readiness", readiness.get("effective_train_tokens"),
                         measured_total, "packed supervision is recomputed from the token index")
        accounting.check("effective tokens vs category rows", measured_total,
                         _sum(row["effective_train_tokens"] for row in categories.values()
                              if row["effective_train_tokens"] is not None),
                         "category effective tokens sum to the total")
        accounting.check("effective tokens vs source rows", measured_total,
                         _sum(row["train"]["effective_train_tokens"] for row in sources.values()
                              if row["train"] and row["train"]["effective_train_tokens"] is not None),
                         "source effective tokens sum to the total")
    if budget_report is not None:
        accounting.check("budget target", target, budget_report.get("target_effective_train_tokens"),
                         "budget uses the configured Pilot target")
        accounting.check("budget measured gap vs readiness",
                         readiness.get("budget_gap", {}).get("measured_category_gap_total"),
                         budget_report.get("total_gap_effective_tokens"),
                         "readiness and budget report the same measured gap")

    consistency = accounting.summary()

    total_estimated = readiness.get("estimated_train_tokens")
    estimated_gaps = [row["estimated_gap_tokens"] for row in categories.values()]
    measured_gaps = [row["effective_gap_tokens"] for row in categories.values()]
    measured_known = all(value is not None for value in measured_gaps)
    estimated_known = all(value is not None for value in estimated_gaps)
    estimated_gap_total = sum(estimated_gaps) if estimated_known else None
    measured_gap_total = sum(measured_gaps) if measured_known else None
    if measured_gap_total is None:
        measured_gap_total = readiness.get("budget_gap", {}).get("measured_category_gap_total")

    blockers = set(readiness.get("blockers", []))
    math_sources = [row for row in sources.values() if row["category"] == "math_logic_code"]
    math_admitted = [row for row in math_sources if row["admitted"]]
    math_category = categories["math_logic_code"]
    math_target = math_category["target_effective_train_tokens"]
    math_supply = math_category["estimated_train_tokens"]
    math_measured = math_category["effective_train_tokens"]
    math_status = _math_supply_status(bool(math_admitted), split_pass, math_supply,
                                      math_measured, math_target)
    math_blockers = {
        "MATH_CODE_SOURCE_REQUIRED": "MATH_CODE_SOURCE_REQUIRED",
        "MATH_CODE_SUPPLY_NOT_ASSESSED": "MATH_CODE_SUPPLY_NOT_ASSESSED",
        "MATH_CODE_ESTIMATED_SHORTFALL": "MATH_CODE_SUPPLY_SHORTFALL",
        "MATH_CODE_MEASURED_SHORTFALL": "MATH_CODE_SUPPLY_SHORTFALL",
        "MATH_CODE_MEASUREMENT_REQUIRED": "MATH_CODE_MEASUREMENT_REQUIRED",
    }
    if math_status in math_blockers:
        blockers.add(math_blockers[math_status])
    tokenizer = _tokenizer_report(dataset, base, readiness, config)
    if measurement_status == "TOKENIZED_EVIDENCE_INVALID" or tokenizer["evidence_status"] == "INVALID":
        blockers.add("TOKENIZED_EVIDENCE_INVALID")
    elif measurement_status == "NOT_MEASURED_TOKENIZED_STAGE_NOT_RUN":
        blockers.add("TOKENIZED_EVIDENCE_REQUIRED")
    if tokenizer["evaluation_status"] == "COMPARED" and tokenizer["selection_status"] == "HUMAN_TOKENIZER_DECISION_REQUIRED":
        blockers.add("HUMAN_TOKENIZER_DECISION_REQUIRED")
    elif tokenizer["evaluation_status"] in (None, "NOT_RUN") and split_pass:
        blockers.add("TOKENIZER_EVALUATION_REQUIRED")
    if consistency["status"] == "FAIL":
        blockers.add("SUPPLY_ACCOUNTING_MISMATCH")

    status = readiness.get("status", "DATA_NOT_READY")
    if consistency["status"] == "FAIL" and status in ("READY_FOR_FREEZE", "FROZEN_VERIFIED"):
        status = "DATA_NOT_READY"
    if measurement_status != "MEASURED_VERIFIED" and status in ("READY_FOR_FREEZE", "FROZEN_VERIFIED"):
        status = "DATA_NOT_READY"
    if tokenizer["evidence_status"] == "INVALID" and status in ("READY_FOR_FREEZE", "FROZEN_VERIFIED"):
        status = "DATA_NOT_READY"
    if status == "READY_FOR_FREEZE" and blockers - {"FINAL_AUDIT_REQUIRED"}:
        status = "DATA_NOT_READY"
    pipeline_status = _pipeline_status(readiness, measured_evidence, dataset, blockers)

    math_section = {
        "status": math_status,
        "target_effective_train_tokens": math_target,
        "estimated_train_tokens": math_supply,
        "measured_train_tokens": math_measured,
        "estimated_gap_tokens": (max(0, math_target - math_supply)
                                 if math_supply is not None else math_target),
        "measured_gap_tokens": (max(0, math_target - math_measured)
                                if math_measured is not None else None),
        "admitted_sources": [row["source_id"] for row in math_admitted],
        "candidate_sources": [{
            "source_id": row["source_id"], "license_status": row["license_status"],
            "file_status": row["file_status"], "admitted": row["admitted"],
            "approved_input_documents": row["approved_input_documents"],
            "missing_evidence": row["missing_evidence"]} for row in math_sources],
        "required_evidence": [
            "publisher and exact upstream release/version",
            "source URI and immutable local file SHA256/size inventory",
            "licence review covering model training, commercial use, redistribution and obligations",
            "deterministic hash-bound JSONL conversion and sample manifest",
            "independent human APPROVED review bound to this Pilot corpus version",
        ],
        "prohibition": ("No math/code document, review decision or licence claim may be invented or "
                        "downloaded without confirmed permission."),
    }

    measured_category_tokens = (readiness.get("effective_train_tokens")
                                if measurement_status == "MEASURED_VERIFIED" else None)
    report = {
        "schema": "v4-pilot-supply/1",
        "run_id": run_id or dataset.name,
        "corpus_version": config["corpus_version"],
        "dataset": {
            "name": dataset.name,
            "status": readiness.get("status"),
            "pilot_preflight_status": readiness.get("pilot_preflight_status"),
            "quality_status": readiness.get("quality_status"),
            "dedup_status": readiness.get("dedup_status"),
            "split_status": readiness.get("split_status"),
            "tokenizer_status": readiness.get("tokenizer_status"),
            "tokenization_evidence_status": measured_evidence.get("status"),
            "dataset_frozen": (dataset / "FROZEN").is_file(),
            "plan": _plan_identity(dataset),
            "stage_identity": _stage_identity(dataset),
        },
        "source_evidence_status": readiness.get("source_evidence_status"),
        "split_isolation": readiness.get("split_isolation"),
        "source_counts": {
            "approved": readiness.get("approved_source_count"),
            "admitted": readiness.get("admitted_source_count"),
            "available": readiness.get("available_source_count"),
        },
        "status": status,
        "blockers": sorted(blockers),
        "train_selection": readiness.get("train_selection"),
        "targets": {
            "effective_train_tokens": target,
            "seq_len": seq_len,
            "category_token_ratios": dict(config["targets"]["category_token_ratios"]),
        },
        "sources": sources,
        "categories": categories,
        "totals": {
            "approved_input_documents": approved_source_total,
            "ingest_success_documents": None if ingest_stats is None else ingest_stats.get("documents"),
            "quality": None if quality_report is None else {
                key: quality_report["totals"][key] for key in ("documents", "kept", "rejected", "review")},
            "dedup": None if dedup_report is None else {
                "kept": dedup_report["documents"]["kept"],
                "deleted": dedup_report["documents"]["deleted"],
                "exact_deleted": dedup_report["exact"]["duplicates"],
                "near_deleted": dedup_report["near"]["duplicates"],
            },
            "split": None if split_manifest is None else {
                name: split_manifest["splits"][name]["documents"] for name in SPLITS},
            "split_estimated_tokens": split_estimates,
            "train_documents": (None if not split_pass else
                                _sum(row["train"]["documents"] for row in sources.values()
                                     if row["train"])),
            "train_characters": None if split_manifest is None else split_manifest["splits"]["train"]["characters"],
            "estimated_train_tokens": total_estimated,
            "estimated_category_gap_total": estimated_gap_total,
            "effective_train_tokens": measured_category_tokens,
            "measured_category_gap_total": measured_gap_total,
            "unverified_target_tokens": readiness.get("budget_gap", {}).get("unverified_target_tokens"),
            "measurement_status": measurement_status,
            "measurement_basis": ("exact packed supervised positions from a verified tokenized Stage"
                                  if measured_category_tokens is not None
                                  else ("NOT_MEASURED: tokenizer not formally selected"
                                        if measurement_status == "NOT_MEASURED_TOKENIZER_NOT_FINALIZED"
                                        else ("TOKENIZED_EVIDENCE_INVALID: formal tokenization verification failed"
                                              if measurement_status == "TOKENIZED_EVIDENCE_INVALID"
                                              else measured_evidence.get("reason", measurement_status)))),
            "val_test_in_train_budget": False,
        },
        "supply_status": {
            "estimated": ("ESTIMATED_SUPPLY_MEETS_TARGET" if estimated_gap_total == 0 else
                          "ESTIMATED_SUPPLY_SHORTFALL" if estimated_gap_total is not None else
                          "NOT_ASSESSED"),
            "measured": ("MEASURED_SUPPLY_MEETS_TARGET" if measured_gap_total == 0 else
                         "MEASURED_SUPPLY_SHORTFALL" if measured_gap_total is not None else
                         "NOT_ASSESSED"),
        },
        "pipeline_status": pipeline_status,
        "tokenizer": tokenizer,
        "math_logic_code": math_section,
        "consistency": consistency,
        "limitations": [
            "Counts describe the locally approved sample inputs, not the complete upstream release.",
            "Estimated tokens use the configured chars-per-token ratio and are never reported as measured.",
            "Val/Test splits are excluded from all Train budget calculations.",
            "No tokenizer was automatically selected and no dataset was frozen by this analysis.",
            "A source review hash proves identity; it does not replace a human licence/training decision.",
        ],
    }
    report["next_actions"] = _next_actions(report)
    binding = {key: report[key] for key in (
        "corpus_version", "status", "targets", "totals", "categories", "sources",
        "pipeline_status", "tokenizer", "math_logic_code", "consistency", "dataset")}
    if report["train_selection"] is not None:
        binding["train_selection"] = report["train_selection"]
    report["content_id"] = identity(binding)
    return report


def _next_actions(report):
    actions = []
    if report["consistency"]["status"] != "PASS":
        actions.append("先修正来源/类别计数不一致，再使用本报告。")
    math = report["math_logic_code"]
    gap = math["estimated_gap_tokens"]
    if math["status"] == "MATH_CODE_SOURCE_REQUIRED":
        actions.append(
            "获取已批准的数学/代码来源，固定发布版本，完成许可审核、转换和样本证据；"
            f"剩余目标为 {gap:,} 个有效 Token。")
    elif math["status"] == "MATH_CODE_SUPPLY_NOT_ASSESSED":
        actions.append("对已准入数学/代码来源执行 Split，再评估供给。")
    elif math["status"] == "MATH_CODE_ESTIMATED_SHORTFALL":
        actions.append(f"数学/代码估算供给不足，缺口 {gap:,} 个有效 Token；请增加已批准数据。")
    elif math["status"] == "MATH_CODE_MEASUREMENT_REQUIRED":
        actions.append("数学/代码估算已达到类别目标，但仍需人工选定 Tokenizer 并完成 tokenize 实测确认。")
    elif math["status"] == "MATH_CODE_MEASURED_SHORTFALL":
        actions.append(f"数学/代码实测供给不足，缺口 {math['measured_gap_tokens']:,} 个有效 Token。")
    tokenizer = report["tokenizer"]
    if tokenizer.get("evidence_status") == "INVALID":
        actions.append("修复 Tokenizer 选择/评估/候选证据；在正式证据恢复前不得进入 Freeze。")
    elif tokenizer.get("evaluation_status") in (None, "NOT_RUN"):
        actions.append("运行现有 Train-only Tokenizer 候选评估（24,576 与 32,768）。")
    elif tokenizer.get("selection_status") == "HUMAN_TOKENIZER_DECISION_REQUIRED":
        actions.append("由人工审核者签署 Tokenizer 选择；不允许自动选择。")
    if report["totals"]["measurement_status"] == "TOKENIZED_EVIDENCE_INVALID":
        actions.append("修复 Tokenized Stage/Token Index 证据；本报告不会发布任何实测 Token。")
    elif report["totals"]["measurement_status"] == "NOT_MEASURED_TOKENIZED_STAGE_NOT_RUN":
        actions.append("人工选择已确认，但 Tokenized Stage 尚未运行；执行 tokenize 后再评估实测供给。")
    elif report["totals"]["effective_train_tokens"] is None:
        actions.append("人工选定 Tokenizer 后，执行 tokenize、contamination、budget、audit，再评估 Freeze。")
    if report["totals"]["estimated_category_gap_total"] not in (None, 0):
        short = [category for category, row in report["categories"].items()
                 if row["estimated_gap_tokens"] not in (None, 0)]
        actions.append("为供给不足类别增加已批准数据：" + ", ".join(short) + "。")
    actions.append("在所有类别达标且最终审计通过前，保持 DATA_NOT_READY。")
    return actions


def render_markdown(report):
    """Render the same facts in a compact reviewable Markdown report."""
    def number(value):
        return "null" if value is None else f"{value:,}"

    def ratio(value):
        return "null" if value is None else f"{value:.1%}"

    def metric(value):
        return "null" if value is None else f"{value:.4f}"

    def scalar(value):
        if value is None:
            return "null"
        if value is True:
            return "true"
        if value is False:
            return "false"
        if isinstance(value, float):
            return f"{value:.4f}"
        return str(value)

    lines = [
        f"# V4 Pilot 供给与 Token 缺口分析（{report['run_id']}）",
        "",
        f"- **结论**：`{report['status']}`",
        f"- **数据集**：`{report['dataset']['name']}` / `{report['corpus_version']}`",
        f"- **来源证据**：`{report['source_evidence_status']}`；已批准 {report['source_counts']['approved']}，"
        f"已准入 {report['source_counts']['admitted']}",
        f"- **Token 目标**：{number(report['targets']['effective_train_tokens'])} 有效监督 Token；"
        f"估算合计 {number(report['totals']['estimated_train_tokens'])}；"
        f"实测合计 {number(report['totals']['effective_train_tokens'])}",
        f"- **估算缺口**：{number(report['totals']['estimated_category_gap_total'])}；"
        f"**实测缺口**：{number(report['totals']['measured_category_gap_total'])}",
        f"- **Blockers**：{', '.join(f'`{item}`' for item in report['blockers']) or '无'}",
        "",
        "## 阶段身份与输入",
        "",
    ]
    plan = report["dataset"].get("plan") or {}
    lines += [
        f"- `plan_key`：`{plan.get('plan_key')}`",
        f"- `implementation_sha256`：`{plan.get('implementation_sha256')}`",
        f"- `config_sha256`：`{plan.get('config_sha256')}`",
        "",
        "| Stage | Key | 输出文件数 |",
        "|---|---|---:|",
    ]
    for stage, row in (report["dataset"].get("stage_identity") or {}).items():
        lines.append(f"| {stage} | `{(row or {}).get('key')}` | {len((row or {}).get('outputs') or {})} |")
    if plan.get("inputs"):
        lines += ["", "| 输入 | Source | 字节 | SHA256 |", "|---|---|---:|---|"]
        for item in plan["inputs"]:
            lines.append(f"| `{item['name']}` | `{item.get('source_id')}` | "
                         f"{number(item.get('bytes'))} | `{item.get('sha256')}` |")
    lines += [
        "",
        "## 来源准入与处理统计",
        "",
        "| 来源 | 类别 | 许可 | 输入文档 | Ingest | KEEP | REJECT | REVIEW | Dedup 删除（exact/near） | Train/Val/Test |",
        "|---|---|---|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in sorted(report["sources"].values(), key=lambda item: item["source_id"]):
        quality = row["quality"] or {}
        dedup = row["dedup"] or {}
        split = row["split"] or {}
        lines.append(
            f"| `{row['source_id']}` | {row['category']} | {row['license_status']} | "
            f"{number(row['approved_input_documents'])} | {number(row['ingest_success_documents'])} | "
            f"{number(quality.get('keep'))} | {number(quality.get('reject'))} | "
            f"{number(quality.get('review'))} | "
            f"{number(dedup.get('deleted'))} ({number(dedup.get('exact_deleted'))}/{number(dedup.get('near_deleted'))}) | "
            f"{number(split.get('train'))}/{number(split.get('val'))}/{number(split.get('test'))} |")
    totals = report["totals"]
    quality = totals["quality"] or {}
    dedup = totals["dedup"] or {}
    split = totals["split"] or {}
    lines += [
        "",
        f"合计：输入 {number(totals['approved_input_documents'])}，Ingest {number(totals['ingest_success_documents'])}，"
        f"KEEP {number(quality.get('kept'))}，REJECT {number(quality.get('rejected'))}，"
        f"REVIEW {number(quality.get('review'))}，Dedup 删除 {number(dedup.get('deleted'))}，"
        f"Split {number(split.get('train'))}/{number(split.get('val'))}/{number(split.get('test'))}。",
        "",
        "## 分类别供给与缺口",
        "",
        "| 类别 | 目标 | 估算 Train Token | 估算缺口 | 估算达成率 | 实测 Token | 实测缺口 | 状态 |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for category, row in report["categories"].items():
        lines.append(
            f"| {category} | {number(row['target_effective_train_tokens'])} | "
            f"{number(row['estimated_train_tokens'])} | {number(row['estimated_gap_tokens'])} | "
            f"{ratio(row['estimated_coverage'])} | {number(row['effective_train_tokens'])} | "
            f"{number(row['effective_gap_tokens'])} | {row['measurement_status']} |")
    selection = report.get("train_selection")
    if selection:
        lines += ["", "### 平衡选样的实测供给与实际 Loss", "",
                  "| 类别 | 全量 Train 编码 Token | 入选编码 Token | 入选文档 | 有效监督 Token | 实际占比 |",
                  "|---|---:|---:|---:|---:|---:|"]
        for category, supply in selection["supply"].items():
            budget = selection["category_budget"][category]
            lines.append(f"| {category} | {number(supply['raw_tokens'])} | "
                         f"{number(selection['selected_raw_tokens_by_category'][category])} | "
                         f"{number(budget['available_documents'])} | {number(budget['available_effective_tokens'])} | "
                         f"{ratio(budget['available_share'])} |")
    lines += [
        "",
        "### Split 与 Train 预算",
        "",
        "| Split | 文档 | 字符 | 估算 Token | 计入 Train 预算 |",
        "|---|---:|---:|---:|---|",
    ]
    for name in ("train", "val", "test"):
        row = (totals["split_estimated_tokens"] or {}).get(name)
        if row is None:
            continue
        lines.append(f"| {name} | {number(row['documents'])} | {number(row['characters'])} | "
                     f"{number(row['estimated_tokens'])} | {scalar(row['included_in_train_budget'])} |")
    lines += [
        "",
        "- Val/Test 不计入 Train 预算（`val_test_in_train_budget=false`）。",
        f"- 估算 Token 不是实测值；实测合计为 `{number(totals['effective_train_tokens'])}`，"
        f"依据 `{totals['measurement_basis']}`。",
        "",
        "## Tokenizer 评估",
        "",
        f"- 评估状态：`{report['tokenizer']['evaluation_status']}`；"
        f"选择状态：`{report['tokenizer']['selection_status']}`；"
        f"证据状态：`{report['tokenizer'].get('evidence_status')}`；"
        f"Tokenization：`{report['tokenizer'].get('tokenization_status')}`。",
    ]
    recommendation = report["tokenizer"].get("recommendation")
    if recommendation:
        lines.append(f"- 建议：`{recommendation.get('status')}` / 低成本候选 "
                     f"`{recommendation.get('preferred_lower_cost')}`；仅为建议，仍需人工签署。")
    if report["tokenizer"]["candidates"]:
        lines += ["", "| 候选 | 词表 | 中文 chars/token | 英文 tokens/word | 代码 chars/token | 参数量 | INT4 < 50M |",
                  "|---|---:|---:|---:|---:|---:|---|"]
        parameter_budget = report["tokenizer"].get("parameter_budget") or {}
        for candidate in report["tokenizer"]["candidates"]:
            budget = (parameter_budget.get("vocab_sizes") or {}).get(str(candidate["vocab_size"]), {})
            lines.append(
                f"| `{candidate['name']}` | {candidate['vocab_size']} | "
                f"{metric(candidate.get('zh_characters_per_token'))} | {metric(candidate.get('en_tokens_per_word'))} | "
                f"{metric(candidate.get('code_characters_per_token'))} | "
                f"{number(budget.get('total_parameters'))} | {scalar(budget.get('int4_under_50m_tensor_payload'))} |")
    lines += [
        "",
        "- 固定评估文本类别：`" + ", ".join(report["tokenizer"].get("fixed_eval_categories") or []) + "`。",
        "",
        "### 流水线阶段状态",
        "",
        "文件存在不等于阶段成功；以下状态来自 Readiness 的正式验证结果。",
        "",
        "| 阶段 | 状态 |",
        "|---|---|",
    ]
    stage_names = (("tokenizer_selection", "Tokenizer 选择"), ("tokenization", "Tokenize"),
                   ("contamination", "Contamination"), ("budget", "Budget"),
                   ("audit", "Audit"), ("freeze", "Freeze"))
    for key, label in stage_names:
        lines.append(f"| {label} | `{report.get('pipeline_status', {}).get(key, 'NOT_RUN')}` |")
    lines += [
        "",
        "## 数学／代码类别",
        "",
        f"- 状态：`{report['math_logic_code']['status']}`；目标 "
        f"{number(report['math_logic_code']['target_effective_train_tokens'])}，估算缺口 "
        f"{number(report['math_logic_code']['estimated_gap_tokens'])}，实测 "
        f"{number(report['math_logic_code'].get('measured_train_tokens'))}，实测缺口 "
        f"{number(report['math_logic_code'].get('measured_gap_tokens'))}。",
    ]
    for row in report["math_logic_code"]["candidate_sources"]:
        lines.append(f"- 候选 `{row['source_id']}`：许可 `{row['license_status']}`，文件 `{row['file_status']}`，"
                     f"已批准输入 {number(row['approved_input_documents'])}。")
    lines += [
        "- 未伪造数学题、代码文档、许可结论或人工审核记录；不从网络自动下载未确认授权的数据。",
        "",
        "## 一致性检查",
        "",
        f"- 状态：`{report['consistency']['status']}`；失败项："
        f"{', '.join(report['consistency']['failed_checks']) or '无'}。",
    ]
    for check in report["consistency"]["checks"]:
        lines.append(f"- `{check['status']}` {check['check']}: {scalar(check['expected'])} == "
                     f"{scalar(check['actual'])}")
    lines += ["", "## 下一步", ""]
    lines.extend(f"- {action}" for action in report["next_actions"])
    lines += ["", "## 限制", ""]
    lines.extend(f"- {item}" for item in report["limitations"])
    lines.append("")
    return "\n".join(lines)
