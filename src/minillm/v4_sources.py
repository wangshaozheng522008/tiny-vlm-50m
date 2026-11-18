"""Local source staging and a single human-maintained V4 approval registry.

Staging does not grant permission to ingest. Originals are read-only, malformed
records are accounted for, and all quality decisions remain in corpus_v4.
"""
from __future__ import annotations

import json
import copy
import resource
import sqlite3
import time
from pathlib import Path

from .assessment_io import report_directory
from .corpus_utils import CorpusError, file_info, identity, json_bytes, write_json
from .data import stream_corpus_records


# Production profiles supervise at least the independently approved Pilot goal.
# Smaller legacy/CPU profiles cannot satisfy PR9's pinned production policy.
MIN_FORMAL_EFFECTIVE_TOKENS = 1_020_000
PLACEHOLDERS = {"PENDING", "NOT_PINNED", "UNKNOWN", "UNVERIFIED", "TBD", "TODO", "N/A",
                "NONE", "NULL", "LICENSE_UNVERIFIED", "NOT_REVIEWED", "REVIEW_REQUIRED"}


def _review_text(value, field):
    if not isinstance(value, str) or not value.strip() or value.strip().upper() in PLACEHOLDERS:
        raise CorpusError(f"approved source requires explicit reviewed {field}, not a placeholder")
    return value.strip()


def _file_identity(info):
    return (isinstance(info, dict) and set(info) == {"sha256", "bytes"}
            and type(info["bytes"]) is int and info["bytes"] >= 0
            and isinstance(info["sha256"], str) and len(info["sha256"]) == 64
            and all(c in "0123456789abcdef" for c in info["sha256"]))


def _license_reference(reference):
    from urllib.parse import urlsplit, unquote
    from .corpus_utils import sha256
    if not isinstance(reference, dict):
        raise CorpusError("license_references entries must be objects")
    uri = _review_text(reference.get("uri"), "license reference uri")
    _review_text(reference.get("verification"), "license reference verification")
    try:
        location = urlsplit(uri)
    except ValueError as exc:
        raise CorpusError("invalid license reference URL") from exc
    if location.scheme in ("https", "http"):
        if not location.hostname or any(c.isspace() for c in uri):
            raise CorpusError("invalid license reference URL")
        return
    if location.scheme == "file" and location.netloc in ("", "localhost") and not location.query and not location.fragment:
        path = Path(unquote(location.path))
    elif not location.scheme:
        path = Path(uri)
    else:
        raise CorpusError("license reference requires HTTP(S) or an absolute local evidence path")
    digest = reference.get("sha256")
    if (not path.is_absolute() or not path.is_file() or not _file_identity({"sha256": digest, "bytes": 0})
            or sha256(path) != digest):
        raise CorpusError("local license evidence missing or SHA256 mismatch")


def _validate_review_entry(entry):
    """Validate original saved values; normalization must never rewrite evidence."""
    if not isinstance(entry, dict) or not isinstance(entry.get("v4_source"), dict):
        raise CorpusError("source review requires a v4_source object")
    source = entry["v4_source"]
    # Incomplete candidates stay readable. Only an actual approval asserts that
    # provenance, input identity and every downstream obligation were reviewed.
    if source.get("license_status") != "APPROVED":
        return
    for field in ("publisher", "release", "commercial_use", "redistribution", "model_training", "license_obligations"):
        _review_text(entry.get(field), field)
    for field in ("source_id", "source_name", "license", "license_evidence", "license_reviewer", "license_reviewed_at"):
        _review_text(source.get(field), field)
    from datetime import datetime
    try:
        datetime.fromisoformat(source["license_reviewed_at"].strip().replace("Z", "+00:00"))
    except ValueError as exc:
        raise CorpusError("approved source requires a real ISO review date") from exc
    if source.get("provenance_status") != "VERIFIED":
        raise CorpusError("approved source requires VERIFIED provenance")
    references = entry.get("license_references")
    if not isinstance(references, list) or not references:
        raise CorpusError("approved source requires nonempty license_references")
    for reference in references:
        _license_reference(reference)
    versions = entry.get("corpus_versions")
    if not isinstance(versions, list) or not versions:
        raise CorpusError("approved source requires explicitly reviewed corpus_versions")
    for version in versions:
        _review_text(version, "corpus_versions")
    if len(set(versions)) != len(versions):
        raise CorpusError("duplicate reviewed corpus_versions")
    inventory = entry.get("local_inventory")
    files = inventory.get("files") if isinstance(inventory, dict) else None
    if not isinstance(files, dict) or not files:
        raise CorpusError("approved source requires reviewed local file SHA256/size evidence")
    for filename, info in files.items():
        if not isinstance(filename, str) or not Path(filename).is_absolute() or not _file_identity(info):
            raise CorpusError("invalid reviewed local input file identity")
    if not isinstance(source.get("transformation_history", []), list):
        raise CorpusError("source transformation_history must be a list")
    if "source_review" in source or any(isinstance(h, dict) and h.get("kind") == "source_review"
                                       for h in source.get("transformation_history", [])):
        raise CorpusError("review entries cannot recursively contain generated source_review evidence")
    from .corpus_v4 import build_registry
    registry = build_registry([source], demo=False, license_policy={"approved_only": True})
    if not registry["sources"] and any(e["reason"] != "disabled" for e in registry["excluded"]):
        raise CorpusError("reviewed source was not admitted by the original V4 licence gate")


def load_source_review(path: Path) -> dict:
    review = json.loads(path.read_text(encoding="utf-8"))
    if review.get("schema_version") != 1 or not isinstance(review.get("sources"), list):
        raise CorpusError("invalid source review registry")
    for entry in review["sources"]:
        _validate_review_entry(entry)
    from .corpus_v4 import build_registry
    build_registry([entry["v4_source"] for entry in review["sources"]], demo=False,
                   license_policy={"approved_only": True})
    return review


def reviewed_source_specs(path: Path, *, corpus_version=None) -> list[dict]:
    """Full evidence stays at source level; document history contains a reference."""
    specs = []
    for entry in load_source_review(path)["sources"]:
        spec = copy.deepcopy(entry["v4_source"])
        record = {"schema_version": 2, "corpus_version": corpus_version,
                  "entry": copy.deepcopy(entry), "entry_sha256": identity(entry)}
        record["sha256"] = identity(record)
        spec["source_review"] = record
        spec["transformation_history"] = list(spec.get("transformation_history", [])) + [review_reference(spec)]
        specs.append(spec)
    return specs


def review_reference(source):
    return {"kind": "source_review", "source_id": source["source_id"],
            "source_review_sha256": source["source_review"]["sha256"]}


def requires_source_review(config):
    target = config["targets"]["effective_train_tokens"]
    if type(target) is not int or target <= 0:
        raise CorpusError("invalid source review scope: effective-token target")
    return target >= MIN_FORMAL_EFFECTIVE_TOKENS


def validate_reviewed_source(source, observed_inputs=None, *, corpus_version=None, verify_inputs=False):
    """One fail-closed validator for build, freeze and subsequent admission.

    A checksum proves identity, not a legal right. Human reviewers remain
    responsible for all licence/training/attribution conclusions in the entry.
    """
    from .corpus_v4 import build_registry
    from .corpus_utils import verify_file
    record = source.get("source_review")
    if not isinstance(record, dict) or type(record.get("schema_version")) is not int or record["schema_version"] != 2:
        raise CorpusError("missing complete source_review evidence")
    entry = record.get("entry")
    _validate_review_entry(entry)
    if entry["v4_source"].get("license_status") != "APPROVED":
        raise CorpusError("source_review has no human APPROVED decision")
    if (record.get("entry_sha256") != identity(entry)
            or record.get("sha256") != identity({k: v for k, v in record.items() if k != "sha256"})):
        raise CorpusError("source_review identity mismatch")
    version = _review_text(record.get("corpus_version"), "source_review corpus_version")
    if version not in entry["corpus_versions"] or (corpus_version is not None and version != corpus_version):
        raise CorpusError("source_review belongs to a different corpus version")
    # Compare canonical source values, not just a marker or a declared SHA.
    actual = copy.deepcopy(source)
    actual.pop("source_review")
    history = actual.get("transformation_history", [])
    reference = review_reference(source)
    if not isinstance(history, list) or history.count(reference) != 1:
        raise CorpusError("source_review history reference mismatch")
    actual["transformation_history"] = [h for h in history if h != reference]
    canonical = lambda spec: build_registry([spec], demo=False, license_policy={"approved_only": True})["sources"]
    if canonical(actual) != canonical(entry["v4_source"]):
        raise CorpusError("source_review source/approval content mismatch")
    expected = entry["local_inventory"]["files"]
    if observed_inputs is not None:
        observed = {}
        for item in observed_inputs:
            if item["source_id"] != source["source_id"]:
                continue
            path = str(Path(item["path"]).resolve())
            if path in observed:
                raise CorpusError("duplicate observed source input")
            observed[path] = {k: item[k] for k in ("sha256", "bytes")}
            if not _file_identity(observed[path]):
                raise CorpusError("invalid observed source input identity")
        if observed != expected:
            raise CorpusError("actual Ingest input set differs from reviewed file SHA256/size evidence")
    if verify_inputs:
        for filename, info in expected.items():
            verify_file(Path(filename), info)
        from .v4_pilot_sampling import verify_reviewed_samples
        verify_reviewed_samples(entry, version)
    return record["sha256"]


def validate_source_inputs(config, registry, inputs, *, require=False, verify_inputs=False):
    required = require or requires_source_review(config) or any("source_review" in s for s in registry["sources"])
    hashes = {}
    if required and not registry["sources"]:
        raise CorpusError("formal V4 data requires reviewed sources")
    for source in registry["sources"]:
        if required or "source_review" in source:
            hashes[source["source_id"]] = validate_reviewed_source(source, inputs,
                corpus_version=config["corpus_version"], verify_inputs=verify_inputs)
    if hashes:
        known = {s["source_id"] for s in registry["sources"]}
        if any(i["source_id"] not in known for i in inputs):
            raise CorpusError("Ingest input has no corresponding reviewed source")
    return hashes


def validate_source_plan(plan, registry, *, require=False, verify_inputs=False):
    config = plan["generation_config"]
    policy = plan.get("source_review_policy")
    if policy is not None:
        if policy != {"schema_version": 2, "corpus_version": config["corpus_version"], "require_complete_reviews": True}:
            raise CorpusError("invalid generation source_review_policy")
        require = True
    if not (require or requires_source_review(config) or any("source_review" in s for s in registry["sources"])):
        return {}
    if (plan["plan_key"] != identity({k: v for k, v in plan.items() if k != "plan_key"})
            or plan["config_sha256"] != identity(config) or plan["registry"] != registry
            or plan["corpus_version"] != config["corpus_version"]
            or registry["sha256"] != identity({k: v for k, v in registry.items() if k != "sha256"})):
        raise CorpusError("source review generation plan/registry identity mismatch")
    # Formal builds must use the very source approvals saved in generation_config.
    if require or requires_source_review(config):
        from .corpus_v4 import build_registry
        if build_registry(config["sources"], demo=False, license_policy=config["license_policy"]) != registry:
            raise CorpusError("Ingest sources differ from reviewed generation configuration")
    return validate_source_inputs(config, registry, plan["inputs"], require=require, verify_inputs=verify_inputs)


def prepare_source(inputs, out, *, source_id, category, language, max_record_bytes=1048576):
    """Canonical JSONL, sorted on disk; never filter valid text for quality.

    Equal source/text/parent records retain their multiplicity. Explicit JSONL
    parents survive conversion; TXT paragraphs remain in one file family.
    JSONL without a supplied parent uses content identity as the document family.
    """
    from .corpus_quality import V4_CATEGORIES
    if category not in V4_CATEGORIES or not isinstance(source_id, str) or not source_id:
        raise CorpusError("source_id and an explicit V4 category are required")
    paths = sorted({Path(path).resolve() for path in inputs})
    if not paths or any(not p.is_file() or p.suffix.lower() not in (".txt", ".jsonl") for p in paths):
        raise CorpusError("inputs must be existing TXT/JSONL files")
    out = Path(out).resolve()
    if any(out == p or out in p.parents for p in paths):
        raise CorpusError("output cannot contain or replace original inputs")
    before = {str(p): file_info(p) for p in paths}
    started = time.monotonic()
    documents = characters = errors = 0
    with report_directory(out) as temp:
        db = sqlite3.connect(temp / "sort.sqlite")
        try:
            db.execute("PRAGMA temp_store=FILE")
            db.execute("PRAGMA cache_size=-16384")
            db.execute("CREATE TABLE records (id TEXT, row BLOB)")
            with (temp / "errors.jsonl").open("wb") as quarantine, (temp / "provenance.jsonl").open("wb") as provenance:
                for path in paths:
                    for record in stream_corpus_records(path, max_record_bytes=max_record_bytes):
                        if "_issue" in record:
                            errors += 1
                            quarantine.write(json_bytes({"input": str(path), "input_sha256": before[str(path)]["sha256"],
                                "line": record["_line"], "issue": record["_issue"]}))
                            continue
                        text = record["text"]
                        # Reject invalid UTF-8 including escaped lone surrogates.
                        try:
                            text.encode("utf-8")
                        except UnicodeEncodeError:
                            errors += 1
                            quarantine.write(json_bytes({"input": str(path), "line": record["_line"],
                                "input_sha256": before[str(path)]["sha256"], "issue": "invalid_utf8"}))
                            continue
                        parent = record.get("parent_id")
                        if parent is not None and (type(parent) not in (str, int) or len(str(parent)) > 512):
                            errors += 1
                            quarantine.write(json_bytes({"input": str(path), "line": record["_line"],
                                "input_sha256": before[str(path)]["sha256"], "issue": "invalid_parent_id"}))
                            continue
                        if parent is None:
                            parent = ("file:" + before[str(path)]["sha256"] if path.suffix.lower() == ".txt"
                                      else "document:" + identity([source_id, text]))
                        stable_id = identity([source_id, text, parent])
                        row = {k: v for k, v in record.items() if not k.startswith("_")}
                        row.update(id=stable_id, parent_id=parent, source=source_id,
                                   category=category, language=language)
                        db.execute("INSERT INTO records VALUES (?,?)", (stable_id, json_bytes(row)))
                        provenance.write(json_bytes({"document_id": stable_id, "input": str(path),
                            "input_sha256": before[str(path)]["sha256"], "line": record["_line"]}))
                        documents += 1
                        characters += len(text)
            db.commit()
            with (temp / "documents.jsonl").open("wb") as handle:
                for row in db.execute("SELECT row FROM records ORDER BY id, row"):
                    handle.write(row[0])
        finally:
            db.close()
        (temp / "sort.sqlite").unlink()
        if before != {str(p): file_info(p) for p in paths}:
            raise CorpusError("source changed during conversion; nothing published")
        files = {name: file_info(temp / name) for name in ("documents.jsonl", "errors.jsonl", "provenance.jsonl")}
        binding = {"schema_version": 1, "converter": "v4-local-canonical-v1",
                   "source_id": source_id, "category": category, "language": language,
                   "max_record_bytes": max_record_bytes, "inputs": before, "files": files}
        report = {**binding, "content_id": identity(binding), "license_status": "NOT_GRANTED_BY_CONVERSION",
                  "documents": documents, "characters": characters, "errors": errors,
                  "estimated_tokens": None, "effective_supervised_tokens": None,
                  "seconds": time.monotonic() - started,
                  "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                  "disk_bytes": sum(item["bytes"] for item in files.values()),
                  "quality_filtering": "NONE; run the approved V4 pipeline"}
        report["documents_per_second"] = documents / max(report["seconds"], 1e-9)
        write_json(temp / "conversion-manifest.json", report)
    return report
