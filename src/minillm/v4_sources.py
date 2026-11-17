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


def load_source_review(path: Path) -> dict:
    review = json.loads(path.read_text(encoding="utf-8"))
    if review.get("schema_version") != 1 or not isinstance(review.get("sources"), list):
        raise CorpusError("invalid source review registry")
    required = {"publisher", "release", "license_references", "commercial_use",
                "redistribution", "limitations", "v4_source"}
    for entry in review["sources"]:
        if not isinstance(entry, dict) or not required <= entry.keys():
            raise CorpusError("source review is missing provenance or licence obligations")
        source = entry["v4_source"]
        if source.get("license_status") == "APPROVED":
            # Source-level approval is insufficient without a pinned release and
            # explicit review of downstream obligations. No defaults grant consent.
            if any(not isinstance(entry.get(key), str) or not entry[key].strip()
                   or entry[key] in ("PENDING", "NOT_PINNED")
                   for key in ("publisher", "release", "commercial_use", "redistribution")):
                raise CorpusError("approved source needs pinned release and reviewed obligations")
            files = entry.get("local_inventory", {}).get("files")
            if not isinstance(files, dict) or not files:
                raise CorpusError("approved source requires reviewed local file SHA256/size evidence")
            for filename, info in files.items():
                if (not isinstance(filename, str) or not Path(filename).is_absolute()
                        or not isinstance(info, dict) or set(info) != {"sha256", "bytes"}
                        or type(info["bytes"]) is not int or info["bytes"] < 0
                        or not isinstance(info["sha256"], str) or len(info["sha256"]) != 64
                        or any(c not in "0123456789abcdef" for c in info["sha256"])):
                    raise CorpusError("invalid reviewed local input file identity")
    from .corpus_v4 import build_registry
    build_registry([entry["v4_source"] for entry in review["sources"]], demo=False,
                   license_policy={"approved_only": True})
    return review


def reviewed_source_specs(path: Path) -> list[dict]:
    """Carry the complete review into the existing plan/registry hash chain."""
    specs = []
    for entry in load_source_review(path)["sources"]:
        spec = copy.deepcopy(entry["v4_source"])
        evidence = {k: copy.deepcopy(v) for k, v in entry.items() if k != "v4_source"}
        spec["transformation_history"] = list(spec.get("transformation_history", [])) + [{
            "kind": "source_review", "entry_sha256": identity(entry), "review": evidence}]
        specs.append(spec)
    return specs


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
