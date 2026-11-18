"""Deterministic family sampling of local V4 JSONL before human review."""
from __future__ import annotations

import json
import os
import sqlite3
from collections import Counter
from contextlib import closing
from copy import deepcopy
from pathlib import Path

from .assessment_io import report_directory
from .corpus_utils import CorpusError, file_info, identity, json_bytes, verify_file, write_json
from .data import stream_corpus_records

PILOT_VERSION = "minillm-corpus-v4-pilot-1m-v1"
SAMPLE_MANIFEST = "pilot-sample-manifest.json"
SAMPLE_FILE = "documents.jsonl"
SAMPLE_KIND = "v4_pilot_sample"
ALGORITHM = "sha256-family-basis-points-v1"
_BASIS = 10_000


def input_paths(paths):
    resolved = sorted({Path(path).resolve() for path in paths})
    if not resolved or any(not path.is_file() for path in resolved):
        raise CorpusError("SOURCE_FILES_MISSING")
    if any(path.suffix.lower() != ".jsonl" for path in resolved):
        raise CorpusError("Pilot sampling requires JSONL; use prepare_v4_sources.py for TXT")
    return resolved


def _stamp(path):
    observed = path.stat()
    return observed.st_dev, observed.st_ino, observed.st_size, observed.st_mtime_ns, observed.st_ctime_ns


def scan_jsonl(path, source_id, max_record_bytes, counts):
    """Validate bounded records while preserving Wiki page and revision fields."""
    with closing(stream_corpus_records(path, max_record_bytes=max_record_bytes,
                                       preserve_json_fields=True)) as records:
        for record in records:
            counts["rows_seen"] += 1
            if "_issue" in record:
                issue = record["_issue"]
                if issue == "record_too_large":
                    counts["record_too_large"] += 1
                    continue
                raise CorpusError(f"{path}:{record['_line']}: {issue}")
            line = record.pop("_line")
            if not record["text"].strip():
                counts["empty_text"] += 1
                continue
            if record.get("source", source_id) != source_id:
                raise CorpusError(f"{path}:{line}: JSONL source differs from {source_id}")
            for field in ("id", "parent_id"):
                value = record.get(field)
                if value is not None and (type(value) not in (str, int) or not str(value).strip()
                                          or len(str(value)) > 512):
                    raise CorpusError(f"{path}:{line}: invalid {field}")
            try:
                record["text"].encode("utf-8")
            except UnicodeEncodeError as exc:
                raise CorpusError(f"{path}:{line}: invalid Unicode text") from exc
            counts["documents"] += 1
            counts["characters"] += len(record["text"])
            yield record


def _select(scratch, inputs, source, seed, basis_points, max_record_bytes):
    threshold = (1 << 256) * basis_points // _BASIS
    counts = Counter()
    scratch.execute("CREATE TABLE selected (id TEXT PRIMARY KEY, family TEXT NOT NULL, row BLOB NOT NULL)")
    for filename, info in inputs.items():
        path = Path(filename)
        before = counts["documents"]
        for record in scan_jsonl(path, source["source_id"], max_record_bytes, counts):
            page_id = record.get("id")
            parent = record.get("parent_id")
            if page_id is None:
                page_id = identity([source["source_id"], parent, record["text"]])
                record["id"] = page_id
            if parent is None:
                parent = "file:" + info["sha256"] if source.get("family_policy") == "file" else page_id
                record["parent_id"] = parent
            family = str(parent)
            rank = int(identity([ALGORITHM, seed, source["source_id"], family]), 16)
            if rank >= threshold:
                continue
            record.setdefault("source", source["source_id"])
            try:
                row = json_bytes(record)
            except (UnicodeError, ValueError, RecursionError) as exc:
                raise CorpusError(f"{path}: invalid JSONL metadata for document {page_id}") from exc
            if len(row) > max_record_bytes:
                counts["canonical_record_too_large"] += 1
                continue
            if scratch.execute("INSERT OR IGNORE INTO selected VALUES (?, ?, ?)",
                               (str(page_id), family, row)).rowcount:
                counts["selected_documents"] += 1
                counts["selected_characters"] += len(record["text"])
            else:
                previous, = scratch.execute("SELECT row FROM selected WHERE id=?", (str(page_id),)).fetchone()
                if previous != row:
                    raise CorpusError(f"conflicting selected records for document ID {page_id}")
                counts["duplicate_documents"] += 1
        if counts["documents"] == before:
            raise CorpusError(f"source file has no readable nonempty records: {path}")
    if not counts["selected_documents"]:
        raise CorpusError(f"no selected documents for {source['source_id']} ({dict(counts)})")
    counts["selected_families"], = scratch.execute("SELECT COUNT(DISTINCT family) FROM selected").fetchone()
    return dict(counts)


def _review_draft(entry, out, manifest, manifest_info):
    draft = deepcopy(entry)
    source = draft["v4_source"]
    source.update(paths=[str(out / SAMPLE_FILE)], license_status="REVIEW_REQUIRED",
                  license_evidence="", license_reviewer="", license_reviewed_at="")
    history = [item for item in source.get("transformation_history", [])
               if not (isinstance(item, dict) and item.get("kind") == SAMPLE_KIND)]
    source["transformation_history"] = history + [
        {"kind": SAMPLE_KIND, "manifest": str(out / SAMPLE_MANIFEST), **manifest_info}]
    draft["local_inventory"] = {
        "files": {str(out / SAMPLE_FILE): manifest["output"]},
        "documents": manifest["counts"]["selected_documents"],
        "characters": manifest["counts"]["selected_characters"],
        "estimated_tokens": None,
    }
    draft["corpus_versions"] = [PILOT_VERSION]
    return draft


def prepare_sample(paths, out, *, entry, seed, basis_points, max_record_bytes=1048576):
    """Publish one immutable sample and a draft entry for the existing registry."""
    if type(seed) is not int or type(basis_points) is not int or not 1 <= basis_points <= _BASIS:
        raise CorpusError("sample requires integer seed and basis_points in [1, 10000]")
    paths = input_paths(paths)
    destination = Path(out)
    if os.path.lexists(destination):
        raise CorpusError("sample output already exists; use a new directory")
    destination = destination.resolve()
    if any(destination == path or destination in path.parents for path in paths):
        raise CorpusError("sample output cannot contain inputs")
    source = entry["v4_source"]
    stamps = {path: _stamp(path) for path in paths}
    inputs = {str(path): file_info(path) for path in paths}
    with report_directory(destination) as temp:
        database = temp / "selection.sqlite"
        with closing(sqlite3.connect(database)) as scratch:
            scratch.execute("PRAGMA temp_store=FILE")
            scratch.execute("PRAGMA cache_size=-8192")
            counts = _select(scratch, inputs, source, seed, basis_points, max_record_bytes)
            with (temp / SAMPLE_FILE).open("wb") as output:
                for row, in scratch.execute("SELECT row FROM selected ORDER BY id"):
                    output.write(row)
        database.unlink()
        for path in paths:
            verify_file(path, inputs[str(path)])
            if _stamp(path) != stamps[path]:
                raise CorpusError(f"sample input changed: {path}")
        manifest = {
            "schema": "v4-pilot-sample/1", "algorithm": ALGORITHM, "seed": seed,
            "basis_points": basis_points, "max_record_bytes": max_record_bytes,
            "corpus_version": PILOT_VERSION, "source_id": source["source_id"],
            "release": entry["release"], "category": source["category"],
            "complete_input_scan": True, "partial_extraction": basis_points != _BASIS,
            "inputs": inputs, "counts": counts, "output": file_info(temp / SAMPLE_FILE),
        }
        manifest["sha256"] = identity(manifest)
        write_json(temp / SAMPLE_MANIFEST, manifest)
        draft = _review_draft(entry, destination, manifest, file_info(temp / SAMPLE_MANIFEST))
        write_json(temp / "review-entry.json", draft)
    return manifest


def verify_sample(path, *, verify_inputs=True):
    path = Path(path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if (manifest.get("schema") != "v4-pilot-sample/1" or manifest.get("algorithm") != ALGORITHM
            or manifest.get("complete_input_scan") is not True
            or manifest.get("sha256") != identity({k: v for k, v in manifest.items() if k != "sha256"})
            or not manifest.get("inputs")):
        raise CorpusError(f"invalid Pilot sample manifest: {path}")
    verify_file(path.parent / SAMPLE_FILE, manifest["output"])
    if verify_inputs:
        for filename, info in manifest["inputs"].items():
            if not Path(filename).is_absolute():
                raise CorpusError("Pilot sample input path must be absolute")
            verify_file(Path(filename), info)
    return manifest


def verify_reviewed_samples(entry, version):
    source = entry["v4_source"]
    reviewed_files = entry["local_inventory"]["files"]
    seen = set()
    for item in source.get("transformation_history", []):
        if not isinstance(item, dict) or item.get("kind") != SAMPLE_KIND:
            continue
        try:
            path = Path(item["manifest"])
            if not path.is_absolute() or path.name != SAMPLE_MANIFEST:
                raise CorpusError("invalid Pilot sample manifest path")
            verify_file(path, {key: item[key] for key in ("bytes", "sha256")})
            sample = verify_sample(path)
            output = str((path.parent / SAMPLE_FILE).resolve())
            if (output in seen or reviewed_files.get(output) != sample["output"]
                    or sample["source_id"] != source["source_id"] or sample["release"] != entry["release"]
                    or sample["category"] != source["category"] or sample["corpus_version"] != version):
                raise CorpusError("Pilot sample differs from reviewed scope or file identity")
            seen.add(output)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise CorpusError(f"invalid Pilot sample lineage: {exc}") from exc
