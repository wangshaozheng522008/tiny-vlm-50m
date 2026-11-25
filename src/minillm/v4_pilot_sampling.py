"""Deterministic family sampling of local V4 JSONL before human review."""
from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
from collections import Counter
from contextlib import closing, contextmanager
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory

from .assessment_io import report_directory
from .corpus_quality import V4_CATEGORIES
from .corpus_utils import CorpusError, file_info, identity, json_bytes, verify_file, write_json
from .data import stream_corpus_records

PILOT_VERSION = "minillm-corpus-v4-pilot-1m-v1"
SAMPLE_MANIFEST = "pilot-sample-manifest.json"
SAMPLE_FILE = "documents.jsonl"
SAMPLE_KIND = "v4_pilot_sample"
ALGORITHM = "sha256-family-basis-points-v1"
_BASIS = 10_000
_MAX_RECORD_BYTES = 8 * 1048576  # V4 quality policy upper bound
_FAMILY_POLICIES = ("document", "file")
_REQUIRED_COUNTS = {"rows_seen", "documents", "characters", "selected_documents",
                    "selected_characters", "selected_families"}
_OPTIONAL_COUNTS = {"record_too_large", "empty_text", "canonical_record_too_large", "duplicate_documents"}


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


@contextmanager
def _selection_db(path):
    with closing(sqlite3.connect(path)) as scratch:
        scratch.execute("PRAGMA temp_store=FILE")
        scratch.execute("PRAGMA cache_size=-8192")
        yield scratch


def _selected_rows(scratch):
    for row, in scratch.execute("SELECT row FROM selected ORDER BY id"):
        yield row


def _sample_limit(seed, basis_points, max_record_bytes):
    return (type(seed) is int and type(basis_points) is int and 1 <= basis_points <= _BASIS
            and type(max_record_bytes) is int and 1 <= max_record_bytes <= _MAX_RECORD_BYTES)


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
    if not _sample_limit(seed, basis_points, max_record_bytes):
        raise CorpusError("sample requires integer seed, basis_points in [1, 10000], and max_record_bytes in [1, 8 MiB]")
    paths = input_paths(paths)
    destination = Path(out)
    if os.path.lexists(destination):
        raise CorpusError("sample output already exists; use a new directory")
    destination = destination.resolve()
    if any(destination == path or destination in path.parents for path in paths):
        raise CorpusError("sample output cannot contain inputs")
    source = entry["v4_source"]
    family_policy = source.get("family_policy", "document")
    if family_policy not in _FAMILY_POLICIES:
        raise CorpusError("invalid sample family policy")
    stamps = {path: _stamp(path) for path in paths}
    inputs = {str(path): file_info(path) for path in paths}
    with report_directory(destination) as temp:
        database = temp / "selection.sqlite"
        with _selection_db(database) as scratch:
            counts = _select(scratch, inputs, source, seed, basis_points, max_record_bytes)
            with (temp / SAMPLE_FILE).open("wb") as output:
                for row in _selected_rows(scratch):
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
        if family_policy != "document":
            manifest["family_policy"] = family_policy
        manifest["sha256"] = identity(manifest)
        write_json(temp / SAMPLE_MANIFEST, manifest)
        draft = _review_draft(entry, destination, manifest, file_info(temp / SAMPLE_MANIFEST))
        write_json(temp / "review-entry.json", draft)
    return manifest


def _file_identity(info):
    return (isinstance(info, dict) and set(info) == {"sha256", "bytes"}
            and type(info["bytes"]) is int and info["bytes"] > 0
            and isinstance(info["sha256"], str) and re.fullmatch(r"[0-9a-f]{64}", info["sha256"]))


def load_sample_manifest(path):
    """Read and validate the manifest fields without accessing its data files."""
    path = Path(path)
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(manifest, dict):
            raise ValueError("expected a manifest object")
        required = {"schema", "algorithm", "seed", "basis_points", "max_record_bytes",
                    "corpus_version", "source_id", "release", "category", "complete_input_scan",
                    "partial_extraction", "inputs", "counts", "output", "sha256"}
        if not required <= manifest.keys() or manifest.keys() - required - {"family_policy"}:
            raise ValueError("unexpected or missing manifest fields")
        if manifest["schema"] != "v4-pilot-sample/1" or manifest["algorithm"] != ALGORITHM:
            raise ValueError("unsupported sample schema or algorithm")
        if not _sample_limit(manifest["seed"], manifest["basis_points"], manifest["max_record_bytes"]):
            raise ValueError("invalid seed, basis_points or max_record_bytes")
        if (not isinstance(manifest["source_id"], str) or not re.fullmatch(r"[A-Za-z0-9._:-]{2,80}", manifest["source_id"])
                or not isinstance(manifest["release"], str) or not manifest["release"].strip()
                or manifest["category"] not in (*V4_CATEGORIES, "auto")
                or manifest["corpus_version"] != PILOT_VERSION
                or manifest.get("family_policy", "document") not in _FAMILY_POLICIES):
            raise ValueError("invalid sample source, release, category, version or family policy")
        if (manifest["complete_input_scan"] is not True
                or manifest["partial_extraction"] is not (manifest["basis_points"] != _BASIS)):
            raise ValueError("invalid input scan or partial extraction status")
        inputs, counts = manifest["inputs"], manifest["counts"]
        if not isinstance(inputs, dict) or not inputs or not _file_identity(manifest["output"]):
            raise ValueError("invalid sample input or output identity")
        for filename, info in inputs.items():
            if not Path(filename).is_absolute() or Path(filename).suffix.lower() != ".jsonl":
                raise ValueError("sample input paths must be absolute JSONL paths")
            if not _file_identity(info):
                raise ValueError("invalid sample input file identity")
        if (not isinstance(counts, dict) or not _REQUIRED_COUNTS <= counts.keys()
                or counts.keys() - _REQUIRED_COUNTS - _OPTIONAL_COUNTS
                or any(type(value) is not int or value < 0 for value in counts.values())
                or not counts["selected_documents"] or not counts["selected_families"]):
            raise ValueError("invalid sample statistics")
        if manifest["sha256"] != identity({k: v for k, v in manifest.items() if k != "sha256"}):
            raise ValueError("sample manifest SHA256 mismatch")
    except (OSError, ValueError, RecursionError) as exc:
        raise CorpusError(f"invalid Pilot sample manifest {path}: {exc}") from exc
    return manifest


def _verify_reproduction(output, manifest, family_policy):
    source = {"source_id": manifest["source_id"], "family_policy": family_policy}
    digest = hashlib.sha256()
    size = 0
    bound = manifest["max_record_bytes"]
    with TemporaryDirectory(prefix="v4-pilot-verify-") as directory:
        with _selection_db(Path(directory) / "selection.sqlite") as scratch:
            counts = _select(scratch, manifest["inputs"], source, manifest["seed"], manifest["basis_points"], bound)
            with output.open("rb") as published:
                for line, expected in enumerate(_selected_rows(scratch), 1):
                    row = published.readline(bound + 1)
                    if not row:
                        raise CorpusError(f"Pilot sample missing selected record {line}: {output}")
                    if row != expected:
                        raise CorpusError(f"Pilot sample record {line} differs from deterministic selection: {output}")
                    digest.update(row)
                    size += len(row)
                if published.read(1):
                    raise CorpusError(f"Pilot sample has extra records after deterministic selection: {output}")
    if {"sha256": digest.hexdigest(), "bytes": size} != manifest["output"]:
        raise CorpusError(f"Pilot sample output SHA256/size mismatch: {output}")
    if counts != manifest["counts"]:
        raise CorpusError(f"Pilot sample statistics differ from deterministic selection: {output}")


def verify_sample(path, *, verify_inputs=True, reproduce=False, family_policy=None):
    """Verify file identities and optionally replay the recorded family selection."""
    path = Path(path)
    manifest = load_sample_manifest(path)
    declared_policy = manifest.get("family_policy", family_policy or "document")
    if (declared_policy not in _FAMILY_POLICIES
            or (family_policy is not None and declared_policy != family_policy)):
        raise CorpusError("Pilot sample family policy differs from the reviewed source")
    if reproduce and not verify_inputs:
        raise CorpusError("Pilot sample reproduction requires input verification")
    output = path.parent / SAMPLE_FILE
    if not output.is_file():
        raise CorpusError(f"file missing or SHA256/size mismatch: {output}")
    output_stamp = _stamp(output)
    stamps = {output: output_stamp}
    if verify_inputs:
        for filename, info in manifest["inputs"].items():
            raw = Path(filename)
            if not raw.is_file():
                raise CorpusError(f"file missing or SHA256/size mismatch: {raw}")
            stamp = _stamp(raw)
            if stamp[:2] == output_stamp[:2]:
                raise CorpusError("Pilot sample output cannot also be an original input")
            stamps[raw] = stamp
            verify_file(raw, info)
    if reproduce:
        _verify_reproduction(output, manifest, declared_policy)
    else:
        verify_file(output, manifest["output"])
    for file, stamp in stamps.items():
        if not file.is_file() or _stamp(file) != stamp:
            raise CorpusError(f"Pilot sample file changed during verification: {file}")
    return manifest


def sample_origin(entry, version, path):
    """A new review may explicitly bind unchanged old sample bytes."""
    from .v4_language_pilot import LANGUAGE_VERSION

    if version == LANGUAGE_VERSION:
        for link in entry["v4_source"].get("transformation_history", []):
            if isinstance(link, dict) and link.get("kind") == SAMPLE_KIND and link.get("manifest") == str(path):
                if link.get("reuse_from_corpus_version") == PILOT_VERSION:
                    return PILOT_VERSION
    return version


def verify_sample_binding(path, manifest, entry, version):
    """Bind one manifest and output to the existing source review entry."""
    path = Path(path)
    source = entry["v4_source"]
    links = [item for item in source.get("transformation_history", [])
             if isinstance(item, dict) and item.get("kind") == SAMPLE_KIND and item.get("manifest") == str(path)]
    if not path.is_absolute() or path.name != SAMPLE_MANIFEST or len(links) != 1:
        raise CorpusError("Pilot sample requires a unique transformation_history manifest binding")
    verify_file(path, {key: links[0][key] for key in ("bytes", "sha256")})
    output = str((path.parent / SAMPLE_FILE).resolve())
    if (entry["local_inventory"]["files"].get(output) != manifest["output"]
            or manifest["source_id"] != source["source_id"] or manifest["release"] != entry["release"]
            or manifest["category"] != source["category"]
            or manifest["corpus_version"] != sample_origin(entry, version, path)):
        raise CorpusError("Pilot sample differs from reviewed scope or file identity")
    return output


def verify_reviewed_samples(entry, version):
    source = entry["v4_source"]
    seen = set()
    for item in source.get("transformation_history", []):
        if not isinstance(item, dict) or item.get("kind") != SAMPLE_KIND:
            continue
        try:
            path = Path(item["manifest"])
            sample = load_sample_manifest(path)
            output = verify_sample_binding(path, sample, entry, version)
            if output in seen:
                raise CorpusError("Pilot sample output has multiple review bindings")
            seen.add(output)
            reproduced = verify_sample(path, reproduce=True, family_policy=source.get("family_policy", "document"))
            if reproduced != sample:
                raise CorpusError("Pilot sample manifest changed during verification")
            verify_file(path, {key: item[key] for key in ("bytes", "sha256")})
        except (OSError, ValueError, KeyError, TypeError, sqlite3.Error) as exc:
            raise CorpusError(f"invalid Pilot sample lineage: {exc}") from exc
