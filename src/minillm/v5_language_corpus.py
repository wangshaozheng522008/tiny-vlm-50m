"""V5 language-corpus inventory, approval gate and release checks.

The V5 profile is deliberately split into two phases:

* **candidate inventory** -- stream local, hash-verified source releases,
  deterministically sample them, run the same quality/dedup/contamination
  primitives as V4 and measure exact supervised-token supply.  No source is
  admitted and nothing is frozen.
* **formal release** -- only after a human approves every source and the
  versioned 10M budget.  This path reuses :class:`corpus_v4.CorpusV4Pipeline`
  and its ingest/quality/dedup/split/tokenize/freeze gates.

Candidate measurements are never labelled trainable.  A missing approval file,
a placeholder reviewer, an unapproved source or an undersized inventory is
reported as a block instead of being inferred as success.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import sqlite3
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import unquote, urlsplit

from .assessment_io import environment, read_json, report_directory
from .corpus_budget import plan as plan_budget
from .corpus_budget import supervision_by_category
from .corpus_contamination import audit as contamination_audit
from .corpus_dedup import (MinHashLSHIndex, NearDupPolicy, chunk_features,
                           chunk_text, exact_jaccard, shingles)
from .corpus_quality import (QUALITY_RULES_VERSION, V4_CATEGORIES, aggregate_action,
                             content_hash, detect_language, evaluate_quality, normalise)
from .corpus_utils import (CorpusError, file_info, identity, json_bytes, sha256,
                           verify_file, write_json)
from .corpus_versions import (V5_CANDIDATE_TARGET, V5_CATEGORY_RATIOS,
                              V5_SEQ_LEN, V5_TARGET, V5_TOLERANCE, V5_UPDATES,
                              V5_VERSION, V5_TOKENS_PER_UPDATE)
from .tokenizer import MiniLLMTokenizer

ROOT = Path(__file__).resolve().parents[2]
CORPUS_CONFIG = ROOT / "configs/corpus_v5_language_10m.yaml"
CANDIDATE_CONFIG = ROOT / "configs/corpus_v5_candidates.json"
HISTORICAL_CONFIG = ROOT / "configs/corpus_v5_historical-eval.json"
SOURCE_REGISTRY = ROOT / "configs/corpus_v5_sources.review.json"
BASE = ROOT / "artifacts/corpus/v4-language-pilot-first-training-reviewed"
BASE_2M = ROOT / "artifacts/corpus/minillm-corpus-v4-language-pilot-2m-v1"
OUTPUT = ROOT / "artifacts/corpus" / V5_VERSION
REPORT = ROOT / "reports/v5-language-corpus-10m-v1"
TOKENIZER = BASE / "tokenizer.json"
BASE_ID = "7222651ef09e4e09c2d613474f50712e8233d1d4ff9017bda897e597ebdb6da5"
TOKENIZER_FINGERPRINT = "845a24b9f21458b1d2161c60dfc8a77d5ed40235ece7ef46891677e423727b8a"
TOKENIZER_VOCAB = 24576

TRAINING_BUDGETS = (
    ("4M", 489, 489 * V5_TOKENS_PER_UPDATE),
    ("10M", V5_UPDATES, V5_TARGET),
    ("50M", 6110, 6110 * V5_TOKENS_PER_UPDATE),
    ("100M", 12219, 12219 * V5_TOKENS_PER_UPDATE),
)
MEASURED_SECONDS_PER_UPDATE = 1531.928 / 245
ALLOWED_LICENSE_REFERENCE_SCHEMES = ("https", "http")


# ---------------------------------------------------------------------------
# small helpers


def _positive_int(value, name):
    if type(value) is not int or value <= 0:
        raise CorpusError(f"{name} must be a positive integer")
    return value


def _safe_rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def _require_text(value, name):
    from .v4_sources import _review_text

    return _review_text(value, name)


def _iso_date(value, name):
    from datetime import date

    try:
        return date.fromisoformat(_require_text(value, name))
    except ValueError as exc:
        raise CorpusError(f"{name} must be an ISO date") from exc


def _check_file_identity(info):
    return (isinstance(info, dict) and set(info) == {"sha256", "bytes"}
            and type(info["bytes"]) is int and info["bytes"] >= 0
            and isinstance(info["sha256"], str) and len(info["sha256"]) == 64
            and all(c in "0123456789abcdef" for c in info["sha256"]))


def _source_files(entry):
    inventory = entry.get("local_inventory")
    files = inventory.get("files") if isinstance(inventory, dict) else None
    if not isinstance(files, dict) or not files:
        raise CorpusError("v5 source requires nonempty local_inventory.files")
    result = {}
    for filename, info in files.items():
        path = Path(filename)
        if not path.is_absolute() or not _check_file_identity(info):
            raise CorpusError("v5 local inventory files require absolute paths and sha256/bytes")
        result[str(path.resolve())] = dict(info)
    return result


def _verify_source_files(entry):
    for filename, info in _source_files(entry).items():
        verify_file(Path(filename), info)
    raw = entry.get("local_inventory", {}).get("raw_files", {})
    if raw is not None:
        if not isinstance(raw, dict):
            raise CorpusError("local_inventory.raw_files must be a mapping")
        for filename, info in raw.items():
            if not Path(filename).is_absolute() or not _check_file_identity(info):
                raise CorpusError("raw_files entries require absolute paths and sha256/bytes")
            verify_file(Path(filename), info)


# ---------------------------------------------------------------------------
# registry / candidate-plan loading


def load_source_review(path: str | Path = SOURCE_REGISTRY, *, verify_inputs=False) -> dict:
    """Load the human-maintained review registry.

    Candidate entries stay readable without approval.  Their local inventory is
    still mandatory so every later decision references bytes that really exist.
    """
    from .v4_sources import _validate_review_entry

    path = Path(path)
    review = read_json(path)
    if review.get("schema_version") != 1 or not isinstance(review.get("sources"), list):
        raise CorpusError("invalid v5 source review registry")
    if not review["sources"]:
        raise CorpusError("v5 source review registry is empty")
    seen = set()
    for entry in review["sources"]:
        if not isinstance(entry, dict) or not isinstance(entry.get("v4_source"), dict):
            raise CorpusError("v5 source review requires a v4_source object")
        source = entry["v4_source"]
        source_id = source.get("source_id")
        if not isinstance(source_id, str) or not source_id or source_id in seen:
            raise CorpusError("v5 source ids must be unique nonempty strings")
        seen.add(source_id)
        if source.get("category") not in V4_CATEGORIES:
            raise CorpusError(f"{source_id}: invalid v5 category")
        if source.get("language") not in ("zh", "en", "mixed"):
            raise CorpusError(f"{source_id}: v5 language must be zh/en/mixed")
        if source.get("license_status") not in ("APPROVED", "REVIEW_REQUIRED", "REJECTED"):
            raise CorpusError(f"{source_id}: invalid licence status")
        _source_files(entry)
        _validate_review_entry(entry)
        if verify_inputs:
            _verify_source_files(entry)
    return review


def approved_entries(review):
    return [entry for entry in review["sources"]
            if entry["v4_source"].get("license_status") == "APPROVED"]


def pending_entries(review):
    return [entry for entry in review["sources"]
            if entry["v4_source"].get("license_status") != "APPROVED"]


def validate_scope(entry):
    """Require a fresh, versioned human scope for v5, not an inherited one."""
    scope = entry.get("training_scope")
    if entry.get("corpus_versions") != [V5_VERSION] or not isinstance(scope, dict):
        raise CorpusError("fresh corpus v5 human training scope required")
    expected = {
        "corpus_version": V5_VERSION,
        "base_content_id": BASE_ID,
        "effective_train_tokens": V5_TARGET,
        "category_token_ratios": V5_CATEGORY_RATIOS,
        "decision": "APPROVED",
    }
    for key, value in expected.items():
        if scope.get(key) != value:
            raise CorpusError(f"v5 human scope {key} differs from the pinned budget")
    if scope.get("public_weight_release") not in (True, False):
        raise CorpusError("v5 scope must state whether public weight release is approved")
    reviewer = _require_text(scope.get("reviewer"), "v5 scope reviewer")
    reviewed = _iso_date(scope.get("reviewed_at"), "v5 scope reviewed_at")
    _require_text(scope.get("confirmation"), "v5 human confirmation")
    source = entry["v4_source"]
    if source.get("license_status") != "APPROVED":
        raise CorpusError("v5 scope requires an APPROVED source licence")
    licensed = _iso_date(source.get("license_reviewed_at"), "v5 source licence review date")
    if reviewer != _require_text(source.get("license_reviewer"), "v5 source licence reviewer"):
        raise CorpusError("v5 scope reviewer differs from the source licence reviewer")
    if reviewed != licensed:
        raise CorpusError("v5 scope date differs from the source licence review date")
    if scope["public_weight_release"] and not scope.get("redistribution_decision"):
        raise CorpusError("public weight release requires an explicit redistribution decision")
    return scope


def validate_source_set(registry):
    """Called by the V4 source gate when the generation version is V5."""
    rows = registry.get("sources", [])
    if not rows:
        raise CorpusError("v5 requires at least one approved source")
    categories = set()
    for source in rows:
        record = source.get("source_review")
        if not isinstance(record, dict):
            raise CorpusError(f"v5 source {source.get('source_id')} lacks source_review evidence")
        entry = record.get("entry")
        validate_scope(entry)
        if entry["v4_source"].get("source_id") != source.get("source_id"):
            raise CorpusError("v5 source_review id mismatch")
        categories.add(source["category"])
    if "math_logic_code" in categories:
        raise CorpusError("v5 must not admit a math/code source")
    required = {c for c, ratio in V5_CATEGORY_RATIOS.items() if ratio}
    missing = required - categories
    if missing:
        raise CorpusError(f"v5 approved sources do not cover categories: {sorted(missing)}")
    return True


def load_candidate_config(path: str | Path = CANDIDATE_CONFIG) -> dict:
    config = read_json(Path(path))
    if config.get("schema_version") != 1 or config.get("corpus_version") != V5_VERSION:
        raise CorpusError("invalid v5 candidate configuration version")
    if config.get("target_candidate_effective_tokens") != V5_CANDIDATE_TARGET:
        raise CorpusError("v5 candidate target must be 15,000,000 effective tokens")
    _positive_int(config.get("seed", 42), "candidate seed")
    _positive_int(config.get("commit_every", 1000), "candidate commit_every")
    sources = config.get("sources")
    if not isinstance(sources, list) or not sources:
        raise CorpusError("v5 candidate configuration requires sources")
    seen = set()
    total = 0
    for source in sources:
        if not isinstance(source, dict):
            raise CorpusError("candidate source entries must be mappings")
        required = {"source_id", "path", "sample_rate", "target_raw_tokens",
                    "max_source_share", "expected_language"}
        if set(source) != required:
            raise CorpusError(f"candidate source fields must be exactly {sorted(required)}")
        if source["source_id"] in seen:
            raise CorpusError("duplicate candidate source id")
        seen.add(source["source_id"])
        rate = source["sample_rate"]
        if isinstance(rate, bool) or not isinstance(rate, (int, float)) or not 0 < rate <= 1:
            raise CorpusError("candidate sample_rate must be in (0, 1]")
        if isinstance(source["max_source_share"], bool) or not 0 < source["max_source_share"] <= 1:
            raise CorpusError("candidate max_source_share must be in (0, 1]")
        _positive_int(source["target_raw_tokens"], "target_raw_tokens")
        if source["expected_language"] not in ("zh", "en"):
            raise CorpusError("candidate expected_language must be zh or en")
        total += source["target_raw_tokens"]
    if total < V5_CANDIDATE_TARGET:
        raise CorpusError("candidate source targets do not reach 15M raw tokens")
    return config


# ---------------------------------------------------------------------------
# resumable byte-range download

class _RangeResponse:
    def __init__(self, response):
        self._response = response
        self.status = getattr(response, "status", 200)
        self.headers = getattr(response, "headers", {})

    def read(self, size=-1):
        return self._response.read(size)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self._response.close()
        return False


def download_resumable(url: str, destination: str | Path, *, expected_sha1=None,
                       timeout=60, opener=None) -> dict:
    """Download with HTTP Range resume, then verify SHA1 before publishing.

    A truncated ``.part`` file is never accepted as a source.  If the server
    ignores ``Range`` the partial file is discarded and the transfer restarts.
    """
    if not isinstance(url, str) or urlsplit(url).scheme not in ("http", "https"):
        raise CorpusError("resumable download requires an HTTP(S) URL")
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    part = destination.with_name(destination.name + ".part")
    offset = part.stat().st_size if part.exists() else 0
    request = urllib.request.Request(url, headers={"Range": f"bytes={offset}-"} if offset else {})
    open_url = opener or urllib.request.urlopen
    started = time.monotonic()
    try:
        response = open_url(request, timeout=timeout)
    except (urllib.error.HTTPError, urllib.error.URLError, OSError) as exc:
        raise CorpusError(f"download failed: {exc}") from exc
    with response:
        status = getattr(response, "status", 200)
        if offset and status != 206:
            offset = 0
            part.unlink(missing_ok=True)
        mode = "ab" if offset else "wb"
        with part.open(mode) as handle:
            while True:
                block = response.read(1048576)
                if not block:
                    break
                handle.write(block)
            handle.flush()
            os.fsync(handle.fileno())
    if expected_sha1 is not None:
        digest = hashlib.sha1()
        with part.open("rb") as handle:
            for block in iter(lambda: handle.read(8 * 1048576), b""):
                digest.update(block)
        if digest.hexdigest() != expected_sha1.lower():
            raise CorpusError("downloaded file SHA1 differs from the advertised digest")
    os.replace(part, destination)
    return {"url": url, "path": str(destination.resolve()), "bytes": destination.stat().st_size,
            "sha1": expected_sha1, "resumed_from": offset, "seconds": time.monotonic() - started}


# ---------------------------------------------------------------------------
# deterministic streaming sample


def _rank(seed: int, source_id: str, record_id: str) -> int:
    data = f"{seed}\0{source_id}\0{record_id}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(data).digest(), "big")


def _threshold(rate: float) -> int:
    return max(1, min(2 ** 256 - 1, int(rate * 2 ** 256)))


def sample_source(spec: dict, config: dict, sampled_dir: str | Path,
                  *, stop_after=None, log=print) -> dict:
    """Hash-sample one JSONL source with byte-offset checkpoints.

    Selection depends only on ``(seed, source_id, record_id)``; changing input
    order would change the sample, so the input SHA256 is part of the state.
    """
    source_id = spec["source_id"]
    path = Path(spec["path"])
    if not path.is_file():
        raise CorpusError(f"candidate source missing: {path}")
    sampled_dir = Path(sampled_dir)
    sampled_dir.mkdir(parents=True, exist_ok=True)
    output = sampled_dir / f"{source_id}.jsonl"
    part = output.with_name(output.name + ".part")
    state_path = sampled_dir / f"{source_id}.state.json"
    input_info = file_info(path)
    state = read_json(state_path) if state_path.is_file() else None
    if state is not None:
        if state.get("input") != input_info or state.get("source_id") != source_id:
            raise CorpusError(f"{source_id}: input changed since the sampling checkpoint")
        offset = state["offset"]
        output_bytes = state["output_bytes"]
        produced = state["records_kept"]
        seen = state["records_seen"]
        malformed = state.get("malformed", 0)
        if state.get("complete"):
            verify_file(output, state["output"])
            return {"source_id": source_id, "status": "COMPLETE", "resumed": True,
                    "records_seen": seen, "records_kept": produced,
                    "malformed": malformed, "output": state["output"]}
        if part.exists():
            with part.open("r+b") as handle:
                handle.truncate(output_bytes)
        else:
            raise CorpusError(f"{source_id}: partial output is missing")
    else:
        offset = produced = seen = malformed = 0
        part.write_bytes(b"")
    threshold = _threshold(spec["sample_rate"])
    commit_every = config["commit_every"]
    started = time.monotonic()
    with path.open("rb") as source, part.open("ab") as sink:
        source.seek(offset)
        while True:
            line_start = source.tell()
            line = source.readline()
            if not line:
                break
            seen += 1
            offset = source.tell()
            if len(line) > 2 * 1048576:
                malformed += 1
                continue
            try:
                row = json.loads(line.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                malformed += 1
                continue
            if not isinstance(row, dict) or not isinstance(row.get("text"), str) or not row["text"].strip():
                malformed += 1
                continue
            raw_id = row.get("id", row.get("document_id"))
            if raw_id is None:
                raw_id = hashlib.sha256(row["text"].encode("utf-8")).hexdigest()
            raw_id = str(raw_id)
            if _rank(config["seed"], source_id, raw_id) >= threshold:
                continue
            document_id = f"{source_id}:{identity([source_id, raw_id, row['text']])[:24]}"
            record = {
                "document_id": document_id,
                "source_record_id": raw_id,
                "source_id": source_id,
                "category": spec["category"],
                "language_hint": spec["expected_language"],
                "text": row["text"],
                "source_url": row.get("url"),
                "revision_id": row.get("revision_id"),
                "input_path": str(path.resolve()),
                "input_sha256": input_info["sha256"],
                "input_line": seen,
                "sample_rank": f"{_rank(config['seed'], source_id, raw_id):064x}",
            }
            sink.write(json_bytes(record))
            produced += 1
            if stop_after is not None and seen >= stop_after:
                sink.flush()
                os.fsync(sink.fileno())
                output_bytes = sink.tell()
                break
            if produced % commit_every == 0:
                sink.flush()
                os.fsync(sink.fileno())
                state = {"schema_version": 1, "source_id": source_id, "input": input_info,
                         "offset": offset, "records_seen": seen, "records_kept": produced,
                         "malformed": malformed, "output_bytes": sink.tell(), "complete": False}
                write_json(state_path, state)
        completed = stop_after is None
        sink.flush()
        os.fsync(sink.fileno())
        output_bytes = sink.tell()
    state = {"schema_version": 1, "source_id": source_id, "input": input_info,
             "offset": offset, "records_seen": seen, "records_kept": produced,
             "malformed": malformed, "output_bytes": output_bytes, "complete": completed}
    if completed:
        os.replace(part, output)
        state["output"] = file_info(output)
        write_json(state_path, state)
    else:
        write_json(state_path, state)
    log(f"[v5 sample] {source_id}: {produced}/{seen} records, {output_bytes} bytes")
    return {"source_id": source_id, "status": "COMPLETE" if completed else "PARTIAL",
            "resumed": state is not None, "records_seen": seen, "records_kept": produced,
            "malformed": malformed, "output": file_info(output) if completed else None,
            "output_bytes": output_bytes, "seconds": time.monotonic() - started}


def sample_all(config: dict, review: dict, sampled_dir: str | Path, *, stop_after=None,
               log=print) -> list[dict]:
    entries = {entry["v4_source"]["source_id"]: entry for entry in review["sources"]}
    results = []
    for spec in config["sources"]:
        entry = entries.get(spec["source_id"])
        if entry is None:
            raise CorpusError(f"candidate source {spec['source_id']} is absent from the review registry")
        source = entry["v4_source"]
        declared = Path(spec["path"]).resolve()
        if declared != Path(source["paths"][0]).resolve():
            raise CorpusError(f"{spec['source_id']}: candidate path differs from the reviewed path")
        if source["category"] != spec.get("category"):
            spec = {**spec, "category": source["category"]}
        else:
            spec = {**spec, "category": source["category"]}
        results.append(sample_source(spec, config, sampled_dir, stop_after=stop_after, log=log))
    return results


# ---------------------------------------------------------------------------
# chinese style diagnostic


_CLASSICAL = ("之", "乎", "者", "也", "矣", "焉", "哉", "曰", "其", "而", "於", "乃",
              "遂", "皆", "未", "無", "此", "何", "吾", "汝", "爾", "兮", "夫", "蓋")
_MODERN = ("的", "了", "着", "著", "们", "們", "这", "這", "那", "是", "在", "有", "和",
           "就", "都", "很", "我", "你", "他", "她", "它", "不", "也", "说", "說")


def chinese_style(text: str) -> str:
    """Heuristic modern/classical composition signal, not a literary classifier."""
    if not text:
        return "unknown"
    sample = text[:20000]
    classical = sum(sample.count(token) for token in _CLASSICAL) * 1000 / len(sample)
    modern = sum(sample.count(token) for token in _MODERN) * 1000 / len(sample)
    if classical >= 12 and modern < 10:
        return "classical"
    if modern >= 10:
        return "modern"
    if classical or modern:
        return "mixed"
    return "unknown"


# ---------------------------------------------------------------------------
# candidate quality / dedup / cross-version audit


def _write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _load_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _load_sampled(sampled_dir: Path) -> list[dict]:
    rows = []
    for path in sorted(sampled_dir.glob("*.jsonl")):
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    rows.append(json.loads(line))
    if not rows:
        raise CorpusError("no sampled candidate records")
    return rows


def _batched(rows, size=1000):
    batch = []
    for row in rows:
        batch.append(row)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch


def split_long_text(text: str, max_chars: int) -> list[str]:
    """Split only at paragraph/sentence boundaries; never drop or rewrite text."""
    if max_chars <= 0:
        raise ValueError("max_chars must be positive")
    if len(text) <= max_chars:
        return [text]
    pieces = re.split(r'(?<=\n)', text)
    chunks: list[str] = []
    current = ""
    for piece in pieces:
        if len(piece) > max_chars:
            if current:
                chunks.append(current)
                current = ""
            sentences = re.split(r'(?<=[。！？!?；;])', piece)
            for sentence in sentences:
                if len(sentence) > max_chars:
                    if current:
                        chunks.append(current)
                        current = ""
                    chunks.extend(sentence[start:start + max_chars]
                                  for start in range(0, len(sentence), max_chars))
                elif current and len(current) + len(sentence) > max_chars:
                    chunks.append(current)
                    current = sentence
                else:
                    current += sentence
        elif current and len(current) + len(piece) > max_chars:
            chunks.append(current)
            current = piece
        else:
            current += piece
    if current:
        chunks.append(current)
    if "".join(chunks) != text:
        raise CorpusError("long-document split changed the source text")
    return chunks


def prepare_candidates(rows: list[dict], config: dict, tokenizer: MiniLLMTokenizer,
                       *, log=print) -> tuple[list[dict], dict]:
    """Normalise, split only overlong documents, quality-review and exact-dedup."""
    policy = config["quality"]
    strip_html = policy["strip_html"]
    max_chars = policy["max_chars"]
    by_action = Counter()
    by_rule = Counter()
    per_source = defaultdict(lambda: {"sampled_documents": 0, "sampled_tokens": 0,
                                      "kept_documents": 0, "kept_tokens": 0,
                                      "rejected_documents": 0, "rejected_tokens": 0,
                                      "review_documents": 0, "review_tokens": 0})
    clean = []
    seen = {}
    split_documents = 0
    candidate_parts = 0
    exact_duplicates = 0
    for row in rows:
        source = row["source_id"]
        stats = per_source[source]
        stats["sampled_documents"] += 1
        text = normalise(row["text"], row["language_hint"], strip_html=strip_html)
        detected = detect_language(text, category=row["category"])
        parts = split_long_text(text, max_chars)
        if len(parts) > 1:
            split_documents += 1
        document_tokens = 0
        for part_index, part in enumerate(parts):
            candidate_parts += 1
            tokens = len(tokenizer.encode_document(part, category=detected))
            document_tokens += tokens
            findings = evaluate_quality(part, language=detected, category=row["category"], policy=policy)
            action = aggregate_action(findings)
            by_action[action] += 1
            for finding in findings:
                by_rule[f"{finding.rule_id}:{finding.action}"] += 1
            if action != "KEEP":
                stats["rejected_documents" if action == "REJECT" else "review_documents"] += 1
                stats["rejected_tokens" if action == "REJECT" else "review_tokens"] += tokens
                continue
            digest = content_hash(part, detected)
            if digest in seen:
                exact_duplicates += 1
                continue
            part_id = (row["document_id"] if len(parts) == 1
                       else f"{row['document_id']}::p{part_index:04d}")
            seen[digest] = part_id
            stats["kept_documents"] += 1
            stats["kept_tokens"] += tokens
            clean.append({
                **{key: row[key] for key in ("source_id", "category", "language_hint",
                                             "source_url", "revision_id", "source_record_id",
                                             "sample_rank", "input_path", "input_sha256")},
                "document_id": part_id,
                "parent_document_id": row["document_id"],
                "part_index": part_index,
                "part_count": len(parts),
                "language": detected,
                "text": part,
                "characters": len(part),
                "tokens": tokens,
                "quality_action": action,
                "quality_rules": sorted({f.rule_id for f in findings}),
                "content_hash": digest,
                "chinese_style": chinese_style(part) if detected == "zh" else None,
            })
        stats["sampled_tokens"] += document_tokens
    report = {
        "schema_version": 1,
        "policy_rules_version": QUALITY_RULES_VERSION,
        "sampled_documents": len(rows),
        "sampled_tokens": sum(row["sampled_tokens"] for row in per_source.values()),
        "split_documents": split_documents,
        "candidate_parts": candidate_parts,
        "kept_documents": len(clean),
        "kept_tokens": sum(row["kept_tokens"] for row in per_source.values()),
        "exact_duplicates": exact_duplicates,
        "actions": dict(sorted(by_action.items())),
        "rules": dict(sorted(by_rule.items())),
        "by_source": dict(sorted(per_source.items())),
        "limitations": ("Explainable heuristic rules; REVIEW is excluded from candidate supply, never treated as approval. "
                        "Overlong documents are split only at paragraph/sentence boundaries and the concatenation is verified byte-for-byte."),
    }
    log(f"[v5 quality] {len(clean)}/{len(rows)} documents kept, {candidate_parts} candidate parts, "
        f"{split_documents} overlong documents split")
    return clean, report



def near_deduplicate(rows: list[dict], policy: NearDupPolicy, *, db_path: Path,
                     log=print) -> tuple[list[dict], dict]:
    """Exact-Jaccard-confirmed MinHash near-dedup with deterministic order."""
    rows = sorted(rows, key=lambda row: (row["source_id"], row["sample_rank"], row["document_id"]))
    by_id = {row["document_id"]: row for row in rows}
    db = sqlite3.connect(db_path)
    try:
        index = MinHashLSHIndex(db, policy)
        kept, duplicates = [], []
        cache = {}

        def feature_set(document_id, chunk_index):
            key = (document_id, chunk_index)
            if key not in cache:
                if len(cache) >= 4096:
                    cache.clear()
                document = by_id[document_id]
                chunk = chunk_text(document["text"], chunk_index, policy)
                if chunk is None:
                    cache[key] = frozenset()
                else:
                    values, _ = shingles(chunk, document["language"], policy)
                    cache[key] = values
            return cache[key]

        for row in rows:
            duplicate = None
            similarity = None
            for chunk_index, _chunk, values, signature, _truncated in chunk_features(
                    row["text"], row["language"], policy):
                for other_id, other_chunk in index.candidates(row["document_id"], chunk_index, signature):
                    score = exact_jaccard(values, feature_set(other_id, other_chunk))
                    if score >= policy.threshold:
                        duplicate, similarity = other_id, score
                        break
                if duplicate:
                    break
            if duplicate:
                duplicates.append({"document_id": row["document_id"], "duplicate_of": duplicate,
                                   "similarity": round(similarity, 6), "source_id": row["source_id"]})
            else:
                index.add(row["document_id"], row["text"], row["language"])
                kept.append(row)
        stats = {**index.annotated(), "input_documents": len(rows), "kept_documents": len(kept),
                 "duplicates": len(duplicates), "policy_sha256": policy.policy_sha256}
    finally:
        db.close()
    log(f"[v5 dedup] {len(kept)} kept, {len(duplicates)} near duplicates")
    return kept, stats


def load_historical(config_path: str | Path = HISTORICAL_CONFIG) -> dict:
    config = read_json(Path(config_path))
    if config.get("schema_version") != 1 or config.get("corpus_version") != V5_VERSION:
        raise CorpusError("invalid v5 historical evaluation configuration")
    sets = config.get("sets")
    if not isinstance(sets, list) or not sets:
        raise CorpusError("v5 historical evaluation sets are required")
    loaded = []
    for item in sets:
        required = {"name", "split", "path", "manifest", "file", "language"}
        if not isinstance(item, dict) or set(item) != required:
            raise CorpusError("historical set fields must be exactly name/split/path/manifest/file/language")
        if item["split"] not in ("val", "test") or item["language"] not in ("zh", "en"):
            raise CorpusError("historical set split/language must be val/test and zh/en")
        path = (ROOT / item["path"]).resolve()
        manifest_path = (ROOT / item["manifest"]).resolve()
        if not path.is_file() or not manifest_path.is_file():
            raise CorpusError(f"missing historical evaluation file: {item['path']}")
        manifest = read_json(manifest_path)
        info = manifest.get("files", {}).get(item["file"])
        if not _check_file_identity(info):
            raise CorpusError(f"historical manifest lacks {item['file']}")
        verify_file(path, info)
        rows = []
        with path.open(encoding="utf-8") as handle:
            for number, line in enumerate(handle, 1):
                if line.strip():
                    row = json.loads(line)
                    if isinstance(row.get("text"), str):
                        document_id = row.get("document_id") or f"{item['name']}:{number}"
                        rows.append({"document_id": document_id,
                                     "text": row["text"], "split": item["split"],
                                     "set": item["name"], "language": item["language"]})
        loaded.append({"name": item["name"], "split": item["split"], "path": str(path),
                       "language": item["language"], "file": info, "documents": rows})
    if not any(item["split"] == "val" for item in loaded) or not any(
            item["split"] == "test" for item in loaded):
        raise CorpusError("v5 requires both historical validation and test sets")
    return {"config": config, "sets": loaded}


def cross_version_audit(rows: list[dict], historical: dict, policy: NearDupPolicy,
                        *, db_path: Path, log=print) -> tuple[list[dict], dict]:
    """Exact + near duplicate audit against historical val/test documents."""
    historical_rows = [row for item in historical["sets"] for row in item["documents"]]
    exact = {}
    for item in historical["sets"]:
        for row in item["documents"]:
            exact.setdefault(content_hash(normalise(row["text"], item["language"]),
                                          item["language"]), row["document_id"])
    db = sqlite3.connect(db_path)
    try:
        index = MinHashLSHIndex(db, policy, prefix="historical")
        for item in historical["sets"]:
            for row in item["documents"]:
                index.add(row["document_id"], row["text"], item["language"])
        kept, hits = [], []
        cache = {}

        def feature_set(text, language, chunk_index):
            key = (hash(text), chunk_index)
            if key not in cache:
                if len(cache) >= 4096:
                    cache.clear()
                chunk = chunk_text(text, chunk_index, policy)
                values, _ = shingles(chunk or "", language, policy)
                cache[key] = values
            return cache[key]

        for row in rows:
            digest = content_hash(row["text"], row["language"])
            if digest in exact:
                hits.append({"document_id": row["document_id"], "kind": "EXACT",
                             "benchmark_id": exact[digest], "similarity": 1.0})
                continue
            duplicate = None
            similarity = None
            for chunk_index, _chunk, values, signature, _truncated in chunk_features(
                    row["text"], row["language"], policy):
                for other_id, other_chunk in index.candidates(row["document_id"], chunk_index, signature):
                    # Historical feature sets are recomputed on demand and cached.
                    historical_doc = next((doc for item in historical["sets"]
                                           for doc in item["documents"]
                                           if doc["document_id"] == other_id), None)
                    if historical_doc is None:
                        continue
                    historical_values = feature_set(historical_doc["text"],
                                                    historical_doc["language"], other_chunk)
                    score = exact_jaccard(values, historical_values)
                    if score >= policy.threshold:
                        duplicate, similarity = other_id, score
                        break
                if duplicate:
                    break
            if duplicate:
                hits.append({"document_id": row["document_id"], "kind": "NEAR",
                             "benchmark_id": duplicate, "similarity": round(similarity, 6)})
            else:
                kept.append(row)
    finally:
        db.close()
    report = {"historical_documents": len(historical_rows), "audited_documents": len(rows),
              "removed_documents": len(hits), "hits": hits,
              "by_kind": dict(Counter(hit["kind"] for hit in hits)),
              "threshold": policy.threshold,
              "sets": [{"name": item["name"], "split": item["split"],
                        "documents": len(item["documents"]), "file": item["file"]}
                       for item in historical["sets"]],
              "limitations": "Lexical exact/near audit against the declared historical val/test files; not a semantic guarantee."}
    log(f"[v5 historical] removed {len(hits)} cross-version duplicates")
    return kept, report


def contamination_audit_candidates(rows: list[dict], config: dict) -> tuple[list[dict], dict]:
    assets = [{**asset, "path": str((ROOT / asset["path"]).resolve())}
              for asset in config["contamination"]["assets"]]
    report = contamination_audit(assets, ({"document_id": row["document_id"],
                                           "text": row["text"], "language": row["language"]}
                                          for row in rows),
                                 decisions=None,
                                 min_prompt_chars=config["contamination"]["min_prompt_chars"],
                                 max_hits=config["contamination"]["max_hits"],
                                 long_span_containment=config["contamination"]["long_span_containment"])
    high = {hit["train_document_id"] for hit in report["hits"] if hit["disposition"] == "REVIEW"}
    kept = [row for row in rows if row["document_id"] not in high]
    report["removed_documents"] = len(rows) - len(kept)
    report["removed_by_id"] = sorted(high)
    return kept, report


# ---------------------------------------------------------------------------
# quota selection and exact budget


def select_candidates(rows: list[dict], targets: dict[str, int], caps: dict[str, float],
                      *, log=print) -> tuple[list[dict], dict]:
    """Per-category, per-source-capped deterministic whole-document selection."""
    by_category = defaultdict(list)
    for row in rows:
        by_category[row["category"]].append(row)
    selected = []
    category_report = {}
    for category in V4_CATEGORIES:
        target = targets.get(category, 0)
        candidates = sorted(by_category.get(category, []),
                            key=lambda row: (row["source_id"], row["sample_rank"], row["document_id"]))
        chosen, per_source, total = [], Counter(), 0
        skipped_by_cap = 0
        if target:
            for row in candidates:
                if total >= target:
                    break
                cap = caps.get(row["source_id"], 1.0)
                if per_source[row["source_id"]] + row["tokens"] > cap * target:
                    skipped_by_cap += 1
                    continue
                chosen.append(row)
                per_source[row["source_id"]] += row["tokens"]
                total += row["tokens"]
        selected.extend(chosen)
        category_report[category] = {
            "target_raw_tokens": target,
            "selected_documents": len(chosen),
            "selected_raw_tokens": total,
            "available_documents": len(candidates),
            "available_raw_tokens": sum(row["tokens"] for row in candidates),
            "gap_raw_tokens": max(0, target - total),
            "skipped_by_source_cap": skipped_by_cap,
            "by_source": dict(sorted(per_source.items())),
        }
    log(f"[v5 selection] {len(selected)} documents, {sum(row['tokens'] for row in selected)} raw tokens")
    return selected, category_report


def _packing(rows: list[dict]) -> dict:
    ordered = []
    for category in V4_CATEGORIES:
        ordered.extend(sorted((row for row in rows if row["category"] == category),
                              key=lambda row: (row["source_id"], row["sample_rank"], row["document_id"])))
    return supervision_by_category(((row["category"], row["tokens"]) for row in ordered), V5_SEQ_LEN)


def budget_for(rows: list[dict], target: int) -> dict:
    packing = _packing(rows)
    available = {category: {"documents": packing["documents_by_category"][category],
                            "effective_tokens": packing["by_category"][category]}
                 for category in V4_CATEGORIES}
    return plan_budget(available, target, V5_CATEGORY_RATIOS, seq_len=V5_SEQ_LEN,
                       tokenizer_fingerprint=TOKENIZER_FINGERPRINT,
                       license_review_status="NOT_APPROVED",
                       contamination_status="CANDIDATE_AUDIT_ONLY",
                       allow_repetition=False, fraction_tolerance=V5_TOLERANCE)


def training_preview(rows: list[dict], historical_seconds_per_update=MEASURED_SECONDS_PER_UPDATE,
                     *, holdout: float = 0.0) -> dict:
    """Exact full-update budgets for 4M/10M/50M/100M on the candidate packing.

    ``holdout`` reserves the validation/test fraction that is never trainable.
    The preview therefore reports post-split supply, not raw candidate tokens.
    """
    if not 0.0 <= holdout < 1.0:
        raise ValueError("holdout must be in [0, 1)")
    packing = _packing(rows)
    supply = {category: int(math.floor(tokens * (1.0 - holdout)))
              for category, tokens in packing["by_category"].items()}
    total_windows = int(math.floor(packing["effective_tokens"] * (1.0 - holdout))) // V5_SEQ_LEN
    report = {}
    for name, updates, tokens in TRAINING_BUDGETS:
        required_windows = updates * 8
        quota = {category: int(math.ceil(tokens * ratio))
                 for category, ratio in V5_CATEGORY_RATIOS.items() if ratio}
        gap = {category: max(0, quota[category] - supply[category]) for category in quota}
        feasible = required_windows <= total_windows and not any(gap.values())
        report[name] = {
            "planned_updates": updates,
            "supervised_tokens_per_update": V5_TOKENS_PER_UPDATE,
            "planned_supervised_tokens": tokens,
            "required_windows": required_windows,
            "available_windows": total_windows,
            "window_gap": max(0, required_windows - total_windows),
            "category_quota": quota,
            "category_supply": {category: supply[category] for category in quota},
            "category_gap": gap,
            "holdout_fraction": holdout,
            "feasible_on_candidate_inventory": feasible,
            "estimated_seconds": round(updates * historical_seconds_per_update, 3),
            "estimated_hours": round(updates * historical_seconds_per_update / 3600, 3),
            "basis": "one optimizer update = 8 windows x 1023 supervised targets = 8184; supply is after the configured val/test holdout; time extrapolated from the measured 2M run",
        }
    return {"seq_len": V5_SEQ_LEN, "tokens_per_update": V5_TOKENS_PER_UPDATE,
            "holdout_fraction": holdout, "candidate_packing": packing, "budgets": report,
            "time_basis": {"source": "v4-language-pilot-2m-v1", "updates": 245,
                           "seconds": 1531.928, "seconds_per_update": round(historical_seconds_per_update, 6)}}


# ---------------------------------------------------------------------------
# approval request / evidence


def approval_request(review: dict, inventory: dict | None = None) -> dict:
    entries = []
    for entry in review["sources"]:
        source = entry["v4_source"]
        by_source = (inventory or {}).get("by_source") or (inventory or {}).get("quality", {}).get("by_source", {})
        measured = by_source.get(source["source_id"], {})
        entries.append({
            "id": source["source_id"],
            "category": source["category"],
            "release": entry["release"],
            "source_uri": source["source_uri"],
            "upstream_snapshot_uri": entry.get("upstream_snapshot_uri"),
            "licence": source["license"],
            "licence_references": entry.get("license_references", []),
            "commercial_use": entry.get("commercial_use", "PENDING"),
            "redistribution": entry.get("redistribution", "PENDING"),
            "model_training": entry.get("model_training", "PENDING"),
            "licence_obligations": entry.get("license_obligations", "PENDING"),
            "local_files": entry["local_inventory"]["files"],
            "raw_files": entry["local_inventory"].get("raw_files", {}),
            "documents": entry["local_inventory"].get("documents"),
            "characters": entry["local_inventory"].get("characters"),
            "estimated_tokens": entry["local_inventory"].get("estimated_tokens"),
            "candidate_measured": measured,
            "new_approval": ("APPROVED" if source.get("license_status") == "APPROVED"
                             else "PENDING_HUMAN"),
        })
    approved_all = bool(review["sources"]) and all(
        entry["v4_source"].get("license_status") == "APPROVED" for entry in review["sources"])
    return {
        "corpus_version": V5_VERSION,
        "dataset_name": "minillm-corpus-v5-language-10m",
        "requested_sources": entries,
        "requested_budget": {
            "effective_train_tokens": V5_TARGET,
            "updated_steps": V5_UPDATES,
            "seq_len": V5_SEQ_LEN,
            "category_token_ratios": V5_CATEGORY_RATIOS,
            "fraction_tolerance": V5_TOLERANCE,
        },
        "requested_scope": {
            "intended_use": "Internal Chinese language-model research and training",
            "commercial_use": False,
            "external_redistribution": False,
            "public_weight_release": False,
            "gpu_training_in_this_pr": False,
        },
        "required_statement": (
            "Approve each source for corpus v5 and the 10,000,848-token internal research budget; "
            "state whether public weight release or commercial use is included."),
        "status": "APPROVED_READY_FOR_BUILD" if approved_all else "PENDING_HUMAN",
    }


def validate_approval(evidence_path: str | Path, request_path: str | Path, review: dict) -> dict:
    evidence = read_json(Path(evidence_path))
    request = read_json(Path(request_path))
    required = {"corpus_version", "decision", "reviewer", "recorded_at",
                "verbatim_human_confirmation", "effective_train_tokens",
                "category_token_ratios", "sources", "approval_request"}
    if not isinstance(evidence, dict) or set(evidence) != required:
        raise CorpusError("v5 approval evidence fields are incomplete")
    if evidence["corpus_version"] != V5_VERSION or evidence["decision"] != "APPROVED":
        raise CorpusError("v5 approval is not an APPROVED decision for this version")
    _require_text(evidence["reviewer"], "approval reviewer")
    _iso_date(evidence["recorded_at"][:10], "approval recorded_at")
    _require_text(evidence["verbatim_human_confirmation"], "verbatim human confirmation")
    if evidence["effective_train_tokens"] != V5_TARGET:
        raise CorpusError("v5 approval budget differs from the pinned 10,000,848 tokens")
    if evidence["category_token_ratios"] != V5_CATEGORY_RATIOS:
        raise CorpusError("v5 approval category ratios differ from the pinned policy")
    if evidence["approval_request"] != file_info(Path(request_path)):
        raise CorpusError("v5 approval does not bind the generated approval request")
    if any(entry["v4_source"].get("license_status") == "REVIEW_REQUIRED"
           for entry in review["sources"]):
        raise CorpusError("v5 approval is incomplete: a source still requires review")
    approved = {entry["v4_source"]["source_id"]: entry for entry in review["sources"]
                if entry["v4_source"].get("license_status") == "APPROVED"}
    rows = evidence["sources"]
    if not isinstance(rows, list) or not rows:
        raise CorpusError("v5 approval must list approved sources")
    if {row.get("source_id") for row in rows} != set(approved):
        raise CorpusError("v5 approval source ids differ from the approved registry")
    for row in rows:
        entry = approved[row["source_id"]]
        if row.get("release") != entry["release"]:
            raise CorpusError(f"{row['source_id']}: approved release differs")
        if row.get("files") != entry["local_inventory"]["files"]:
            raise CorpusError(f"{row['source_id']}: approved file identities differ")
        if row.get("decision") != "APPROVED":
            raise CorpusError(f"{row['source_id']}: approval decision is not APPROVED")
    for entry in review["sources"]:
        if entry["v4_source"].get("license_status") == "REVIEW_REQUIRED":
            raise CorpusError("v5 approval is incomplete: a source still requires review")
    return {"evidence": evidence, "request": request, "approval_sha256": sha256(Path(evidence_path)),
            "request_sha256": sha256(Path(request_path))}


# ---------------------------------------------------------------------------
# tokenizer reuse and release checks


def tokenizer_receipt() -> dict:
    from .v4_tokenizer_decision import verify_selection
    from .v4_language_scaling import BASE as BASE_1M, BASE_MANIFEST

    verify_file(BASE_1M / "dataset-manifest.json", BASE_MANIFEST)
    verify_file(TOKENIZER, (read_json(BASE_1M / "dataset-manifest.json"))["files"]["tokenizer.json"])
    receipt = verify_selection(BASE_1M)
    if receipt["fingerprint"] != TOKENIZER_FINGERPRINT or receipt["vocab_size"] != TOKENIZER_VOCAB:
        raise CorpusError("v5 requires the original human-reviewed 24,576 tokenizer")
    return {**receipt, "path": str(TOKENIZER), "reused_from_content_id": BASE_ID,
            "v5_decision": "REUSE_FROZEN_TOKENIZER"}


def verify_tokenizer_reuse(directory: str | Path) -> dict:
    from .v4_language_scaling import BASE as BASE_1M

    directory = Path(directory)
    receipt = tokenizer_receipt()
    plan = read_json(directory / ".v4/plan.json")
    stage = read_json(directory / ".v4/tokenizer_selection.stage.json")
    split = read_json(directory / ".v4/split.stage.json")["key"]
    expected = identity([4, plan["plan_key"], "tokenizer_selection",
                         {"split": split, "base_content_id": BASE_ID}])
    if (stage["key"] != expected
            or read_json(directory / "tokenizer-selection.json") != receipt
            or stage["files"].get(str((directory / "tokenizer-selection.json").resolve()))
            != file_info(directory / "tokenizer-selection.json")):
        raise CorpusError("v5 tokenizer reuse receipt differs from the frozen decision")
    for name in ("tokenizer-evaluation.json", "tokenizer-candidates.json"):
        if (file_info(directory / name) != file_info(BASE_1M / name)
                or stage["files"].get(str((directory / name).resolve()))
                != file_info(BASE_1M / name)):
            raise CorpusError("v5 historical tokenizer evidence changed")
    return receipt


def verify_release(directory: str | Path) -> dict:
    """V5-specific release evidence on top of the generic V4/language gates."""
    directory = Path(directory)
    manifest = read_json(directory / "dataset-manifest.json")
    plan = read_json(directory / ".v4/plan.json")
    if (manifest["corpus_version"] != V5_VERSION or plan["corpus_version"] != V5_VERSION
            or manifest["budget"]["target_effective_train_tokens"] != V5_TARGET):
        raise CorpusError("v5 release version or target differs")
    from .v4_sources import validate_source_set
    validate_source_set(manifest["license_sources"])
    receipt = verify_tokenizer_reuse(directory)
    if receipt["fingerprint"] != manifest["tokenizer_fingerprint"]:
        raise CorpusError("v5 release tokenizer differs from the reuse receipt")
    historical = load_historical()
    packing = read_json(directory / "budget-report.json")
    if packing["target_effective_train_tokens"] != V5_TARGET:
        raise CorpusError("v5 release budget target differs")
    return {"corpus_version": V5_VERSION, "content_id": manifest["content_id"],
            "target_effective_train_tokens": V5_TARGET,
            "tokenizer_fingerprint": receipt["fingerprint"],
            "historical_sets": [item["name"] for item in historical["sets"]]}


# ---------------------------------------------------------------------------
# inventory orchestration


def inventory(out_dir: str | Path, *, registry_path: str | Path = SOURCE_REGISTRY,
              candidate_path: str | Path = CANDIDATE_CONFIG, resume=True,
              log=print) -> dict:
    """Run the candidate inventory.  Never writes a freeze marker."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    review = load_source_review(registry_path, verify_inputs=True)
    candidate_config = load_candidate_config(candidate_path)
    sampled = out / "sampled"
    sample_stats = sample_all(candidate_config, review, sampled, log=log)
    rows = _load_sampled(sampled)
    tokenizer = MiniLLMTokenizer.load(TOKENIZER)
    if tokenizer.fingerprint != TOKENIZER_FINGERPRINT or tokenizer.vocab_size != TOKENIZER_VOCAB:
        raise CorpusError("frozen tokenizer identity changed")
    clean, quality = prepare_candidates(rows, load_corpus_config(), tokenizer, log=log)
    policy = NearDupPolicy.from_config(load_corpus_config()["deduplication"]["near"])
    clean, dedup = near_deduplicate(clean, policy, db_path=out / "near-dedup.sqlite", log=log)
    historical = load_historical()
    clean, cross = cross_version_audit(clean, historical, policy,
                                       db_path=out / "historical-dedup.sqlite", log=log)
    clean, contamination = contamination_audit_candidates(clean, load_corpus_config())
    _write_rows(out / "clean-audited.jsonl", clean)
    targets = {category: int(round(V5_CANDIDATE_TARGET * ratio))
               for category, ratio in V5_CATEGORY_RATIOS.items()}
    caps = {}
    for spec in candidate_config["sources"]:
        entry = next((e for e in review["sources"]
                      if e["v4_source"]["source_id"] == spec["source_id"]), None)
        caps[spec["source_id"]] = spec["max_source_share"]
    selected, selection = select_candidates(clean, targets, caps, log=log)
    packing = _packing(selected)
    candidate_budget = budget_for(selected, V5_CANDIDATE_TARGET)
    corpus_config = load_corpus_config()
    seq_len = corpus_config["targets"]["seq_len"]
    split_ratios = corpus_config["split"]["ratios"]
    holdout = split_ratios["val"] + split_ratios["test"]
    # Reserve the held-out fraction first, then leave 1% + packing slack so the
    # fixed 10,000,848-token train budget survives group-based split assignment.
    pre_split_target = math.ceil(V5_TARGET / (1.0 - holdout))
    freeze_targets = {category: (math.ceil(pre_split_target * ratio * 1.01) + 10 * seq_len
                                 if ratio else 0)
                      for category, ratio in V5_CATEGORY_RATIOS.items()}
    freeze_selected, freeze_selection = select_candidates(
        clean, freeze_targets, caps, log=lambda message: log(f"[v5 freeze-preview] {message}"))
    freeze_packing = _packing(freeze_selected)
    freeze_budget = budget_for(freeze_selected, V5_TARGET)
    training = training_preview(freeze_selected, holdout=holdout)
    styles = Counter(row["chinese_style"] for row in freeze_selected if row["chinese_style"])
    source_contrib = Counter()
    for row in freeze_selected:
        source_contrib[row["source_id"]] += row["tokens"]
    report = {
        "schema_version": 1,
        "corpus_version": V5_VERSION,
        "status": ("CANDIDATE_INVENTORY_READY"
                   if (freeze_packing["effective_tokens"] >= V5_TARGET
                       and all(category["satisfied"] for category in freeze_budget["categories"].values()))
                   else "CANDIDATE_INVENTORY_INSUFFICIENT"),
        "freeze_status": "BLOCKED_SOURCE_APPROVAL",
        "formal_freeze": "NOT_RUN",
        "gpu_training": "NOT_RUN",
        "seed": candidate_config["seed"],
        "source_registry": file_info(Path(registry_path)),
        "candidate_config": file_info(Path(candidate_path)),
        "tokenizer": {"fingerprint": tokenizer.fingerprint, "vocab_size": tokenizer.vocab_size,
                      "path": str(TOKENIZER)},
        "sampling": sample_stats,
        "quality": quality,
        "dedup": dedup,
        "historical": cross,
        "contamination": {key: value for key, value in contamination.items() if key != "hits"},
        "selection": selection,
        "packing": packing,
        "candidate_budget": candidate_budget,
        "freeze_preview": {"selection": freeze_selection, "packing": freeze_packing,
                           "budget": freeze_budget, "style": dict(styles),
                           "source_contributions": dict(source_contrib),
                           "holdout_fraction": holdout,
                           "pre_split_target": pre_split_target,
                           "estimated_train_effective_tokens":
                               int(freeze_packing["effective_tokens"] * (1.0 - holdout))},
        "freeze_budget": freeze_budget,
        "training_preview": training,
        "style": dict(styles),
        "source_contributions": dict(source_contrib),
        "limitations": [
            "Candidate inventory only; no source is approved and nothing is frozen.",
            "Token counts are exact frozen-tokenizer encodings but category counts are supply, not yet a formal split.",
            "Contamination is a bounded lexical audit, not semantic proof.",
        ],
    }
    report["content_id"] = identity(report)
    write_json(out / "inventory.json", report)
    write_json(out / "quality-report.json", quality)
    write_json(out / "dedup-report.json", dedup)
    write_json(out / "historical-dedup-report.json", cross)
    write_json(out / "contamination-report.json", contamination)
    write_json(out / "budget-preview.json", {"candidate": candidate_budget,
                                             "freeze": freeze_budget,
                                             "training": training})
    write_json(out / "source-approval-request.json", approval_request(review, quality))
    write_json(out / "candidate-manifest.json", {
        "schema_version": 1, "corpus_version": V5_VERSION,
        "content_id": report["content_id"],
        "files": {path.name: file_info(path) for path in sorted(out.glob("*.json"))
                  if path.name != "candidate-manifest.json"},
        "status": "CANDIDATE_ONLY_NOT_TRAINABLE",
    })
    log(f"[v5 inventory] {packing['effective_tokens']} effective candidate tokens; freeze BLOCKED")
    return report


def load_corpus_config(path: str | Path = CORPUS_CONFIG) -> dict:
    from .corpus_v4 import load_config

    config = load_config(path)
    if config["corpus_version"] != V5_VERSION:
        raise CorpusError("v5 corpus configuration version mismatch")
    if config["targets"]["effective_train_tokens"] != V5_TARGET:
        raise CorpusError("v5 corpus target must be 10,000,848 supervised tokens")
    if config["targets"]["category_token_ratios"] != V5_CATEGORY_RATIOS:
        raise CorpusError("v5 corpus ratios must be 45/30/15/10/0")
    return config


def prepare(out_dir: str | Path, *, registry_path: str | Path = SOURCE_REGISTRY,
            inventory_path: str | Path | None = None, log=print) -> dict:
    """Write the pending approval request and a fail-closed status."""
    out = Path(out_dir)
    review = load_source_review(registry_path, verify_inputs=True)
    inventory = read_json(inventory_path) if inventory_path else None
    request = approval_request(review, inventory)
    approved = len(approved_entries(review))
    total = len(review["sources"])
    ready = approved == total and total > 0
    if ready:
        # A licence marker alone is not enough: the pinned V5 scope must also
        # validate before prepare can advertise READY_FOR_BUILD.
        for entry in review["sources"]:
            validate_scope(entry)
    status = "READY_FOR_BUILD" if ready else "BLOCKED_SOURCE_APPROVAL"
    with report_directory(out) as directory:
        write_json(directory / "source-approval-request.json", request)
        if ready:
            write_json(directory / "source-review.approved.json", review)
        else:
            write_json(directory / "source-review.pending.json", review)
        write_json(directory / "status.json", {
            "corpus_version": V5_VERSION,
            "status": status,
            "approved_sources": approved,
            "pending_sources": [entry["v4_source"]["source_id"] for entry in pending_entries(review)],
            "target_effective_train_tokens": V5_TARGET,
            "updated_steps": V5_UPDATES,
            "formal_freeze": "NOT_RUN",
            "gpu_training": "NOT_RUN",
            "next": ("Run build with the approved source-approval evidence; formal freeze and GPU training remain not run."
                     if ready else
                     "A human must approve every source and the versioned 10M budget, then run build with the approval file."),
        })
        write_json(directory / "manifest.json", {"schema_version": 1,
                   "files": {path.name: file_info(path) for path in sorted(directory.glob("*.json"))}})
    log(f"[v5 prepare] {status} ({approved}/{total} sources approved)")
    return request


def build(dataset: str | Path, *, registry_path: str | Path, approval_path: str | Path,
          request_path: str | Path, out_dir: str | Path, log=print) -> dict:
    """Formal V5 build; requires complete human approval evidence."""
    review = load_source_review(registry_path, verify_inputs=True)
    evidence = validate_approval(approval_path, request_path, review)
    from .corpus_v4 import CorpusV4Pipeline, build_registry, inventory, load_config
    from .v4_admission import validate_v4_pilot_policy

    config = load_config(CORPUS_CONFIG, source_registry=registry_path)
    registry = build_registry(config["sources"], demo=False, license_policy=config["license_policy"])
    if registry["counts"]["excluded"]:
        raise CorpusError("v5 build requires every reviewed source to be admitted")
    pipeline = CorpusV4Pipeline(Path(dataset), config, registry,
                                inventory(registry, ROOT, config=config), log=log)
    audit = pipeline.audit()
    manifest = pipeline.finalize(audit=audit)
    if manifest["freeze_status"] != "READY":
        raise CorpusError(f"v5 build did not freeze: {manifest['freeze_status']}")
    release = verify_release(dataset)
    policy = validate_v4_pilot_policy(dataset)
    with report_directory(out_dir) as directory:
        write_json(directory / "approval.json", evidence)
        write_json(directory / "freeze-status.json", {"audit": audit, "policy": policy,
                   "release": release, "manifest": manifest["content_id"]})
    return {"manifest": manifest, "release": release, "policy": policy}
