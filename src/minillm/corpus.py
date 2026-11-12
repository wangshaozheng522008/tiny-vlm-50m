"""Reproducible, single-machine corpus pipeline with verified disk stages."""
from __future__ import annotations

import copy
import hashlib
import shutil
import importlib.metadata
import json
import math
import os
import subprocess
import time
import unicodedata
from pathlib import Path

from .checkpoint import atomic_write
from .corpus_quality import CATEGORIES, DEFAULT_QUALITY
from .corpus_store import deduplicate, find, normalize_sources
from .corpus_utils import (CorpusError, Stages, connect, database_output, file_info,
                           histogram_add, identity, json_bytes, sha256, verify_file, write_json)
from .data import encode_document_tokens
from .tokenizer import MiniLLMTokenizer
from .text_policy import validate_policy

SPLITS = ("train", "val", "test")
DEFAULT_DEDUP = {"method": "simhash", "hamming_distance": 4, "threshold": 0.85,
                 "ngram": 4, "max_candidates": 2000, "near_code": False}
DEFAULT_MIX = {"by": "category", "weights": {"zh": 0.8, "en": 0.15, "code": 0.05},
               "token_budget": 0, "basis": "estimated", "chars_per_token": {
                   "zh": 1.8, "en": 4.0, "code": 3.0, "mixed": 2.0, "unknown": 3.0}}
DEFAULT_RUNTIME = {"sqlite_cache_mib": 16, "buffer_bytes": 1048576, "commit_every": 1000}


def merge(base, override):
    result = copy.deepcopy(base)
    for key, value in override.items():
        if key not in result:
            raise CorpusError(f"unknown configuration field: {key}")
        if isinstance(result[key], dict) and isinstance(value, dict):
            # Weights/thresholds are user-defined maps rather than fixed schemas.
            result[key] = {**result[key], **value} if key != "weights" else dict(value)
        else:
            result[key] = value
    return result


def make_content(config: dict, overrides: dict | None = None):
    allowed = {"dataset_name", "sources", "quality", "deduplication", "mixing", "license_policy",
               "runtime", "seed", "shuffle_seed", "sample", "max_records", "split", "schema_version", "text_policy"}
    if set(config) - allowed:
        raise CorpusError(f"unknown config fields: {sorted(set(config) - allowed)}")
    if config.get("schema_version", 1) != 1:
        raise CorpusError("unsupported corpus configuration schema_version")
    content = {
        "dataset_name": config.get("dataset_name", "minillm-pretrain-v1"), "cleaning_version": 2,
        "seed": config.get("seed", 42), "shuffle_seed": config.get("shuffle_seed", 13),
        "sample": config.get("sample", 0), "max_records": config.get("max_records", 0),
        "split": config.get("split", [0.96, 0.02, 0.02]),
        "quality": merge(DEFAULT_QUALITY, config.get("quality", {})),
        "deduplication": merge(DEFAULT_DEDUP, config.get("deduplication", {})),
        "mixing": merge(DEFAULT_MIX, config.get("mixing", {})),
        "license_policy": merge({"allow": [], "exclude_unknown": False}, config.get("license_policy", {})),
    }
    for key, value in (overrides or {}).items():
        if value is not None:
            if "." in key:
                parent, child = key.split(".", 1)
                content[parent][child] = value
            else:
                content[key] = value
    if "text_policy" in config:
        content["text_policy"] = validate_policy(config["text_policy"])
    q, d, m = (content[name] for name in ("quality", "deduplication", "mixing"))
    policy = content["license_policy"]
    if (not isinstance(policy["allow"], list) or len(policy["allow"]) > 128
            or any(not isinstance(label, str) or not label or not label.isprintable() or len(label) > 128 for label in policy["allow"])
            or type(policy["exclude_unknown"]) is not bool):
        raise CorpusError("license_policy requires a list of license labels and a boolean exclude_unknown")
    if content["sample"] < 0 or content["max_records"] < 0:
        raise CorpusError("sample/max_records must be nonnegative")
    if len(content["split"]) != 3 or any(not math.isfinite(x) or x < 0 for x in content["split"]) or not math.isclose(sum(content["split"]), 1.0) or content["split"][0] <= 0:
        raise CorpusError("split fractions must be nonnegative, sum to one and include train")
    if not 1 <= q["min_chars"] <= q["max_chars"] <= q["max_parent_chars"] <= 2000000:
        raise CorpusError("require 1 <= min_chars <= max_chars <= max_parent_chars <= 2000000")
    if not 1024 <= q["max_record_bytes"] <= 8 * 1048576 or q["long_document"] not in ("split", "reject"):
        raise CorpusError("invalid max_record_bytes or long_document policy")
    if d["method"] not in ("exact", "simhash") or not 0 <= d["hamming_distance"] <= 8 or not 0 < d["threshold"] <= 1 or not 2 <= d["ngram"] <= 8 or not 1 <= d["max_candidates"] <= 100000:
        raise CorpusError("invalid deduplication method/radius/threshold/ngram/candidate budget")
    if m["by"] not in ("source", "category") or m["basis"] != "estimated" or m["token_budget"] < 0:
        raise CorpusError("mixing requires by=source/category, basis=estimated and nonnegative token_budget")
    if not m["weights"] or any(not math.isfinite(w) or w < 0 for w in m["weights"].values()) or not math.isclose(sum(m["weights"].values()), 1.0):
        raise CorpusError("mixing weights must be finite, nonnegative and sum to one")
    if any(not math.isfinite(m["chars_per_token"][c]) or m["chars_per_token"][c] <= 0 for c in CATEGORIES):
        raise CorpusError("chars_per_token must be positive for each category")
    return content


def inventory(paths: list[str], sources: list[dict] | None = None, *, base: Path = Path(".")):
    specifications = sources or [{"name": "unknown", "paths": paths}]
    if len(specifications) > 128:
        raise CorpusError("at most 128 configured sources are supported")
    result = []
    for spec in specifications:
        if set(spec) - {"name", "paths", "license", "category", "priority", "enabled"}:
            raise CorpusError("unknown source configuration field")
        name = spec.get("name", "unknown")
        if not isinstance(name, str) or not name or len(name) > 128:
            raise CorpusError("source name must be a nonempty string of at most 128 characters")
        if not name.isprintable() or type(spec.get("enabled", True)) is not bool:
            raise CorpusError("source name must be printable and enabled must be a boolean")
        license_ = spec.get("license", "unknown")
        if not isinstance(license_, str) or not license_ or not license_.isprintable() or len(license_) > 128:
            raise CorpusError("configured source license must be a printable label")
        category = spec.get("category", "auto")
        if category not in (*CATEGORIES, "auto"):
            raise CorpusError(f"invalid category: {category}")
        for pattern in spec.get("paths", []):
            candidate = Path(pattern) if Path(pattern).is_absolute() else base / pattern
            # Explicit files or directory trees; stable sorting is independent of CLI order.
            candidates = candidate.rglob("*") if candidate.is_dir() else (candidate,)
            for path in candidates:
                if path.suffix.lower() not in (".jsonl", ".txt"):
                    continue
                if len(result) >= 10000:
                    raise CorpusError("at most 10000 input files; consolidate files or prepare separate versions")
                if not path.is_file():
                    raise CorpusError(f"missing source file: {path}")
                result.append({"path": str(path.resolve()), "source": name, "category": category,
                               "license": spec.get("license", "unknown"), "priority": int(spec.get("priority", 0)),
                               "enabled": bool(spec.get("enabled", True)), **file_info(path)})
    if not result:
        raise CorpusError("no JSONL/TXT source files found")
    # Reading the same file twice is unnecessary; identical bytes from different sources
    # remain in the inventory so cross-source duplicate accounting is auditable.
    unique = {json_bytes(item): item for item in result}
    return sorted(unique.values(), key=lambda x: (x["source"], x["sha256"], x["path"]))


def input_identity(inputs):
    return sorted(({k: v for k, v in row.items() if k != "path"} for row in inputs), key=json_bytes)


def implementation_identity():
    modules = ("corpus.py", "corpus_store.py", "corpus_quality.py", "corpus_utils.py", "data.py", "tokenizer.py", "text_policy.py")
    return identity({"modules": {name: sha256(Path(__file__).with_name(name)) for name in modules},
                     "tokenizers_version": importlib.metadata.version("tokenizers"),
                     "unicode_version": unicodedata.unidata_version})


def read_records(path: Path):
    with path.open("rb") as handle:
        while line := handle.readline(16 * 1048576 + 1):
            if len(line) > 16 * 1048576:
                raise CorpusError(f"oversized intermediate record: {path}")
            try:
                record = json.loads(line)
            except (ValueError, UnicodeError) as exc:
                raise CorpusError(f"invalid intermediate JSON: {path}") from exc
            if not isinstance(record, dict) or not isinstance(record.get("text"), str):
                raise CorpusError(f"missing intermediate text: {path}")
            yield record


def write_splits(path: Path, dedup_path: Path, out: Path, content: dict, runtime: dict):
    stats = {name: {"documents": 0, "chars": 0, "estimated_tokens": 0} for name in SPLITS}
    mixing = content["mixing"]
    selection = {}
    with database_output(path, source=dedup_path, cache_mib=runtime["sqlite_cache_mib"]) as db:
        db.execute("CREATE TABLE chosen(id TEXT PRIMARY KEY, split TEXT, group_id TEXT)")
        field = mixing["by"]
        available = [dict(r) for r in db.execute(f"SELECT f.{field} bucket,COUNT(*) documents,SUM(estimated) estimated_tokens FROM fragments f JOIN eligible e ON e.id=f.id GROUP BY f.{field} ORDER BY f.{field}")]
        if mixing["token_budget"] and any(r["bucket"] not in mixing["weights"] for r in available):
            raise CorpusError("budgeted mixing requires an explicit weight (possibly zero) for every available source/category")
        for bucket in available:
            name, used, documents = bucket["bucket"], 0, 0
            quota = math.floor(mixing["token_budget"] * mixing["weights"].get(name, 0)) if mixing["token_budget"] else None
            for row in db.execute(f"SELECT f.id,f.parent_id,f.estimated FROM fragments f JOIN eligible e ON e.id=f.id WHERE f.{field}=? ORDER BY order_key,f.id", (name,)):
                if quota is not None and used + row["estimated"] > quota:
                    continue
                group = find(db, "docs", "split_parent", row["parent_id"])
                score = int(identity([content["seed"], "split", group])[:16], 16) / 2**64
                split = "train" if score < content["split"][0] else ("val" if score < sum(content["split"][:2]) else "test")
                db.execute("INSERT INTO chosen VALUES(?,?,?)", (row["id"], split, group))
                used += row["estimated"]
                documents += 1
            selection[name] = {**bucket, "target_estimated_tokens": quota, "selected_estimated_tokens": used,
                               "selected_documents": documents, "exhausted": documents == bucket["documents"],
                               "estimated_shortfall": max(0, (quota or 0) - used)}
        for name in SPLITS:
            def writer(handle):
                query = "SELECT f.*,c.group_id,d.metadata,k.hash comparison_key FROM fragments f JOIN chosen c ON c.id=f.id JOIN docs d ON d.id=f.parent_id JOIN comparison_keys k ON k.id=f.id WHERE c.split=? ORDER BY f.order_key,f.id"
                for row in db.execute(query, (name,)):
                    record = {"id": row["id"], "parent_id": row["parent_id"], "group_id": row["group_id"],
                              "text": row["text"], "source": row["source"], "category": row["category"],
                              "language": row["language"], "license": row["license"], "estimated_tokens": row["estimated"],
                              "declared_category": json.loads(row["metadata"])["declared_category"],
                              "raw_sha256": row["parent_id"], "comparison_sha256": row["comparison_key"]}
                    handle.write(json_bytes(record))
                    stats[name]["documents"] += 1
                    stats[name]["chars"] += len(row["text"])
                    stats[name]["estimated_tokens"] += row["estimated"]
            atomic_write(out / f"{name}.text.jsonl", writer)
        if not stats["train"]["documents"]:
            raise CorpusError("empty train split; add independent document groups or adjust seed/split/mixing budget")
        known = db.execute("SELECT COUNT(*) FROM (SELECT group_id FROM chosen GROUP BY group_id HAVING COUNT(DISTINCT split)>1)").fetchone()[0]
        if known:
            raise CorpusError("known duplicate groups cross splits")
        source_distribution = [dict(r) for r in db.execute("SELECT f.source,COUNT(*) documents,SUM(estimated) estimated_tokens FROM chosen c JOIN fragments f ON f.id=c.id GROUP BY f.source ORDER BY f.source")]
    return {"documents": sum(r["documents"] for r in stats.values()), "splits": stats, "selection": selection,
            "by_source": source_distribution, "known_near_duplicate_cross_split_groups": 0}, [path, *(out / f"{n}.text.jsonl" for n in SPLITS)]


def tokenize_split(records, tokenizer: MiniLLMTokenizer, out_path: Path, *, buffer_bytes: int = 1048576):
    if not 1 <= tokenizer.vocab_size <= 65536 or buffer_bytes < 2:
        raise CorpusError("invalid uint16 vocabulary or token buffer size")
    stats = {"documents": 0, "chars": 0, "tokens": 0, "by_source": {}, "by_category": {},
             "character_length_histogram": {}, "token_length_histogram": {}}
    digest = hashlib.sha256()
    def writer(handle):
        buffer = bytearray()
        for record in records:
            text = record["text"] if isinstance(record, dict) else record
            source = record.get("source", "unknown") if isinstance(record, dict) else "unknown"
            category = record.get("category", "unknown") if isinstance(record, dict) else "unknown"
            if len(stats["by_source"]) >= 128 and source not in stats["by_source"]:
                raise CorpusError("too many tokenization sources")
            tokens = next(iter(encode_document_tokens((text,), tokenizer, drop_empty=False)))
            data = tokens.tobytes()
            # The staging buffer is fixed; one already-bounded document may be larger.
            for offset in range(0, len(data), buffer_bytes):
                piece = data[offset:offset + buffer_bytes]
                if len(buffer) + len(piece) > buffer_bytes:
                    handle.write(buffer)
                    digest.update(buffer)
                    buffer.clear()
                buffer.extend(piece)
            stats["documents"] += 1
            stats["chars"] += len(text)
            stats["tokens"] += len(tokens)
            histogram_add(stats["character_length_histogram"], len(text))
            histogram_add(stats["token_length_histogram"], len(tokens))
            for field, label in (("by_source", source), ("by_category", category)):
                item = stats[field].setdefault(label, {"documents": 0, "tokens": 0})
                item["documents"] += 1
                item["tokens"] += len(tokens)
        if buffer:
            handle.write(buffer)
            digest.update(buffer)
        if handle.tell() != stats["tokens"] * 2:
            raise CorpusError("token stream length mismatch before publication")
    atomic_write(out_path, writer)
    stats.update(bytes=out_path.stat().st_size, sha256=digest.hexdigest())
    if stats["bytes"] != stats["tokens"] * 2:
        raise CorpusError(f"token output length mismatch: {out_path}")
    return stats


class Pipeline:
    def __init__(self, out: Path, content: dict, inputs: list[dict], *, runtime: dict | None = None,
                 temp_dir: Path | None = None, log=print):
        self.out, self.content, self.inputs, self.log = out, content, inputs, log
        self.runtime = merge(DEFAULT_RUNTIME, runtime or {})
        if not 1 <= self.runtime["sqlite_cache_mib"] <= 128 or not 2 <= self.runtime["buffer_bytes"] <= 16 * 1048576 or self.runtime["commit_every"] <= 0:
            raise CorpusError("invalid bounded runtime configuration")
        semantic = {"schema_version": 1, "content": content, "inputs": input_identity(inputs),
                    "implementation": implementation_identity()}
        self.plan = {"content_key": identity(semantic), "identity": semantic, "content": content, "inputs": inputs}
        owner = out / ".pipeline" / "plan.json"
        if owner.exists():
            previous = json.loads(owner.read_text())
            if previous["content_key"] != self.plan["content_key"]:
                raise CorpusError("pipeline input/content parameters changed; refusing reuse/overwrite, select a new --out")
        elif out.exists() and any(out.iterdir()):
            raise CorpusError("nonempty output has no pipeline plan; select a new empty --out")
        self.work = temp_dir / identity(str(out.resolve())) if temp_dir else out / ".pipeline"
        self.stages = Stages(out, self.work, self.runtime, log)
        if not owner.exists():
            write_json(owner, self.plan)

    def split(self):
        if (self.out / "dataset-manifest.json").exists():
            self.verify_published()
            self.log("[published] verified immutable dataset; no files changed")
            return
        norm, dedup, split = (self.work / f"{name}.sqlite" for name in ("normalize", "deduplicate", "split"))
        key = self.plan["content_key"]
        normalized = self.stages.run("normalize", key, lambda: normalize_sources(norm, self.inputs, self.content, self.runtime))
        normalized_path = Path(next(iter(normalized["files"])))
        deduplicated = self.stages.run("deduplicate", key, lambda: deduplicate(dedup, normalized_path, self.content, self.runtime))
        dedup_path = Path(next(iter(deduplicated["files"])))
        record = self.stages.run("split", key, lambda: write_splits(split, dedup_path, self.out, self.content, self.runtime))
        write_json(self.out / "dataset.json", {"stage": "split", "seed": self.content["seed"],
                                               "texts": record["stats"]["splits"], "plan_key": key})

    def verify_published(self, tokenizer: MiniLLMTokenizer | None = None):
        from .corpus_audit import audit_dataset
        report = audit_dataset(self.out)
        if not report["ok"]:
            raise CorpusError("published dataset verification failed: " + "; ".join(report["errors"]))
        manifest = json.loads((self.out / "dataset-manifest.json").read_text())
        if manifest["plan_key"] != self.plan["content_key"] or (tokenizer and tokenizer.fingerprint != manifest["tokenizer"]["fingerprint"]):
            raise CorpusError("published dataset plan/tokenizer mismatch; versions cannot be overwritten")
        return manifest

    def tokenize(self, tokenizer_path: Path | None = None, *, vocab_size=24576, max_tokenizer_chars=8000000):
        if (self.out / "dataset-manifest.json").exists():
            tokenizer = MiniLLMTokenizer.load(tokenizer_path) if tokenizer_path else None
            return self.verify_published(tokenizer)
        self.split()
        split_marker = json.loads((self.stages.control / "split.stage.json").read_text())
        key = self.plan["content_key"]
        target = self.out / "tokenizer.json"
        if tokenizer_path:
            loaded = MiniLLMTokenizer.load(tokenizer_path)
            tokenizer_key = identity([key, "load", loaded.fingerprint, file_info(tokenizer_path)])
            def prepare_tokenizer():
                with tokenizer_path.open("rb") as source:
                    atomic_write(target, lambda handle: shutil.copyfileobj(source, handle, length=1048576))
                return {"mode": "loaded", "fingerprint": loaded.fingerprint, "vocab_size": loaded.vocab_size}, [target]
        else:
            if max_tokenizer_chars <= 0 or not 264 <= vocab_size <= 65536:
                raise CorpusError("tokenizer character budget must be positive; vocabulary must fit uint16")
            tokenizer_key = identity([key, "train-only", vocab_size, max_tokenizer_chars, importlib.metadata.version("tokenizers")])
            def prepare_tokenizer():
                used = {"characters": 0, "documents": 0}
                def training_texts():
                    records = read_records(self.out / "train.text.jsonl")
                    try:
                        for record in records:
                            remaining = max_tokenizer_chars - used["characters"]
                            if remaining <= 0:
                                break
                            text = record["text"][:remaining]
                            used["characters"] += len(text)
                            used["documents"] += 1
                            yield text
                    finally:
                        records.close()
                tokenizer = MiniLLMTokenizer.train(training_texts(), vocab_size=vocab_size)
                atomic_write(target, lambda handle: handle.write(tokenizer._tok.to_str().encode("utf-8")))
                return {"mode": "trained_on_train_only", "fingerprint": tokenizer.fingerprint,
                        "vocab_size": tokenizer.vocab_size, "budget_characters": max_tokenizer_chars, **used,
                        "training_split_sha256": sha256(self.out / "train.text.jsonl")}, [target]
        tok_record = self.stages.run("tokenizer", tokenizer_key, prepare_tokenizer)
        tokenizer = MiniLLMTokenizer.load(target)
        results = {}
        for name in SPLITS:
            text_path, bin_path = self.out / f"{name}.text.jsonl", self.out / f"{name}.bin"
            tok_key = identity([key, tokenizer.fingerprint, split_marker["files"][str(text_path.resolve())]])
            def tokenize_one():
                records = read_records(text_path)
                try:
                    stats = tokenize_split(records, tokenizer, bin_path, buffer_bytes=self.runtime["buffer_bytes"])
                finally:
                    records.close()
                return stats, [bin_path]
            results[name] = self.stages.run("tokenize-" + name, tok_key, tokenize_one)["stats"]
        normalization = json.loads((self.stages.control / "normalize.stage.json").read_text())["stats"]
        deduplication = json.loads((self.stages.control / "deduplicate.stage.json").read_text())["stats"]
        quality_report = {"quality": normalization, "deduplication": deduplication, "selection": split_marker["stats"]}
        write_json(self.out / "quality-report.json", quality_report)
        write_json(self.out / "dataset.json", {"stage": "tokenized", "manifest": "dataset-manifest.json",
                                               "seed": self.content["seed"], "splits": results})
        files = {name: file_info(self.out / name) for name in ["tokenizer.json", "quality-report.json", "dataset.json",
                 *(f"{s}.{suffix}" for s in SPLITS for suffix in ("bin", "text.jsonl"))]}
        semantic = {"plan_key": key, "tokenizer_fingerprint": tokenizer.fingerprint, "files": files}
        try:
            revision = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            revision = None
        fractions = {}
        for field in ("by_source", "by_category"):
            counts = {}
            for stats in results.values():
                for label, values in stats[field].items():
                    counts[label] = counts.get(label, 0) + values["tokens"]
            total = sum(counts.values())
            fractions[field] = {label: {"tokens": n, "fraction": n / max(total, 1)} for label, n in sorted(counts.items())}
        manifest = {"schema_version": 1, "dataset_name": self.content["dataset_name"], "content_id": identity(semantic),
                    "identity": semantic, "plan_key": key, "seed": self.content["seed"], "content_config": self.content,
                    "runtime_config": self.runtime, "inputs": self.inputs, "sources": split_marker["stats"]["by_source"],
                    "split_fractions": self.content["split"], "splits": results, "files": files,
                    "tokenizer": {**tok_record["stats"], "file": "tokenizer.json", "fingerprint": tokenizer.fingerprint},
                    "quality": normalization, "deduplication": {"parameters": self.content["deduplication"], **deduplication},
                    "mixing": {"requested": self.content["mixing"], "selection": split_marker["stats"]["selection"],
                               "actual_token_fractions": fractions,
                               "actual_tokens_by_split": {name: {"by_source": s["by_source"], "by_category": s["by_category"]} for name, s in results.items()}},
                    "leakage": {"exact_cross_split_overlap": 0, "category_independent_cross_split_overlap": 0, "known_near_duplicate_cross_split_groups": 0,
                                "scope": "all exact hashes and recognized SimHash+Jaccard groups; unknown near duplicates are not exhaustively excluded"},
                    "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                    "code": {"git_revision": revision, "implementation_sha256": implementation_identity()}}
        # Audit the proposed manifest before its atomic publication commit marker.
        from .corpus_audit import audit_dataset
        audit = audit_dataset(self.out, manifest=manifest)
        if not audit["ok"]:
            raise CorpusError("prepublication audit FAILED: " + "; ".join(audit["errors"]))
        write_json(self.out / "dataset-manifest.json", manifest)
        self.log(f"[published] {self.out}: {manifest['content_id']}")
        return manifest


def guard_tokenizer_training(paths: list[Path], forbidden: list[Path]):
    """Recognize pipeline-held-out splits automatically, and explicit forbidden copies."""
    forbidden_paths = {p.resolve() for p in forbidden}
    forbidden_hashes = {sha256(p) for p in forbidden if p.is_file()}
    for path in paths:
        if path.name in ("val.text.jsonl", "test.text.jsonl") or path.resolve() in forbidden_paths or sha256(path) in forbidden_hashes:
            raise CorpusError(f"refusing tokenizer fitting on held-out data: {path}")
        marker = path.parent / ".pipeline" / "split.stage.json"
        if marker.exists():
            record = json.loads(marker.read_text())
            for filename, info in record["files"].items():
                if filename.endswith(".text.jsonl"):
                    verify_file(Path(filename), info)
            if path.resolve() != (path.parent / "train.text.jsonl").resolve():
                raise CorpusError(f"only train.text.jsonl may fit the tokenizer: {path}")
