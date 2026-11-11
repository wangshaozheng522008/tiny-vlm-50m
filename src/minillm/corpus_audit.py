"""Streaming integrity and split-isolation audit of a published corpus version."""
from __future__ import annotations

import json
import sqlite3
import tempfile
from pathlib import Path

import numpy as np

from .corpus import SPLITS, input_identity, read_records
from .corpus_quality import CATEGORIES, content_hash
from .corpus_utils import CorpusError, connect, histogram_add, identity, verify_file
from .tokenizer import MiniLLMTokenizer


def audit_dataset(directory: Path, *, manifest: dict | None = None, tokenizer_path: Path | None = None,
                  retokenize: bool = False, verify_inputs: bool = False) -> dict:
    directory = Path(directory)
    report = {"ok": False, "errors": [], "checks": {}, "splits": {}, "exact_overlap": {},
              "known_near_duplicate_cross_split_groups": None}
    def error(message):
        if len(report["errors"]) < 20:
            report["errors"].append(str(message))
    try:
        if manifest is None:
            manifest = json.loads((directory / "dataset-manifest.json").read_text())
        if manifest["schema_version"] != 1 or manifest["identity"]["files"] != manifest["files"]:
            raise CorpusError("invalid dataset schema or inconsistent identity/files")
        if identity(manifest["identity"]) != manifest["content_id"]:
            raise CorpusError("dataset content identity mismatch")
        plan = {"schema_version": 1, "content": manifest["content_config"], "inputs": input_identity(manifest["inputs"]),
                "implementation": manifest["code"]["implementation_sha256"]}
        if identity(plan) != manifest["plan_key"] or manifest["identity"]["plan_key"] != manifest["plan_key"]:
            raise CorpusError("dataset plan identity mismatch")
        required = {"tokenizer.json", "quality-report.json", "dataset.json", *(f"{s}.{suffix}" for s in SPLITS for suffix in ("bin", "text.jsonl"))}
        if set(manifest["files"]) != required:
            raise CorpusError("missing/unexpected published artifacts")
        for filename, info in manifest["files"].items():
            if Path(filename).name != filename:
                raise CorpusError("artifact paths must be local filenames")
            verify_file(directory / filename, info)
        report["checks"]["artifact_hashes"] = "full"
        summary = json.loads((directory / "dataset.json").read_text())
        quality = json.loads((directory / "quality-report.json").read_text())
        if summary["splits"] != manifest["splits"] or quality["quality"] != manifest["quality"] or quality["selection"]["selection"] != manifest["mixing"]["selection"]:
            raise CorpusError("generation report/manifest statistics mismatch")
        if quality["deduplication"] != {k: v for k, v in manifest["deduplication"].items() if k != "parameters"}:
            raise CorpusError("deduplication report/manifest mismatch")
        tokenizer = MiniLLMTokenizer.load(directory / "tokenizer.json")
        if not 264 <= tokenizer.vocab_size <= 65536:
            raise CorpusError("tokenizer vocabulary does not fit uint16 corpus format")
        if tokenizer.fingerprint != manifest["tokenizer"]["fingerprint"] or tokenizer.fingerprint != manifest["identity"]["tokenizer_fingerprint"] or tokenizer.vocab_size != manifest["tokenizer"]["vocab_size"]:
            raise CorpusError("tokenizer fingerprint/vocabulary mismatch")
        if tokenizer_path and MiniLLMTokenizer.load(tokenizer_path).fingerprint != tokenizer.fingerprint:
            raise CorpusError("supplied tokenizer fingerprint mismatch")
        report["checks"]["tokenizer_fingerprint"] = "full"
        if verify_inputs:
            for item in manifest["inputs"]:
                verify_file(Path(item["path"]), {k: item[k] for k in ("bytes", "sha256")})
        report["checks"]["raw_input_hashes"] = "full" if verify_inputs else "not checked; original snapshot hashes recorded"
        with tempfile.TemporaryDirectory(prefix="minillm-audit-") as temporary:
            db = connect(Path(temporary) / "audit.sqlite")
            try:
                db.executescript("CREATE TABLE records(hash TEXT,split TEXT,parent TEXT,grp TEXT); CREATE INDEX hashes ON records(hash,split); CREATE INDEX groups_ ON records(grp,split); CREATE INDEX parents ON records(parent,split);")
                for split in SPLITS:
                    stats = {"documents": 0, "chars": 0, "tokens": 0, "by_source": {}, "by_category": {},
                             "character_length_histogram": {}, "unknown_license_documents": 0}
                    for record in read_records(directory / f"{split}.text.jsonl"):
                        category, source = record["category"], record["source"]
                        if category not in CATEGORIES or len(stats["by_source"]) >= 128 and source not in stats["by_source"]:
                            raise CorpusError("invalid category or excessive source cardinality")
                        digest = content_hash(record["text"], category)
                        if digest != record["id"]:
                            error(f"{split}: normalized text ID mismatch")
                        db.execute("INSERT INTO records VALUES(?,?,?,?)", (digest, split, record["parent_id"], record["group_id"]))
                        stats["documents"] += 1
                        stats["chars"] += len(record["text"])
                        histogram_add(stats["character_length_histogram"], len(record["text"]))
                        for field, key in (("by_source", source), ("by_category", category)):
                            stats[field][key] = stats[field].get(key, 0) + 1
                        stats["unknown_license_documents"] += record["license"] == "unknown"
                        if stats["documents"] % 1000 == 0:
                            db.commit()
                    with (directory / f"{split}.bin").open("rb") as handle:
                        while block := handle.read(1048576):
                            if len(block) % 2:
                                raise CorpusError(f"{split}: odd-sized uint16 BIN")
                            ids = np.frombuffer(block, dtype="<u2")
                            if len(ids) and int(ids.max()) >= tokenizer.vocab_size:
                                raise CorpusError(f"{split}: token outside tokenizer vocabulary")
                            stats["tokens"] += len(ids)
                    expected = manifest["splits"][split]
                    if {k: expected[k] for k in ("bytes", "sha256")} != manifest["files"][f"{split}.bin"]:
                        error(f"{split}: streaming checksum differs from artifact checksum")
                    for key in ("documents", "chars", "tokens", "character_length_histogram"):
                        if stats[key] != expected[key]:
                            error(f"{split}: {key} statistics mismatch")
                    for field in ("by_source", "by_category"):
                        if stats[field] != {key: value["documents"] for key, value in expected[field].items()}:
                            error(f"{split}: {field} distribution mismatch")
                        if sum(v["tokens"] for v in expected[field].values()) != stats["tokens"]:
                            error(f"{split}: recorded token distribution does not sum to total")
                    if retokenize:
                        histogram = {}
                        with (directory / f"{split}.bin").open("rb") as handle:
                            for record in read_records(directory / f"{split}.text.jsonl"):
                                ids = tokenizer.encode_document(record["text"])
                                encoded = np.asarray(ids, dtype="<u2").tobytes()
                                if handle.read(len(encoded)) != encoded:
                                    raise CorpusError(f"{split}: retokenization differs from BIN")
                                histogram_add(histogram, len(ids))
                            if handle.read(1) or histogram != expected["token_length_histogram"]:
                                raise CorpusError(f"{split}: token-length distribution mismatch")
                    stats["token_length_histogram"] = expected["token_length_histogram"]
                    report["splits"][split] = stats
                for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
                    n = db.execute("SELECT COUNT(DISTINCT a.hash) FROM records a JOIN records b ON a.hash=b.hash WHERE a.split=? AND b.split=?", (a, b)).fetchone()[0]
                    report["exact_overlap"][a + "/" + b] = n
                    if n:
                        error(f"exact cross-split leakage: {a}/{b} ({n})")
                duplicate_count = db.execute("SELECT COALESCE(SUM(n-1),0) FROM (SELECT COUNT(*) n FROM records GROUP BY hash HAVING n>1)").fetchone()[0]
                if duplicate_count:
                    error(f"remaining exact duplicates: {duplicate_count}")
                groups = db.execute("SELECT COUNT(*) FROM (SELECT grp FROM records GROUP BY grp HAVING COUNT(DISTINCT split)>1)").fetchone()[0]
                parents = db.execute("SELECT COUNT(*) FROM (SELECT parent FROM records GROUP BY parent HAVING COUNT(DISTINCT split)>1 OR COUNT(DISTINCT grp)>1)").fetchone()[0]
                report["known_near_duplicate_cross_split_groups"] = groups
                report["parent_isolation_violations"] = parents
                if groups or parents:
                    error(f"cross-split group/parent leakage: {groups}/{parents}")
                report["remaining_exact_duplicates"] = duplicate_count
            finally:
                db.close()
        report["checks"].update(token_range="full", exact_overlap="full disk-indexed normalized hashes",
                                 known_group_isolation="full declared groups/parents",
                                 near_duplicates={"scope": "recognized groups only; independent all-pairs near search not performed",
                                                  "algorithm": "64-bit full-character-ngram SimHash + Jaccard confirmation",
                                                  "parameters": manifest["deduplication"]["parameters"], "type": "approximate"},
                                 retokenization="full" if retokenize else "not requested; token-length/source token statistics are generation records")
        reasons = {}
        for row in manifest["quality"]["by_source_reason"]:
            reasons[row["reason"]] = reasons.get(row["reason"], 0) + row["n"]
        excluded = {"records_seen", "accepted_unique", "duplicates_exact", "excluded_source_files"}
        report["quality"] = {"records_seen": manifest["quality"]["records_seen"], "by_reason": reasons,
                             "filter_rate": sum(n for reason, n in reasons.items() if reason not in excluded) / max(manifest["quality"]["records_seen"], 1),
                             "exact_duplicate_rate": manifest["deduplication"]["duplicates_exact"] / max(manifest["quality"]["records_seen"], 1),
                             "near_duplicate_rate": manifest["deduplication"]["duplicates_near"] / max(manifest["deduplication"]["parents_before_near"], 1)}
        report["content_id"] = manifest["content_id"]
        report["mixing"] = manifest["mixing"]
    except (OSError, ValueError, KeyError, TypeError, sqlite3.Error, CorpusError) as exc:
        error(exc)
    report["ok"] = not report["errors"]
    return report
