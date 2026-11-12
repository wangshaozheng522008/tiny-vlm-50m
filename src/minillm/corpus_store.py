"""SQLite normalization, full-document deduplication and grouped fragmentation."""
from __future__ import annotations

import math
import hashlib
from pathlib import Path

from .corpus_quality import (CATEGORIES, bands, classify, content_hash, fragments, normalise,
                             quality_issue, shingles, simhash, similar, raw_content_hash, comparison_hash)
from .corpus_utils import CorpusError, connect, database_output, identity, json_bytes, verify_file
from .data import stream_corpus_records

SCHEMA = """
CREATE TABLE docs(id TEXT PRIMARY KEY, text TEXT NOT NULL, source TEXT, category TEXT, language TEXT,
 license TEXT, priority INTEGER, metadata TEXT, near_parent TEXT, split_parent TEXT, near_root TEXT, sample_key TEXT, content_key TEXT);
CREATE TABLE families(family TEXT, id TEXT, PRIMARY KEY(family,id));
CREATE TABLE counts(source TEXT, reason TEXT, n INTEGER, PRIMARY KEY(source,reason));
CREATE TABLE edges(kind TEXT, a TEXT, b TEXT, PRIMARY KEY(kind,a,b));
"""


def count(db, source, reason, n=1):
    db.execute("INSERT INTO counts VALUES(?,?,?) ON CONFLICT(source,reason) DO UPDATE SET n=n+excluded.n", (source, reason, n))


def find(db, table: str, column: str, key: str) -> str:
    root = key
    while True:
        parent = db.execute(f"SELECT {column} FROM {table} WHERE id=?", (root,)).fetchone()[0]
        if parent == root:
            break
        root = parent
    if key != root:
        db.execute(f"UPDATE {table} SET {column}=? WHERE id=?", (root, key))
    return root


def union(db, table, column, a, b):
    a, b = find(db, table, column, a), find(db, table, column, b)
    if a != b:
        lower, upper = sorted((a, b))
        db.execute(f"UPDATE {table} SET {column}=? WHERE id=?", (lower, upper))


def normalize_sources(path: Path, inputs: list[dict], content: dict, runtime: dict):
    quality = content["quality"]
    seen = 0
    source_names = set()
    with database_output(path, cache_mib=runtime["sqlite_cache_mib"]) as db:
        db.executescript(SCHEMA)
        for item in inputs:
            source = item["source"]
            if not item["enabled"]:
                count(db, source, "excluded_source_files")
                continue
            records = stream_corpus_records(item["path"], max_record_bytes=quality["max_record_bytes"])
            try:
                for record in records:
                    if content["max_records"] and seen >= content["max_records"]:
                        break
                    seen += 1
                    source = item["source"] if item["source"] != "unknown" else record.get("source", "unknown")
                    if not isinstance(source, str) or not source.strip() or len(source) > 128 or not source.isprintable():
                        source = "unknown"
                    source_names.add(source)
                    if len(source_names) > 128:
                        raise CorpusError("more than 128 sources; supply a source manifest to aggregate source identifiers")
                    count(db, source, "records_seen")
                    if "_issue" in record:
                        count(db, source, record["_issue"])
                        continue
                    license_ = item["license"] if item["license"] != "unknown" else record.get("license", "unknown")
                    if not isinstance(license_, str) or not license_.strip() or len(license_) > 128 or not license_.isprintable():
                        license_ = "unknown"
                    allowed = content["license_policy"]["allow"]
                    if (content["license_policy"]["exclude_unknown"] and license_ == "unknown") or (allowed and license_ not in allowed):
                        count(db, source, "license_excluded")
                        continue
                    hint = record.get("category", item["category"])
                    category = classify(record["text"], hint)
                    from .text_policy import IDENTITY, apply_text_policy
                    text = apply_text_policy(record["text"], content.get("text_policy", IDENTITY), category=category)
                    text = normalise(text, category, strip_html=quality["strip_html"])
                    category = classify(text, hint)
                    issue = quality_issue(text, category, quality)
                    if issue:
                        count(db, source, issue)
                        continue
                    doc = raw_content_hash(record["text"])
                    language = classify(text)
                    declared = hint if isinstance(hint, str) and hint in CATEGORIES else "unknown"
                    metadata = json_bytes({"source": source, "category": category, "declared_category": declared,
                                           "language": language, "license": license_,
                                           "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest()}).decode()
                    old = db.execute("SELECT priority,metadata,category FROM docs WHERE id=?", (doc,)).fetchone()
                    values = (doc, text, source, category, language, license_, item["priority"], metadata, doc, doc, doc,
                              identity([content["seed"], "sample", doc]), content_hash(text, category))
                    if old is None:
                        db.execute("INSERT INTO docs VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)", values)
                        count(db, source, "accepted_unique")
                    else:
                        count(db, source, "duplicates_exact")
                        if category != old["category"]:
                            count(db, source, "duplicates_cross_category")
                        # Same original: retain an explicit code version before source priority,
                        # so a mislabeled prose copy cannot remove meaningful indentation.
                        if (-int(category == "code"), -item["priority"], metadata) < (-int(old["category"] == "code"), -old["priority"], old["metadata"]):
                            db.execute("UPDATE docs SET text=?,source=?,category=?,language=?,license=?,priority=?,metadata=?,content_key=? WHERE id=?", (*values[1:8], values[-1], doc))
                    family = record.get("parent_id")
                    if Path(item["path"]).suffix.lower() != ".jsonl":
                        family = "txt-file:" + item["sha256"]
                    if family is not None:
                        if not isinstance(family, (str, int)) or len(str(family)) > 512:
                            raise CorpusError("parent_id must be a string/integer of at most 512 characters")
                        db.execute("INSERT OR IGNORE INTO families VALUES(?,?)", (identity([source, str(family)]), doc))
                    if seen % runtime["commit_every"] == 0:
                        db.commit()
            finally:
                records.close()
            verify_file(Path(item["path"]), {k: item[k] for k in ("bytes", "sha256")})
        documents = db.execute("SELECT COUNT(*) FROM docs").fetchone()[0]
        if documents == 0:
            raise CorpusError("no documents survived normalization/quality/license filtering")
        stats = {"records_seen": seen, "documents": documents,
                 "by_source_reason": [dict(r) for r in db.execute("SELECT * FROM counts ORDER BY source,reason")]}
    return stats, [path]


def near_deduplicate(db, table: str, cfg: dict, on_edge):
    stats = {"documents": 0, "candidates": 0, "near_edges": 0, "max_candidates_observed": 0}
    if cfg["method"] == "exact":
        return stats
    db.execute("DROP TABLE IF EXISTS buckets")
    db.execute("CREATE TABLE buckets(band INTEGER, bucket TEXT, id TEXT, fingerprint TEXT, PRIMARY KEY(band,bucket,id))")
    query = ("SELECT d.id,d.text,d.category FROM docs d JOIN exact_representatives e ON e.id=d.id ORDER BY d.id"
             if table == "docs" else f"SELECT id,text,category FROM {table} ORDER BY id")
    for row in db.execute(query):
        if row["category"] == "code" and not cfg["near_code"]:
            continue
        features = shingles(row["text"], row["category"], cfg["ngram"])
        fingerprint = simhash(features)
        clauses, parameters = [], []
        for band, bucket in bands(fingerprint, cfg["hamming_distance"]):
            clauses.append("(band=? AND bucket=?)")
            parameters.extend((band, f"{bucket:016x}"))
        query = "SELECT id,MIN(fingerprint) fingerprint FROM buckets WHERE " + " OR ".join(clauses)
        query += " GROUP BY id ORDER BY id LIMIT ?"
        candidates = db.execute(query, (*parameters, cfg["max_candidates"] + 1))
        compared = 0
        for candidate in candidates:
            compared += 1
            if compared > cfg["max_candidates"]:
                raise CorpusError("near-duplicate candidate budget exceeded; no dataset published. Increase max_candidates or use a stricter Hamming radius")
            stats["candidates"] += 1
            if (fingerprint ^ int(candidate["fingerprint"], 16)).bit_count() > cfg["hamming_distance"]:
                continue
            other = db.execute(f"SELECT text,category FROM {table} WHERE id=?", (candidate["id"],)).fetchone()
            if similar(features, shingles(other["text"], other["category"], cfg["ngram"]), cfg["threshold"]):
                on_edge(row["id"], candidate["id"])
                db.execute("INSERT OR IGNORE INTO edges VALUES(?,?,?)", (table, row["id"], candidate["id"]))
                stats["near_edges"] += 1
        stats["max_candidates_observed"] = max(stats["max_candidates_observed"], compared)
        stats["documents"] += 1
        db.executemany("INSERT INTO buckets VALUES(?,?,?,?)", [(i, f"{b:016x}", row["id"], f"{fingerprint:016x}") for i, b in bands(fingerprint, cfg["hamming_distance"])])
    db.execute("DROP TABLE buckets")
    return stats


def deduplicate(path: Path, normalized: Path, content: dict, runtime: dict):
    cfg = content["deduplication"]
    with database_output(path, source=normalized, cache_mib=runtime["sqlite_cache_mib"]) as db:
        def doc_edge(a, b):
            union(db, "docs", "near_parent", a, b)
            union(db, "docs", "split_parent", a, b)
        # Original identities are already category-neutral. Apply the existing
        # category-specific normalized exact rule only after choosing that version.
        db.execute("CREATE INDEX docs_content ON docs(content_key,id)")
        for group in db.execute("SELECT content_key,MIN(id) first FROM docs GROUP BY content_key HAVING COUNT(*)>1"):
            for member in db.execute("SELECT id FROM docs WHERE content_key=? ORDER BY id", (group["content_key"],)):
                doc_edge(group["first"], member["id"])
        for row in db.execute("SELECT id FROM docs ORDER BY id"):
            db.execute("UPDATE docs SET near_root=? WHERE id=?", (find(db, "docs", "near_parent", row["id"]), row["id"]))
        db.execute("CREATE TABLE exact_representatives(id TEXT PRIMARY KEY)")
        db.execute("INSERT INTO exact_representatives SELECT id FROM (SELECT id,ROW_NUMBER() OVER(PARTITION BY near_root ORDER BY (category='code') DESC,priority DESC,id) rn FROM docs) WHERE rn=1")
        before_near = db.execute("SELECT COUNT(*) FROM exact_representatives").fetchone()[0]
        near = near_deduplicate(db, "docs", cfg, doc_edge)
        for row in db.execute("SELECT family, MIN(id) first FROM families GROUP BY family ORDER BY family"):
            for member in db.execute("SELECT id FROM families WHERE family=? ORDER BY id", (row["family"],)):
                union(db, "docs", "split_parent", row["first"], member["id"])
        for row in db.execute("SELECT id FROM docs ORDER BY id"):
            root = find(db, "docs", "near_parent", row["id"])
            db.execute("UPDATE docs SET near_root=? WHERE id=?", (root, row["id"]))
        db.execute("CREATE TABLE representatives(id TEXT PRIMARY KEY)")
        db.execute("INSERT INTO representatives SELECT id FROM (SELECT id,sample_key,ROW_NUMBER() OVER(PARTITION BY near_root ORDER BY (category='code') DESC,priority DESC,id) rn FROM docs) WHERE rn=1 ORDER BY sample_key,id LIMIT ?", (content["sample"] or -1,))
        db.executescript("""CREATE TABLE fragments(id TEXT PRIMARY KEY, parent_id TEXT, text TEXT, source TEXT,
          category TEXT, language TEXT, license TEXT, priority INTEGER, near_parent TEXT, near_root TEXT,
          estimated INTEGER, order_key TEXT);
          CREATE INDEX docs_near ON docs(near_root,priority,id);
          CREATE TABLE comparison_keys(hash TEXT,id TEXT,PRIMARY KEY(hash,id));""")
        duplicate_fragments = 0
        for row in db.execute("SELECT d.* FROM docs d JOIN representatives r ON r.id=d.id ORDER BY d.id"):
            for text in fragments(row["text"], content["quality"]["max_chars"]):
                if not text.strip():
                    continue
                key = content_hash(text, row["category"])
                previous = db.execute("SELECT parent_id,priority,source FROM fragments WHERE id=?", (key,)).fetchone()
                if previous:
                    duplicate_fragments += 1
                    union(db, "docs", "split_parent", row["id"], previous["parent_id"])
                    if (-row["priority"], row["source"], row["id"]) >= (-previous["priority"], previous["source"], previous["parent_id"]):
                        continue
                estimate = max(1, math.ceil(len(text) / content["mixing"]["chars_per_token"][row["category"]]))
                db.execute("INSERT OR REPLACE INTO fragments VALUES(?,?,?,?,?,?,?,?,?,?,?,?)", (
                    key, row["id"], text, row["source"], row["category"], row["language"], row["license"],
                    row["priority"], key, key, estimate, identity([content["shuffle_seed"], "order", key])))
                db.execute("INSERT OR IGNORE INTO comparison_keys VALUES(?,?)", (comparison_hash(text), key))
        # Conservative grouping across categories does not delete code variants.
        for group in db.execute("SELECT hash,MIN(id) first FROM comparison_keys GROUP BY hash HAVING COUNT(*)>1"):
            first = db.execute("SELECT parent_id FROM fragments WHERE id=?", (group["first"],)).fetchone()[0]
            for member in db.execute("SELECT f.parent_id FROM fragments f JOIN comparison_keys c ON c.id=f.id WHERE c.hash=? ORDER BY f.id", (group["hash"],)):
                union(db, "docs", "split_parent", first, member["parent_id"])
        def fragment_edge(a, b):
            union(db, "fragments", "near_parent", a, b)
            parents = [db.execute("SELECT parent_id FROM fragments WHERE id=?", (key,)).fetchone()[0] for key in (a, b)]
            union(db, "docs", "split_parent", *parents)
        fragment_near = near_deduplicate(db, "fragments", cfg, fragment_edge)
        for row in db.execute("SELECT id FROM fragments ORDER BY id"):
            db.execute("UPDATE fragments SET near_root=? WHERE id=?", (find(db, "fragments", "near_parent", row["id"]), row["id"]))
        db.execute("CREATE TABLE eligible(id TEXT PRIMARY KEY)")
        db.execute("INSERT INTO eligible SELECT id FROM (SELECT id,ROW_NUMBER() OVER(PARTITION BY near_root ORDER BY priority DESC,id) rn FROM fragments) WHERE rn=1")
        parents = db.execute("SELECT COUNT(*) FROM docs").fetchone()[0]
        groups = db.execute("SELECT COUNT(DISTINCT near_root) FROM docs").fetchone()[0]
        total_fragments = db.execute("SELECT COUNT(*) FROM fragments").fetchone()[0]
        eligible = db.execute("SELECT COUNT(*) FROM eligible").fetchone()[0]
        stats = {"documents": eligible, "parents_before_near": before_near, "parents_after_near": groups,
                 "parents_selected": db.execute("SELECT COUNT(*) FROM representatives").fetchone()[0],
                 "duplicates_raw_exact": db.execute("SELECT COALESCE(SUM(n),0) FROM counts WHERE reason='duplicates_exact'").fetchone()[0],
                 "duplicates_normalized_exact": parents - before_near,
                 "duplicates_exact": db.execute("SELECT COALESCE(SUM(n),0) FROM counts WHERE reason='duplicates_exact'").fetchone()[0] + parents - before_near,
                 "duplicates_near": before_near - groups, "fragment_duplicates_exact": duplicate_fragments,
                 "fragment_duplicates_near": total_fragments - eligible, "parent_search": near, "fragment_search": fragment_near}
    return stats, [path]
