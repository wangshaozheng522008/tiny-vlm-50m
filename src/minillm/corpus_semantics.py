"""Bounded, reproducible semantic review queues; heuristics are not truth labels.

No external judge or downloads. Curation requires explicit reviewer decisions;
by default only reviewed accepts are emitted, always to a new directory.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sqlite3
import sys
from collections import Counter
from decimal import Decimal
from pathlib import Path

from .assessment_io import publish_manifest, read_json, records, report_directory
from .corpus_quality import classify, raw_content_hash
from .corpus_utils import file_info, identity
from .data import stream_corpus_records
from .training_manifest import write_json

_EQUATION = re.compile(r"(?<![\w.])([0-9]{1,12}(?:\.[0-9]{1,6})?)\s*([+*/×÷−-])\s*([0-9]{1,12}(?:\.[0-9]{1,6})?)\s*(?:=|等于)\s*(-?[0-9]{1,12}(?:\.[0-9]{1,6})?)(?![\w.])")
_TASK = re.compile(r"(?:任务|指令|instruction)\s*[:：].*(?:输入|input)\s*[:：].*(?:答案|输出|output)\s*[:：]", re.I | re.S)
RULE_VERSION = "semantic-review-v1"


def semantic_flags(text, category="auto"):
    """Possible contradictions, including quoted/counterexample text: REVIEW only."""
    result = []
    if classify(text, category) == "code":
        return result
    # Width mapping here is diagnostic only; never rewrite original examples.
    observed = text.translate(str.maketrans({chr(i): chr(i-0xFEE0) for i in range(0xFF01, 0xFF5F)}))
    for match in _EQUATION.finditer(observed):
        # Do not extract a binary fragment from a longer expression or an
        # algebraic equality (e.g. a*b + c*d = 40, or 5x - 3 + 3 = 17 + 3).
        # These are explicitly outside this small literal rule's coverage.
        left = observed[max(0,match.start()-80):match.start()].rstrip()
        right = observed[match.end():match.end()+80].lstrip()
        math_context = '+-−*/×÷=^().'
        if ((left and (left[-1] in math_context or left[-1].isascii() and left[-1].isalnum()))
                or (right and (right[0] in math_context or right[0].isascii() and right[0].isalnum()))
                or (left.endswith(',') and len(left)>1 and left[-2].isdigit())
                or re.match(r',\d', right)  # a thousands separator, not the whole number
                or re.match(r'(?:余|余数|余下|又|\.\.\.|…|≈|约)', right)):
            continue
        a, op, b, claimed_text = match.groups()
        a, b, claimed = Decimal(a), Decimal(b), Decimal(claimed_text)
        if op in ("/", "÷"):
            # Plausible rounding to the displayed precision is not a mismatch.
            quantum = Decimal(1).scaleb(claimed.as_tuple().exponent)
            valid = b != 0 and abs(a/b-claimed) <= quantum/2
        else:
            expected = a+b if op == "+" else a-b if op in ("-", "−") else a*b
            valid = expected == claimed
        if not valid:
            result.append({"reason": "literal_arithmetic_mismatch", "evidence": match.group(),
                           "scope": "literal equation only; rounding, quotation and counterexamples require review"})
        if len(result) >= 20:
            break
    if _TASK.search(text):
        result.append({"reason": "flattened_instruction_record", "scope": "format/domain flag, not a falsehood judgement"})
    return result


def audit_semantics(paths, out, *, sample=200, seed=42):
    if type(sample) is not int or not 1 <= sample <= 10000:
        raise ValueError("sample must be in [1, 10000]")
    inputs = {str(Path(p).resolve()): file_info(Path(p)) for p in paths}
    if not inputs:
        raise ValueError("no input files")
    reasons, categories, sources = Counter(), Counter(), Counter()
    seen = rejected = duplicates = 0
    with report_directory(out) as directory:
        db_path = directory / "queue.sqlite"
        with sqlite3.connect(db_path) as db:
            db.execute("PRAGMA cache_size=-8192")
            db.execute("CREATE TABLE docs(id TEXT PRIMARY KEY, rank TEXT, flagged INT, row TEXT)")
            for path in sorted(inputs):
                for record in stream_corpus_records(path):
                    seen += 1
                    if "_issue" in record:
                        rejected += 1
                        reasons[record["_issue"]] += 1
                        continue
                    text = record["text"]
                    doc = raw_content_hash(text)
                    category = classify(text, record.get("category"))
                    flags = semantic_flags(text, category)
                    source = record.get("source", "unknown")
                    if not isinstance(source,str) or not source or len(source)>128 or not source.isprintable():
                        source = "unknown"
                    if source not in sources and len(sources)>=128:
                        raise ValueError("more than 128 source labels; group sources before audit")
                    row = {"id": doc, "text": text, "source": source,
                           "category": category, "license": record.get("license", "unknown"), "flags": flags}
                    rank = hashlib.sha256(f"{seed}:{doc}".encode()).hexdigest()
                    cursor = db.execute("INSERT OR IGNORE INTO docs VALUES(?,?,?,?)", (
                        doc, rank, bool(flags), json.dumps(row, ensure_ascii=False, sort_keys=True)))
                    if not cursor.rowcount:
                        duplicates += 1
                        continue
                    categories[category] += 1
                    sources[str(row["source"])] += 1
                    reasons.update(flag["reason"] for flag in flags)
                    if seen % 1000 == 0:
                        db.commit()
            db.execute("CREATE INDEX ranking ON docs(flagged DESC,rank,id)")
            total = db.execute("SELECT COUNT(*) FROM docs").fetchone()[0]
            flagged = db.execute("SELECT COUNT(*) FROM docs WHERE flagged=1").fetchone()[0]
            # Half the queue targets flags, half samples the remaining corpus.
            chosen = [r[0] for r in db.execute(
                "SELECT id FROM docs WHERE flagged=1 ORDER BY rank,id LIMIT ?", ((sample+1)//2,))]
            selected = set(chosen)
            for doc, in db.execute("SELECT id FROM docs ORDER BY rank,id"):
                if len(chosen) >= sample:
                    break
                if doc not in selected:
                    chosen.append(doc); selected.add(doc)
            with (directory / "review.jsonl").open("w", encoding="utf-8") as handle:
                for doc in chosen:
                    handle.write(db.execute("SELECT row FROM docs WHERE id=?", (doc,)).fetchone()[0] + "\n")
        db_path.unlink()
        if not total:
            raise ValueError("no valid documents for semantic review")
        for path, expected in inputs.items():
            if file_info(Path(path)) != expected:
                raise ValueError(f"input changed during audit: {path}")
        report = {"schema_version": 1, "status": "REVIEW_PENDING", "rules": RULE_VERSION,
                  "implementation_sha256": file_info(Path(__file__))["sha256"],
                  "seed": seed, "inputs": inputs, "records_seen": seen, "unique_documents": total,
                  "malformed_records": rejected, "duplicates": duplicates, "flagged_documents": flagged,
                  "flags_by_reason": dict(reasons), "by_category": dict(categories), "by_source": dict(sources),
                  "review_documents": len(chosen), "review_sampling": "half flag-prioritized, remainder hash-seeded sample; not unbiased error-rate estimator",
                  "coverage": {"literal_and_format_scan": "all valid records", "factual_correctness": "NOT_CHECKED",
                               "human_review": "NOT_RUN", "semantic_error_rate": None},
                  "limitations": "No general factual/logical verifier. Flags must be reviewed; absence of flags is not semantic approval."}
        write_json(directory / "semantic-audit.json", report)
        content = {"inputs": inputs, "implementation_sha256": report["implementation_sha256"],
                   "rules": RULE_VERSION, "seed": seed, "sample": sample}
        publish_manifest(directory, "review-manifest.json", {"schema_version": 1, **content,
                         "identity": content, "content_id": identity(content)})
    return report


def curate_review(review, decisions, out, *, include_unreviewed=False):
    review = Path(review)
    review_before = file_info(review / "review-manifest.json")
    decisions_before = file_info(Path(decisions))
    manifest = read_json(review / "review-manifest.json")
    if manifest.get("schema_version") != 1 or manifest.get("rules") != RULE_VERSION:
        raise ValueError("unsupported review manifest")
    if (identity(manifest.get("identity")) != manifest.get("content_id")
            or any(manifest.get(k) != v for k,v in manifest["identity"].items())):
        raise ValueError("review identity mismatch")
    for name, expected in manifest["files"].items():
        if Path(name).name != name or file_info(review / name) != expected:
            raise ValueError("review artifact changed/damaged")
    if not {"review.jsonl", "semantic-audit.json"} <= manifest["files"].keys():
        raise ValueError("incomplete review")
    queue = {}
    characters = 0
    for row in records(review / "review.jsonl"):
        characters += len(row["text"])
        if row["id"] in queue or row["id"] != raw_content_hash(row["text"]) or len(queue) >= 10000 or characters > 8*1048576:
            raise ValueError("invalid/oversized review queue; use smaller review batches")
        queue[row["id"]] = row
    actions = {}
    for row in records(decisions):
        doc = row.get("id")
        if (doc not in queue or doc in actions or row.get("action") not in ("accept", "exclude")
                or set(row) != {"id", "action", "reviewer", "reason"}
                or any(not isinstance(row.get(k), str) or not row[k].strip() or len(row[k])>limit
                       for k,limit in (("reviewer",200),("reason",4000)))):
            raise ValueError("invalid, stale, duplicate or unreasoned review decision")
        actions[doc] = row
    if actions.keys() != queue.keys():
        raise ValueError("every queued document requires a decision")
    for path, expected in manifest["inputs"].items():
        if file_info(Path(path)) != expected:
            raise ValueError(f"review source changed: {path}")
    counts = Counter()
    # Source deduplication remains the corpus pipeline's job, not an in-memory set.
    with report_directory(out) as directory:
        with (directory / "accepted.jsonl").open("w", encoding="utf-8") as handle:
            for path in sorted(manifest["inputs"]):
                for row in stream_corpus_records(path):
                    if "_issue" in row:
                        counts["malformed"] += 1; continue
                    action = actions.get(raw_content_hash(row["text"]))
                    if action and action["action"] == "exclude":
                        counts["excluded"] += 1; continue
                    if not action and not include_unreviewed:
                        counts["unreviewed_held_back"] += 1; continue
                    counts["reviewed_accepted" if action else "unreviewed_included"] += 1
                    handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        if file_info(Path(decisions)) != decisions_before or any(file_info(Path(p)) != f for p, f in manifest["inputs"].items()):
            raise ValueError("curation inputs changed")
        report = {"schema_version": 1, "status": "PARTIALLY_REVIEWED" if counts["unreviewed_included"] else "REVIEWED_SUBSET",
                  "counts": dict(counts), "review_manifest_sha256": file_info(review / "review-manifest.json")["sha256"],
                  "decisions": decisions_before, "include_unreviewed": include_unreviewed,
                  "scope": "review decisions, not a proof of factual correctness; preserve historical split exposure separately"}
        write_json(directory / "curation.json", report)
        # Preserve the exact reason/reviewer evidence with the curated source.
        with Path(decisions).open("rb") as source, (directory / "decisions.jsonl").open("wb") as target:
            shutil.copyfileobj(source, target, length=1048576)
        if (file_info(directory / "decisions.jsonl") != decisions_before
                or file_info(review / "review-manifest.json") != review_before
                or any(file_info(review / name) != f for name,f in manifest["files"].items())):
            raise ValueError("review/decision evidence changed during curation")
        publish_manifest(directory, "curation-manifest.json", report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    audit = sub.add_parser("audit")
    audit.add_argument("--input", nargs="+", required=True); audit.add_argument("--out", required=True)
    audit.add_argument("--sample", type=int, default=200); audit.add_argument("--seed", type=int, default=42)
    curate = sub.add_parser("curate")
    curate.add_argument("--review", required=True); curate.add_argument("--decisions", required=True)
    curate.add_argument("--out", required=True); curate.add_argument("--include-unreviewed", action="store_true")
    args = vars(parser.parse_args(argv)); command = args.pop("command")
    try:
        if command == "audit":
            args["paths"] = args.pop("input")
        result = audit_semantics(**args) if command == "audit" else curate_review(**args)
    except (ValueError, OSError, KeyError, TypeError, sqlite3.Error) as exc:
        print(f"FAIL semantic review: {exc}", file=sys.stderr); return 1
    print(json.dumps(result, ensure_ascii=False, indent=2)); return 0
