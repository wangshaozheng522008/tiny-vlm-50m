"""Prepare an immutable pilot source excluding known historical model exposure.

This is an experiment input guard. It does not alter the published corpus pipeline.
Near matching is approximate; unknown historical sources cannot be checked.
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

from .assessment_io import records, report_directory
from .corpus_quality import bands, classify, comparison_hash, normalise, shingles, similar, simhash
from .corpus_utils import connect, file_info, identity
from .training_manifest import write_json


def _text(record):
    text = record.get("text")
    if not isinstance(text, str):
        raise ValueError("historical exclusion requires a string text in every record")
    if len(text) > 200000:
        raise ValueError("historical exclusion record exceeds 200000 characters")
    return text


def _forms(text):
    # Check both original and current cleaning output, independently of declared category.
    return tuple(dict.fromkeys((text, normalise(text, classify(text)), normalise(text, "zh"))))


def prepare_source(source, historical, out, *, skip=0, take=0, radius=4, threshold=0.85, max_candidates=20000):
    source, out = Path(source), Path(out)
    historical = sorted({Path(p).resolve() for p in historical})
    if not historical or skip < 0 or take < 0 or not 0 <= radius <= 8 or not 0 < threshold <= 1 or max_candidates < 1:
        raise ValueError("invalid historical exclusion configuration")
    paths = [source, *historical]
    if any(p.resolve() == out.resolve() or out.resolve() in p.resolve().parents for p in paths):
        raise ValueError("source and historical files must be outside the output")
    inputs = [{"path": str(p.resolve()), **file_info(p)} for p in paths]
    counts = {"historical_records": 0, "candidates": 0, "excluded_exact": 0, "excluded_near": 0, "emitted": 0}
    with report_directory(out) as directory:
        index = directory / "historical.sqlite"
        db = connect(index)
        try:
            db.executescript("CREATE TABLE docs(id TEXT PRIMARY KEY,text TEXT,fp TEXT);"
                             "CREATE TABLE keys(hash TEXT PRIMARY KEY);"
                             "CREATE TABLE buckets(band INTEGER,value TEXT,id TEXT);"
                             "CREATE INDEX bucket_keys ON buckets(band,value);")
            for path in historical:
                for record in records(path):
                    text = _text(record)
                    counts["historical_records"] += 1
                    forms = _forms(text)
                    for form in forms:
                        db.execute("INSERT OR IGNORE INTO keys VALUES(?)", (comparison_hash(form),))
                    text = normalise(text, classify(text))
                    key = comparison_hash(text)
                    if db.execute("SELECT 1 FROM docs WHERE id=?", (key,)).fetchone():
                        continue
                    fp = simhash(shingles(text, "unknown"))
                    db.execute("INSERT INTO docs VALUES(?,?,?)", (key, text, str(fp)))
                    db.executemany("INSERT INTO buckets VALUES(?,?,?)", ((i, str(v), key) for i, v in bands(fp, radius)))
                    if counts["historical_records"] % 1000 == 0:
                        db.commit()
            db.commit()
            with (directory / "source.jsonl").open("w", encoding="utf-8") as handle:
                selected = itertools.islice(records(source), skip, skip + take if take else None)
                for record in selected:
                    text = _text(record)
                    counts["candidates"] += 1
                    forms = _forms(text)
                    if any(db.execute("SELECT 1 FROM keys WHERE hash=?", (comparison_hash(form),)).fetchone() for form in forms):
                        counts["excluded_exact"] += 1
                        continue
                    cleaned = normalise(text, classify(text))
                    features = shingles(cleaned, "unknown")
                    fp = simhash(features)
                    candidates = set()
                    for band, value in bands(fp, radius):
                        rows = db.execute("SELECT id FROM buckets WHERE band=? AND value=? LIMIT ?", (band, str(value), max_candidates + 1))
                        candidates.update(row[0] for row in rows)
                        if len(candidates) > max_candidates:
                            raise ValueError("historical near-match candidate budget exceeded; refusing truncated audit")
                    rejected = False
                    for key in sorted(candidates):
                        row = db.execute("SELECT text,fp FROM docs WHERE id=?", (key,)).fetchone()
                        if (fp ^ int(row["fp"])).bit_count() <= radius and similar(features, shingles(row["text"], "unknown"), threshold):
                            rejected = True
                            break
                    if rejected:
                        counts["excluded_near"] += 1
                        continue
                    handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
                    counts["emitted"] += 1
                    if counts["candidates"] % 10000 == 0:
                        print(json.dumps(counts), flush=True)
        finally:
            db.close()
            for path in directory.glob("historical.sqlite*"):
                path.unlink()
        if not counts["emitted"]:
            raise ValueError("historical exclusion produced no source records")
        if any(file_info(p) != {k: item[k] for k in ("bytes", "sha256")} for p, item in zip(paths, inputs)):
            raise ValueError("source or historical corpus changed during exclusion")
        report = {"schema_version": 1, "inputs": inputs, "selection": {"skip_records": skip, "take_records": take},
                  "algorithm": {"exact": "category-neutral whitespace-normalized SHA256 before/after cleaning",
                                "near": "64-bit full-document character 4-gram SimHash and Jaccard confirmation, including code",
                                "hamming_distance": radius, "jaccard_threshold": threshold, "max_candidates": max_candidates},
                  "counts": counts, "source": file_info(directory / "source.jsonl"),
                  "implementation_sha256": file_info(Path(__file__))["sha256"],
                  "scope": "known supplied historical records only; no exhaustive semantic, substring or unknown-source contamination guarantee"}
        report["content_id"] = identity(report)
        write_json(directory / "historical-exclusion.json", report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--historical", required=True, type=Path, nargs="+")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--skip-records", type=int, default=0)
    parser.add_argument("--take-records", type=int, default=0)
    args = parser.parse_args(argv)
    try:
        report = prepare_source(args.input, args.historical, args.out, skip=args.skip_records, take=args.take_records)
    except (ValueError, OSError, KeyError) as exc:
        print(f"FAIL historical exclusion: {exc}")
        return 1
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0
