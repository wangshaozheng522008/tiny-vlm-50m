#!/usr/bin/env python3
"""Offline synthetic performance benchmark for the corpus-v4 stages.

Default is intentionally small (1,000 documents).  The 100k/1m runs must be
requested explicitly with ``--documents`` and are never part of normal CI.

Examples:
    python scripts/benchmark_corpus_v4.py --documents 1000 \
        --out reports/bench-small --seed 42

    python scripts/benchmark_corpus_v4.py --documents 100000 \
        --out reports/bench-100k --seed 42 --stages ingest,quality,dedup,split

    # 1m may be limited to ingest/dedup/split; no tokenizer is ever trained.
    python scripts/benchmark_corpus_v4.py --documents 1000000 \
        --out reports/bench-1m --seed 42 --stages ingest,quality,dedup,split
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from minillm.corpus_benchmark import DEFAULT_DOCUMENTS, STAGES, run_benchmark  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/corpus_v4.yaml")
    parser.add_argument("--documents", type=int, default=DEFAULT_DOCUMENTS,
                        help="synthetic document count; large values are an explicit opt-in")
    parser.add_argument("--out", required=True, help="new output directory (never overwritten)")
    parser.add_argument("--input-dir", default=None,
                        help="optional new directory for generated JSONL sources")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stages", default="ingest,quality,dedup,split",
                        help=f"comma-separated subset of {','.join(STAGES)}; prerequisites run as needed")
    args = parser.parse_args(argv)
    stages = tuple(part.strip() for part in args.stages.split(",") if part.strip())
    try:
        report = run_benchmark(
            args.config, Path(args.out), documents=args.documents, seed=args.seed,
            input_dir=Path(args.input_dir) if args.input_dir else None, stages=stages,
            log=lambda message: print(message, file=sys.stderr))
    except Exception as exc:
        report_path = Path(args.out) / "benchmark-report.json"
        print(f"benchmark FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        print(f"failure report: {report_path}", file=sys.stderr)
        return 1
    print(json.dumps({
        "status": report["status"],
        "documents": report["parameters"]["documents"],
        "output": report["parameters"]["out"],
        "input_sha256": report["corpus"]["input_sha256"],
        "retained_documents": report["metrics"]["retained_documents"],
        "deleted_documents": report["metrics"]["deleted_documents"],
        "total_seconds": report["metrics"]["total_seconds"],
        "peak_rss_bytes": report["metrics"]["peak_rss_bytes"],
        "sqlite_bytes": report["metrics"]["sqlite_bytes"],
        "temp_peak_bytes": report["metrics"]["temp_peak_bytes"],
        "leakage": report["metrics"]["leakage"],
        "report": str(Path(args.out) / "benchmark-report.json"),
        "summary": str(Path(args.out) / "benchmark-summary.md"),
    }, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
