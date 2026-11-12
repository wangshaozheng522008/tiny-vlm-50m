#!/usr/bin/env python3
"""Plan, build and audit the MiniLLM corpus v4 dataset.

Examples:
    # Read the formal readiness report (no download, no fabricated source).
    python scripts/plan_corpus_v4.py --config configs/corpus_v4.yaml --status-only

    # End-to-end synthetic fixture demonstration (DEMO_ONLY, never trainable).
    python scripts/plan_corpus_v4.py --config configs/corpus_v4.yaml \
        --demo --out reports/corpus-v4-demo

    # After a human fills configs/corpus_v4.yaml and selects a tokenizer:
    python scripts/plan_corpus_v4.py --config configs/corpus_v4.yaml \
        --out artifacts/corpus/v4 --stage all

The script never downloads data and never approves a source automatically.  A
formal build without approved sources stops at a NOT_READY status report.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from minillm.corpus_utils import CorpusError, write_json  # noqa: E402
from minillm.corpus_v4 import (CorpusV4Pipeline, build_registry, inventory,  # noqa: E402
                               load_config, status_report, verify_frozen)

STAGES = ("ingest", "quality", "dedup", "split", "tokenizers", "evaluate",
          "tokenize", "contamination", "budget", "audit", "freeze", "all")


def _print(value):
    print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))


def _registry(config: dict, demo: bool):
    base = Path.cwd()
    if demo:
        manifest_path = (base / config["demo"]["sources_manifest"]).resolve()
        if not manifest_path.is_file():
            raise CorpusError(f"demo source manifest not found: {manifest_path}")
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
        sources, source_base = raw.get("sources", []), manifest_path.parent
    else:
        sources, source_base = config["sources"], base
    registry = build_registry(sources, demo=demo, license_policy=config["license_policy"])
    return registry, inventory(registry, source_base)


def _stage_result(pipeline: CorpusV4Pipeline, stage: str):
    if stage == "ingest":
        return pipeline.ingest()
    if stage == "quality":
        return pipeline.quality()
    if stage == "dedup":
        return pipeline.deduplicate()
    if stage == "split":
        return pipeline.split()
    if stage == "tokenizers":
        return pipeline.train_tokenizers()
    if stage == "evaluate":
        return pipeline.evaluate_tokenizers()
    if stage == "tokenize":
        return pipeline.tokenize()
    if stage == "contamination":
        return pipeline.contamination()
    if stage == "budget":
        return pipeline.budget()
    if stage == "audit":
        return pipeline.audit()
    if stage == "freeze":
        return pipeline.finalize()
    raise ValueError(stage)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__
                                      , formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/corpus_v4.yaml")
    parser.add_argument("--demo", action="store_true",
                        help="run the synthetic fixture demonstration and mark every artifact DEMO_ONLY")
    parser.add_argument("--out", default=None,
                        help="new output directory; existing nonempty directories are never overwritten")
    parser.add_argument("--stage", choices=STAGES, default="all")
    parser.add_argument("--status-only", action="store_true")
    parser.add_argument("--tokenizer", default=None,
                        help="explicitly use an existing train-only tokenizer instead of training candidates")
    parser.add_argument("--baseline", default=None,
                        help="override the tokenizer baseline path for the comparison report")
    parser.add_argument("--verify", default=None,
                        help="verify a frozen dataset directory and exit")
    args = parser.parse_args(argv)

    if args.verify:
        manifest = verify_frozen(Path(args.verify))
        _print({"status": "FROZEN_VERIFIED", "content_id": manifest["content_id"],
                "corpus_version": manifest["corpus_version"],
                "effective_train_tokens": manifest["effective_train_tokens"]})
        return 0

    config = load_config(Path(args.config))
    report = status_report(config, Path.cwd(), demo=args.demo)
    if args.status_only:
        _print(report)
        return 0
    if not args.demo and report["status"] != "READY_FOR_BUILD":
        out = Path(args.out or "reports/corpus-v4")
        out.mkdir(parents=True, exist_ok=True)
        write_json(out / "status-report.json", report)
        _print(report)
        print(f"\nNOT_READY: no corpus was fabricated. Status written to {out / 'status-report.json'}",
              file=sys.stderr)
        return 2

    out = Path(args.out or ("reports/corpus-v4-demo" if args.demo else "artifacts/corpus/v4"))
    registry, inputs = _registry(config, args.demo)
    pipeline = CorpusV4Pipeline(out, config, registry, inputs, demo=args.demo)
    write_json(out / "status-report.json", report)

    if args.stage != "all":
        if args.stage == "tokenize" and args.tokenizer:
            result = pipeline.tokenize(Path(args.tokenizer))
        else:
            result = _stage_result(pipeline, args.stage)
        _print(result)
        return 0

    pipeline.ingest()
    pipeline.quality()
    pipeline.deduplicate()
    pipeline.split()
    if args.tokenizer:
        pipeline.tokenize(Path(args.tokenizer))
    else:
        pipeline.train_tokenizers()
        pipeline.evaluate_tokenizers(baseline=args.baseline)
        pipeline.tokenize()
    pipeline.contamination()
    pipeline.budget()
    audit = pipeline.audit()
    manifest = pipeline.finalize(audit=audit)
    _print({
        "status": audit["status"],
        "dataset_status": manifest["dataset_status"],
        "freeze_status": manifest["freeze_status"],
        "content_id": manifest["content_id"],
        "out": str(out.resolve()),
        "effective_train_tokens": manifest["effective_train_tokens"],
        "target_effective_train_tokens": pipeline.target_tokens,
        "license_review_status": manifest["license_review_status"],
        "contamination_audit_status": manifest["contamination_audit_status"],
        "budget_status": manifest["budget_status"],
        "tokenizer_fingerprint": manifest["tokenizer_fingerprint"],
        "blockers": audit["blockers"],
        "demo_only": args.demo,
    })
    return 0 if audit["status"] == "READY" or args.demo else 2


if __name__ == "__main__":
    raise SystemExit(main())
