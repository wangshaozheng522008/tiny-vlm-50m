#!/usr/bin/env python3
"""Publish a read-only V4 Pilot supply/gap report; never builds or trains.

The report is derived from existing readiness, quality, dedup, split, tokenizer
evaluation and budget artifacts.  It never changes the dataset directory, never
approves a source and never selects a tokenizer.

Example:
    python scripts/analyze_v4_pilot_supply.py \
        --config configs/corpus_v4_pilot_1m.yaml \
        --source-registry "$PILOT_REGISTRY" \
        --dataset artifacts/corpus/<pilot-build> \
        --out reports/<new-report-directory>

Exit status is 0 when the analysis is complete, even when the corpus is
``DATA_NOT_READY``.  Exit status 2 means the evidence itself is inconsistent and
the report must not be used.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from minillm.assessment_io import report_directory  # noqa: E402
from minillm.corpus_utils import write_json  # noqa: E402
from minillm.corpus_v4 import load_config  # noqa: E402
from minillm.v4_pilot_supply import analyze_supply, render_markdown  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/corpus_v4_pilot_1m.yaml")
    parser.add_argument("--source-registry", help="local human-reviewed Pilot registry")
    parser.add_argument("--dataset", required=True, help="existing V4 Pilot build directory")
    parser.add_argument("--out", required=True, help="new report directory; never overwrites")
    parser.add_argument("--base", default=".", help="root for relative paths")
    parser.add_argument("--sample", action="append", default=[],
                        help="optional verified sample manifest (repeatable)")
    parser.add_argument("--run-id", help="optional stable report id; defaults to the output name")
    args = parser.parse_args(argv)

    config = load_config(args.config, source_registry=args.source_registry)
    report = analyze_supply(config, args.base, args.dataset,
                            sample_manifests=args.sample,
                            run_id=args.run_id or Path(args.out).name)
    with report_directory(args.out) as out:
        write_json(out / "supply-analysis.json", report)
        (out / "supply-analysis.md").write_text(render_markdown(report), encoding="utf-8")
        write_json(out / "report-manifest.json", {
            "schema": "v4-pilot-supply-report/1",
            "run_id": report["run_id"],
            "status": report["status"],
            "consistency": report["consistency"]["status"],
            "content_id": report["content_id"],
            "files": ["supply-analysis.json", "supply-analysis.md"],
            "note": "Read-only analysis; no dataset artifact was built, selected or frozen.",
        })
    print(report["status"], report["consistency"]["status"],
          "estimated_gap=" + str(report["totals"]["estimated_category_gap_total"]),
          "blockers=" + ",".join(report["blockers"]))
    return 0 if report["consistency"]["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
