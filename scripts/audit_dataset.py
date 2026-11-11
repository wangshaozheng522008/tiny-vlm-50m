#!/usr/bin/env python3
"""Verify an immutable dataset version without loading its corpus or BINs into RAM."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from minillm.corpus_audit import audit_dataset
from minillm.corpus_utils import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--tokenizer", type=Path)
    parser.add_argument("--retokenize", action="store_true", help="compare every document's encoding to the BIN stream")
    parser.add_argument("--verify-inputs", action="store_true", help="also rehash original source snapshots (must be available)")
    args = parser.parse_args()
    if args.report and (args.dataset.resolve() in args.report.resolve().parents or args.dataset.resolve() == args.report.resolve()):
        print("ERROR: write the audit report outside the immutable dataset", file=sys.stderr)
        return 1
    report = audit_dataset(args.dataset, tokenizer_path=args.tokenizer, retokenize=args.retokenize, verify_inputs=args.verify_inputs)
    if args.report:
        write_json(args.report, report)
    for pair, count in report["exact_overlap"].items():
        print(f"{pair} exact overlap: {count}")
    print(f"Known near-duplicate cross-split groups: {report['known_near_duplicate_cross_split_groups']}")
    print("PASS dataset audit" if report["ok"] else "FAIL dataset audit: " + "; ".join(report["errors"]))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
