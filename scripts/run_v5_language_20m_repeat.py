#!/usr/bin/env python3
"""Verify, train, or evaluate the independently authorized V5 repeat stage."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from minillm import v5_repeat_training as experiment
from minillm.corpus_utils import CorpusError


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("preflight", "execute", "evaluate"))
    parser.add_argument("--resume", choices=("state.pt", "state.prev.pt"))
    parser.add_argument("--final-test", action="store_true",
                        help="measure the locked final FP32 model once after comparison publication")
    args = parser.parse_args()
    if (args.resume and args.stage != "execute") or (args.final_test and args.stage != "evaluate"):
        parser.error("--resume requires execute and --final-test requires evaluate")
    if Path.cwd() != experiment.profile.ROOT:
        parser.error("run from the repository root")
    if args.stage == "evaluate":
        from minillm.v5_repeat_evaluation import evaluate
        evaluate(final_test=args.final_test)
    elif args.stage == "preflight":
        experiment.prepare()
    else:
        with experiment.run_lock():
            return experiment.execute(args.resume)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (CorpusError, OSError, ValueError, RuntimeError, KeyError) as exc:
        print(f"BLOCKED V5 repeat stage: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
