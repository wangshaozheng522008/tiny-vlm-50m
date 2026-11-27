#!/usr/bin/env python3
"""Probe, train or evaluate the pinned, internally approved V5 Pilot."""

import argparse
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from minillm import v5_language_training as profile
from minillm.assessment_io import read_json, write_json
from minillm.corpus_utils import CorpusError
from minillm.evaluation import EvaluationConfig
from minillm.v4_training_pilot import _initialize, _resolved, preflight
from minillm.workload import probe_workload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("probe", "execute", "evaluate"))
    parser.add_argument("--resume", choices=("state.pt", "state.prev.pt"))
    args = parser.parse_args()
    if args.resume and args.stage != "execute":
        parser.error("only execute can resume")
    if Path.cwd() != profile.ROOT:
        parser.error("run from the repository root")
    if args.stage == "evaluate":
        from minillm.v5_language_evaluation import evaluate
        evaluate()
        return 0
    raw = read_json(profile.TRAINING_CONFIG)
    if args.resume:
        raw["resume"] = str(profile.RUN / args.resume)
    model, dataset = "minillm-87m", profile.DATASET
    tokenizer = dataset / "tokenizer.json"
    out = profile.REPORT / (f"{args.stage}-{time.time_ns()}")
    audit = preflight(dataset, tokenizer, model, raw, profile.WORKSPACE,
                      require_workload=args.stage == "execute",
                      evaluation_config=EvaluationConfig(**read_json(profile.EVALUATION_CONFIG)))
    write_json(out / "preflight.json", audit)
    if audit["readiness"] != "READY":
        print(audit["blocker_details"], file=sys.stderr)
        return 2
    resolved = _resolved(raw, profile.WORKSPACE)
    if args.stage == "probe":
        _initialize(profile.WORKSPACE, audit["request"], model, tokenizer)
        proof = probe_workload(dataset, tokenizer, model, resolved, resolved["workload_report"])
        print(proof["status"], flush=True)
        return 0 if proof["status"] == "PASS" else 2
    from minillm.pilot_launcher import launch_pilot
    result = launch_pilot(dataset, tokenizer, model, resolved, out / "launch", execute=True)
    return 0 if result.get("executed") and result["summary"]["stop_reason"] == "target_reached" else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (CorpusError, OSError, ValueError, RuntimeError, KeyError) as exc:
        print(f"BLOCKED V5 Pilot: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
