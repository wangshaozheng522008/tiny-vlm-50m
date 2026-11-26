#!/usr/bin/env python3
"""Prepare, build, probe, execute or evaluate the independently approved 2M Pilot."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from minillm import v4_language_scaling as scaling
from minillm.assessment_io import read_json, report_directory, write_json
from minillm.corpus_utils import CorpusError
from minillm.evaluation import EvaluationConfig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "build", "probe", "execute", "evaluate"))
    parser.add_argument("--out", type=Path, help="fresh report directory")
    parser.add_argument("--dataset", type=Path, default=scaling.OUTPUT,
                        help="independent output directory for a new 2M build plan")
    parser.add_argument("--source-registry", type=Path, help="human-reviewed 2M source registry for build")
    parser.add_argument("--contamination-decisions", type=Path, help="reviewed decisions for the fresh formal scan")
    parser.add_argument("--resume", choices=("state.pt", "state.prev.pt"), help="resume this same 245-step run")
    args = parser.parse_args()
    output = args.out or scaling.REPORT / args.stage
    dataset = args.dataset.resolve()
    protected = (scaling.BASE, scaling.WORKSPACE,
                 scaling.HISTORICAL.parent, scaling.INITIAL.parent,
                 scaling.ROOT / "reports/v4-language-pilot-first-training",
                 scaling.ROOT / "reports/v4-language-pilot-pr16-source-verification",
                 scaling.ROOT / "reports/v4-language-degeneration-diagnostics",
                 scaling.ROOT / "runs/v4-language-degeneration-memorization",
                 scaling.ROOT / read_json(scaling.TRAINING_CONFIG)["out"])
    output = output.resolve()
    if any(output == p or output in p.parents or p in output.parents for p in (*protected, dataset)):
        parser.error("reports must be separate from all frozen, historical and training directories")
    if any(dataset == p or dataset in p.parents or p in dataset.parents for p in protected):
        parser.error("the 2M dataset must be separate from historical and training directories")
    try:
        if args.stage == "prepare" or (args.stage == "build" and not args.source_registry):
            print(scaling.prepare_request(output)["status"])
            return 0 if args.stage == "prepare" else 2
        if args.stage == "build":
            from minillm.corpus_v4 import CorpusV4Pipeline, build_registry, inventory, load_config
            from minillm.v4_admission import validate_v4_pilot_policy
            config = load_config(scaling.CORPUS_CONFIG, source_registry=args.source_registry)
            if args.contamination_decisions:
                config["contamination"]["decisions"] = str(args.contamination_decisions.resolve())
            registry = build_registry(config["sources"], demo=False, license_policy=config["license_policy"])
            scaling.validate_source_set(registry)
            pipeline = CorpusV4Pipeline(dataset, config, registry,
                                        inventory(registry, scaling.ROOT, config=config))
            audit = pipeline.audit()
            manifest = pipeline.finalize(audit=audit)
            policy = validate_v4_pilot_policy(dataset) if manifest["freeze_status"] == "READY" else None
            with report_directory(output) as directory:
                write_json(directory / "freeze-status.json", {"audit": audit, "policy": policy,
                           "content_id": manifest["content_id"], "freeze_status": manifest["freeze_status"]})
            return 0 if policy and policy["ok"] else 2
        if args.stage == "evaluate":
            from minillm.language_scaling_evaluation import evaluate_scaling
            evaluate_scaling(output, dataset=dataset)
            return 0
        from minillm.v4_training_pilot import run_pilot
        training = read_json(scaling.TRAINING_CONFIG)
        if args.resume:
            training["resume"] = str(scaling.ROOT / training["out"] / args.resume)
        report = run_pilot(dataset, dataset / "tokenizer.json", "minillm-87m", training,
                           output, scaling.WORKSPACE, probe=args.stage == "probe", execute=args.stage == "execute",
                           evaluation_config=EvaluationConfig(**read_json(
                               scaling.ROOT / "configs/evaluation.v4-language-pilot.json")))
        print(report["status"], "training=" + report["training_status"])
        return 0 if report["status"] in ("COMPLETED", "WORKLOAD_PASS") else 2
    except (CorpusError, OSError, ValueError, RuntimeError, KeyError) as exc:
        print(f"BLOCKED 2M Pilot: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
