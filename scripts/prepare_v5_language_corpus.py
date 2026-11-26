#!/usr/bin/env python3
"""Inventory, prepare, approve and build the V5 10M Chinese language corpus."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from minillm import v5_language_corpus as v5
from minillm.assessment_io import read_json
from minillm.corpus_utils import CorpusError


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("inventory", "prepare", "check", "build", "report", "run"))
    parser.add_argument("--out", type=Path, default=v5.REPORT, help="report directory")
    parser.add_argument("--dataset", type=Path, default=v5.OUTPUT, help="formal frozen dataset directory")
    parser.add_argument("--registry", type=Path, default=v5.SOURCE_REGISTRY)
    parser.add_argument("--candidate-config", type=Path, default=v5.CANDIDATE_CONFIG)
    parser.add_argument("--approval", type=Path, help="human approval evidence")
    parser.add_argument("--request", type=Path, help="approval request bound by the evidence")
    args = parser.parse_args(argv)
    out = args.out.resolve()
    dataset = args.dataset.resolve()
    try:
        if args.stage in ("inventory", "run"):
            report = v5.inventory(out / "candidate", registry_path=args.registry,
                                  candidate_path=args.candidate_config)
            print(report["status"], report["packing"]["effective_tokens"], "effective candidate tokens")
        if args.stage in ("prepare", "run"):
            inventory_path = out / "candidate/inventory.json"
            request = v5.prepare(out / "prepare", registry_path=args.registry,
                                 inventory_path=inventory_path if inventory_path.is_file() else None)
            print(request["status"])
        if args.stage == "check":
            review = v5.load_source_review(args.registry, verify_inputs=True)
            pending = [entry["v4_source"]["source_id"] for entry in v5.pending_entries(review)]
            print({"approved": len(v5.approved_entries(review)), "pending": pending})
            return 0 if not pending else 2
        if args.stage == "build":
            if not args.approval or not args.request:
                parser.error("build requires --approval and --request")
            result = v5.build(dataset, registry_path=args.registry, approval_path=args.approval,
                              request_path=args.request, out_dir=out / "build")
            print(result["manifest"]["freeze_status"], result["release"]["content_id"])
        if args.stage == "report":
            inventory = read_json(out / "candidate/inventory.json")
            print(inventory["content_id"])
        return 0
    except (CorpusError, OSError, ValueError, KeyError) as exc:
        print(f"BLOCKED V5 corpus: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
