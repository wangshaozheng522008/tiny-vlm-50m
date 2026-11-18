#!/usr/bin/env python3
"""Publish a new read-only PR10 readiness snapshot; never builds or trains."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from minillm.assessment_io import report_directory
from minillm.corpus_utils import write_json
from minillm.corpus_v4 import load_config
from minillm.v4_pilot_readiness import pilot_readiness
from minillm.v4_readiness import corpus_readiness, tokenizer_readiness


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, help="new snapshot directory; never overwrite")
    parser.add_argument("--pilot", default="configs/corpus_v4_pilot_1m.yaml")
    parser.add_argument("--formal", default="configs/corpus_v4.yaml")
    parser.add_argument("--excluded-local", action="append", default=[])
    parser.add_argument("--pilot-only", action="store_true", help="report PR12 Pilot evidence and blockers")
    parser.add_argument("--source-registry", help="local reviewed Pilot registry")
    parser.add_argument("--base", default=".", help="root for relative source and evaluation paths")
    parser.add_argument("--dataset", help="existing Pilot build directory to verify")
    parser.add_argument("--sample", action="append", default=[], help="optional verified preview sample manifest")
    args = parser.parse_args()
    if args.pilot_only:
        config = load_config(args.pilot, source_registry=args.source_registry)
        dataset = args.dataset or Path(args.base) / "artifacts/corpus" / config["corpus_version"]
        report = pilot_readiness(config, args.base, dataset, sample_manifests=args.sample)
        with report_directory(args.out) as out:
            write_json(out / "pr12-pilot-readiness.json", report)
        print(report["status"], report["tokenizer_status"])
        return
    configs = {"pilot": args.pilot, "formal": args.formal}
    directories = {label: f"artifacts/corpus/{load_config(path)['corpus_version']}" for label, path in configs.items()}
    corpus = corpus_readiness(configs, excluded_local=args.excluded_local)
    tokenizer = tokenizer_readiness(directories, eval_asset="configs/tokenizer-v4-evaluation.jsonl",
                                   baseline_path="runs/minillm-87m/corpus/tokenizer.json")
    with report_directory(args.out) as out:
        write_json(out / "pr10-corpus-readiness.json", corpus)
        write_json(out / "pr10-tokenizer-comparison.json", tokenizer)
    print(corpus["data_status"], tokenizer["tokenizer_status"])


if __name__ == "__main__":
    main()
