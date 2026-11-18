#!/usr/bin/env python3
"""Stage a deterministic local Pilot sample and its human review draft."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from minillm.corpus_utils import CorpusError
from minillm.corpus_v4 import load_config
from minillm.v4_pilot_sampling import PILOT_VERSION, prepare_sample


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", help="explicit local JSONL files for one source")
    parser.add_argument("--source-id", required=True)
    parser.add_argument("--out", required=True, help="new sample directory")
    parser.add_argument("--config", default="configs/corpus_v4_pilot_1m.yaml")
    parser.add_argument("--source-registry", help="local V4 source review JSON")
    parser.add_argument("--basis-points", required=True, type=int, help="100 is 1%% of family hash space")
    parser.add_argument("--seed", type=int, help="defaults to the Pilot configuration seed")
    args = parser.parse_args()
    try:
        config = load_config(args.config, source_registry=args.source_registry)
        if config["corpus_version"] != PILOT_VERSION:
            raise CorpusError("use the V4 1M Pilot configuration")
        entry = next((source["source_review"]["entry"] for source in config["sources"]
                      if source["source_id"] == args.source_id), None)
        if entry is None:
            raise CorpusError(f"source not in review registry: {args.source_id}")
        manifest = prepare_sample(args.inputs, args.out, entry=entry, basis_points=args.basis_points,
                                  seed=config["seed"] if args.seed is None else args.seed,
                                  max_record_bytes=config["quality"]["max_record_bytes"])
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
    except (CorpusError, OSError, ValueError) as exc:
        parser.exit(2, f"{parser.prog}: {exc}\n")


if __name__ == "__main__":
    main()
