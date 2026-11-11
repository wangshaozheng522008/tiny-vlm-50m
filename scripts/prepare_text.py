#!/usr/bin/env python3
"""Bounded corpus stages; legacy split -> train_tokenizer -> tokenize-only still works."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from minillm.corpus import Pipeline, inventory, make_content, merge, tokenize_split
from minillm.corpus_utils import CorpusError, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", nargs="+", help="JSONL/TXT files or directories (no downloads)")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--source-manifest", type=Path, help="JSON with a sources list")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--sample", type=int, default=None, help="stable disk sampling, 0=all")
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--min-chars", type=int, default=None)
    parser.add_argument("--max-chars", type=int, default=None)
    parser.add_argument("--split", nargs=3, type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--shuffle-seed", type=int, default=None)
    parser.add_argument("--dedup-method", choices=("simhash", "exact"), default=None)
    parser.add_argument("--dedup-threshold", type=float, default=None, help="full-shingle Jaccard confirmation threshold")
    parser.add_argument("--quality-config", type=Path)
    parser.add_argument("--token-budget", type=int, default=None, help="estimated tokens, category/source quotas; 0=all")
    parser.add_argument("--temp-dir", type=Path, help="SQLite working files; per-output namespace")
    parser.add_argument("--tokenize-only", action="store_true")
    parser.add_argument("--tokenizer", type=Path, help="load an existing tokenizer; never alter a model's tokenizer")
    parser.add_argument("--train-tokenizer", action="store_true", help="new experiment: train on the train split only, then pack")
    parser.add_argument("--vocab-size", type=int, default=24576)
    parser.add_argument("--tokenizer-max-chars", type=int, default=8000000)
    parser.add_argument("--report", type=Path, help="optional report copy, outside the immutable dataset")
    args = parser.parse_args()
    try:
        if args.report and (args.out.resolve() in args.report.resolve().parents or args.report.resolve() == args.out.resolve()):
            raise CorpusError("--report must be outside the immutable dataset directory")
        if args.tokenizer and args.train_tokenizer:
            raise CorpusError("choose --tokenizer or --train-tokenizer")
        if args.tokenize_only and not (args.tokenizer or args.train_tokenizer):
            raise CorpusError("--tokenize-only requires --tokenizer or --train-tokenizer")
        config = json.loads(args.config.read_text()) if args.config else {}
        if args.source_manifest:
            config["sources"] = json.loads(args.source_manifest.read_text())["sources"]
        if args.input and config.get("sources"):
            raise CorpusError("choose --input or configured sources, not both")
        if args.quality_config:
            config["quality"] = {**config.get("quality", {}), **json.loads(args.quality_config.read_text())}
        overrides = {"seed": args.seed, "shuffle_seed": args.shuffle_seed, "sample": args.sample,
                     "max_records": args.max_records, "split": args.split, "quality.min_chars": args.min_chars,
                     "quality.max_chars": args.max_chars, "deduplication.method": args.dedup_method,
                     "deduplication.threshold": args.dedup_threshold, "mixing.token_budget": args.token_budget}
        base = (args.source_manifest or args.config).resolve().parent if args.source_manifest or args.config else Path.cwd()
        owner = args.out / ".pipeline" / "plan.json"
        if args.tokenize_only:
            if not owner.exists():
                raise CorpusError("missing verified split plan; run the split stage in a new --out first")
            saved = json.loads(owner.read_text())
            content = saved["content"]
            if config:
                inherited = {k: v for k, v in content.items() if k != "cleaning_version"}
                for key, value in config.items():
                    if key in ("sources", "runtime"):
                        continue
                    inherited[key] = merge(inherited[key], value) if isinstance(value, dict) and key in inherited else value
                requested = make_content(inherited, overrides)
                if requested != content:
                    raise CorpusError("tokenize-only content configuration mismatch")
            for key, value in overrides.items():
                if value is not None:
                    parts = key.split(".")
                    existing = content[parts[0]][parts[1]] if len(parts) == 2 else content[key]
                    if value != existing:
                        raise CorpusError(f"tokenize-only parameter mismatch: {key}")
            inputs = inventory(args.input or [], config.get("sources"), base=base) if args.input or config.get("sources") else saved["inputs"]
            # Rehash snapshots even when the original input arguments are inherited.
            if not args.input and not config.get("sources"):
                from minillm.corpus_utils import verify_file
                for item in inputs:
                    verify_file(Path(item["path"]), {k: item[k] for k in ("bytes", "sha256")})
        else:
            content = make_content(config, overrides)
            inputs = inventory(args.input or [], config.get("sources"), base=base)
        published = args.out / "dataset-manifest.json"
        if published.exists() and args.train_tokenizer:
            previous = json.loads(published.read_text())["tokenizer"]
            if (previous["mode"] != "trained_on_train_only" or previous["vocab_size"] != args.vocab_size
                    or previous["budget_characters"] != args.tokenizer_max_chars):
                raise CorpusError("published tokenizer training plan changed; choose a new dataset/model experiment")
        pipeline = Pipeline(args.out, content, inputs, runtime=config.get("runtime"), temp_dir=args.temp_dir)
        if args.tokenizer or args.train_tokenizer:
            manifest = pipeline.tokenize(args.tokenizer, vocab_size=args.vocab_size, max_tokenizer_chars=args.tokenizer_max_chars)
            if args.report:
                write_json(args.report, manifest)
        else:
            pipeline.split()
            print(f"[ready] text splits in {args.out}; train a tokenizer on train.text.jsonl, then --tokenize-only")
    except (CorpusError, OSError, ValueError, KeyError, TypeError) as exc:
        print(f"ERROR corpus pipeline: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
