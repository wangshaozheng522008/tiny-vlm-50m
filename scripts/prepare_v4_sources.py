#!/usr/bin/env python3
"""Stage local data only. This command never approves sources or downloads data."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from minillm.v4_sources import prepare_source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+")
    parser.add_argument("--out", required=True)
    parser.add_argument("--source-id", required=True)
    parser.add_argument("--category", required=True)
    parser.add_argument("--language", required=True)
    args = parser.parse_args()
    print(json.dumps(prepare_source(args.inputs, args.out, source_id=args.source_id,
        category=args.category, language=args.language), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
