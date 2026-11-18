#!/usr/bin/env python3
"""Extract a Wikimedia pages-articles dump into canonical JSONL.

Extraction is not approval: the source keeps its registry ``license_status`` and
the output only becomes admissible after a human review admits the source.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from minillm.wiki_dump import DEFAULT_NAMESPACES, extract_dump, write_manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dump", help="pages-articles XML or XML.bz2 dump")
    parser.add_argument("--out", required=True, help="JSONL output path")
    parser.add_argument("--source-id", required=True, help="V4 source_id this dump belongs to")
    parser.add_argument("--namespaces", default=",".join(str(n) for n in DEFAULT_NAMESPACES),
                        help="comma separated namespace ids to keep (default: 0)")
    parser.add_argument("--max-pages", type=int, default=None, help="stop after N kept pages")
    parser.add_argument("--manifest", default=None, help="manifest path (default: <out>.manifest.json)")
    args = parser.parse_args()
    namespaces = tuple(sorted({int(part) for part in args.namespaces.split(",") if part.strip()}))
    manifest = extract_dump(args.dump, args.out, source_id=args.source_id,
                            namespaces=namespaces, max_pages=args.max_pages)
    write_manifest(manifest, Path(args.manifest or (args.out + ".manifest.json")))
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
