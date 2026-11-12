#!/usr/bin/env python3
"""Exclude known historical model/tokenizer exposure before preparing a new pilot dataset."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from minillm.pilot_source import main

if __name__ == "__main__":
    raise SystemExit(main())
