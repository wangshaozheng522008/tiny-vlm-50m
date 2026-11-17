#!/usr/bin/env python3
"""Inspect by default; --probe and --execute are separate explicit operations."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from minillm.v4_training_pilot import main

if __name__ == "__main__":
    raise SystemExit(main())
