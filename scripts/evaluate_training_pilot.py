#!/usr/bin/env python3
"""CPU-compatible MiniLLM evaluate training pilot entry point."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from minillm.training_pilot import main

if __name__ == "__main__":
    raise SystemExit(main())
