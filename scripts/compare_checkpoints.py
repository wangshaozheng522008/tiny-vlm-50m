#!/usr/bin/env python3
"""CPU-compatible MiniLLM compare checkpoints entry point."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from minillm.evaluation_compare import main

if __name__ == "__main__":
    raise SystemExit(main())
