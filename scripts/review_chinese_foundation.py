#!/usr/bin/env python3
"""Interactively review frozen Chinese candidates and save evaluator-ready decisions."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from minillm.foundation_review import main

if __name__ == '__main__':
    raise SystemExit(main())
