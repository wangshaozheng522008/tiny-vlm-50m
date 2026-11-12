#!/usr/bin/env python3
"""Explicit MiniLLM pilot tooling."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from minillm.pilot_launcher import main
if __name__=="__main__":
    raise SystemExit(main())
