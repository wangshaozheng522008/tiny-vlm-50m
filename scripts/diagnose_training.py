#!/usr/bin/env python3
"""Run the bounded CPU Nano correctness, memorization and Chinese baseline."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from minillm.diagnostic_workflow import main
if __name__=='__main__':
    raise SystemExit(main('all'))
