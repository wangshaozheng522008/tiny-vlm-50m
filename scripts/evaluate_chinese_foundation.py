#!/usr/bin/env python3
"""Read-only frozen Chinese candidate and raw-generation diagnostics."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from minillm.diagnostic_workflow import main
if __name__=='__main__':
    raise SystemExit(main('evaluate'))
