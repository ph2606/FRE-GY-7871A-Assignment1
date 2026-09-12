"""Run the local analysis after the four acquisition scripts finish."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import ROOT
from src.pipeline import readiness, run_analysis

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="show input readiness without estimation")
    args = parser.parse_args()
    if args.check:
        print(json.dumps(readiness(ROOT), indent=2))
    else:
        _, diagnostics = run_analysis(ROOT)
        print(json.dumps(diagnostics, indent=2, default=str))
