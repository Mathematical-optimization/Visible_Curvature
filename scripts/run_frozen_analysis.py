#!/usr/bin/env python
import sys
from pathlib import Path as _BootstrapPath
_PROJECT_ROOT = _BootstrapPath(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from pathlib import Path
import argparse

from visible_curvature.analysis_runner import run_frozen_analysis
from visible_curvature.config import load_yaml


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    out = run_frozen_analysis(cfg)
    print(out)


if __name__ == "__main__":
    main()
