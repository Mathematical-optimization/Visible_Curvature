#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from visible_curvature.config import load_yaml
from visible_curvature.synthetic_theory import run_synthetic_theory


def main() -> None:
    parser = argparse.ArgumentParser(description="Run exact frozen-theory checks.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    print(run_synthetic_theory(load_yaml(args.config)))


if __name__ == "__main__":
    main()
