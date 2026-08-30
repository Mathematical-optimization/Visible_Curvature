#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from visible_curvature.figures import make_all_frozen_figures


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the four focused frozen-mechanism figures.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--figure-dir")
    args = parser.parse_args()
    for path in make_all_frozen_figures(args.output_dir, args.figure_dir):
        print(path)


if __name__ == "__main__":
    main()
