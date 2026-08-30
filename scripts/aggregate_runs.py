#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from visible_curvature.aggregate import aggregate_frozen_runs
from visible_curvature.figures import make_all_frozen_figures


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate compatible focused frozen runs across seeds.")
    parser.add_argument("--run-dir", action="append", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--minimum-seed-count", type=int, default=3)
    parser.add_argument("--allow-incompatible", action="store_true")
    parser.add_argument("--make-figures", action="store_true")
    args = parser.parse_args()
    output = aggregate_frozen_runs(
        args.run_dir,
        args.output_dir,
        minimum_seed_count=args.minimum_seed_count,
        allow_incompatible=args.allow_incompatible,
    )
    print(output)
    if args.make_figures:
        for path in make_all_frozen_figures(output):
            print(path)


if __name__ == "__main__":
    main()
