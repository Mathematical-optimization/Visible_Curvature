#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from visible_curvature.paper_export import export_paper_assets


def main() -> None:
    parser = argparse.ArgumentParser(description="Export focused frozen-mechanism LaTeX assets.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--paper-root", required=True)
    parser.add_argument("--allow-debug-export", action="store_true")
    args = parser.parse_args()
    for path in export_paper_assets(
        args.output_dir,
        args.paper_root,
        allow_debug_export=args.allow_debug_export,
    ):
        print(path)


if __name__ == "__main__":
    main()
