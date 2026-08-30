#!/usr/bin/env python3
"""Bootstrap policy runtime settings before loading the numerical pipeline."""
from __future__ import annotations

import argparse
import sys

from visible_curvature.runtime_bootstrap import (
    apply_runtime_env_from_policy_file,
)


def _policy_path(argv: list[str]) -> str:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--policy", required=True)
    args, _ = parser.parse_known_args(argv)
    return str(args.policy)


def _main() -> int:
    apply_runtime_env_from_policy_file(_policy_path(sys.argv[1:]))

    # Import after applying CUDA/BLAS/OpenMP settings. reliability_balanced
    # imports NumPy, pandas, torch-backed partial-trace utilities, and SciPy.
    from visible_curvature.reliability_balanced import main

    return int(main(sys.argv[1:]))


if __name__ == "__main__":
    raise SystemExit(_main())
