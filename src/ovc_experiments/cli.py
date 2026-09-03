from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .config import load_config
from .runners import (
    run_aggregate,
    run_checkpoint_sweep,
    run_continuations,
    run_dynamics,
    run_geometry,
    run_interventions,
    run_smoke,
    run_synthetic,
    run_training,
)
from .streaming_runner import run_streaming_geometry


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ovc-experiments",
        description="Optimizer-visible curvature experiment suite",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    for command in (
        "train",
        "geometry",
        "streaming-geometry",
        "interventions",
        "dynamics",
        "continuation",
        "synthetic",
        "smoke",
        "checkpoint-sweep",
    ):
        subparser = subparsers.add_parser(command)
        subparser.add_argument("--config", required=True, type=Path)

    aggregate = subparsers.add_parser("aggregate")
    aggregate.add_argument("--geometry", required=True, nargs="+", type=Path)
    aggregate.add_argument("--interventions", nargs="*", type=Path, default=None)
    aggregate.add_argument("--output-dir", required=True, type=Path)
    aggregate.add_argument("--sign-threshold", type=float, default=0.22314355131420976)
    aggregate.add_argument("--bootstrap-replicates", type=int, default=2000)
    aggregate.add_argument("--seed", type=int, default=0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "aggregate":
        result = run_aggregate(
            args.geometry,
            output_dir=args.output_dir,
            intervention_paths=args.interventions,
            sign_threshold=args.sign_threshold,
            bootstrap_replicates=args.bootstrap_replicates,
            seed=args.seed,
        )
    else:
        config = load_config(args.config)
        dispatch = {
            "train": run_training,
            "checkpoint-sweep": run_checkpoint_sweep,
            "continuation": run_continuations,
            "geometry": run_geometry,
            "streaming-geometry": run_streaming_geometry,
            "interventions": run_interventions,
            "dynamics": run_dynamics,
            "synthetic": run_synthetic,
            "smoke": run_smoke,
        }
        result = dispatch[args.command](config)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
