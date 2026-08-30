#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
from pathlib import Path

import yaml


_THREAD_ENV_KEYS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def _validate_gpu_map(seeds: list[int], gpus: list[str] | None) -> list[str | None]:
    if gpus is None:
        return [None] * len(seeds)
    if len(gpus) != len(seeds):
        raise ValueError(
            "--gpus must contain exactly one device identifier per seed "
            f"({len(seeds)} seeds, {len(gpus)} GPU identifiers)"
        )
    return [str(item) for item in gpus]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create seed-specific balanced reliability policies"
    )
    parser.add_argument(
        "--template",
        default="configs/hf_opt125m_balanced_reliability.yaml",
    )
    parser.add_argument(
        "--base-config",
        required=True,
        help="Exact-block confirmatory YAML",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument(
        "--gpus",
        nargs="+",
        default=None,
        help=(
            "Optional CUDA_VISIBLE_DEVICES value for each seed, in the same "
            "order as --seeds (for example: --gpus 0 1 2)"
        ),
    )
    parser.add_argument(
        "--cpu-threads",
        type=int,
        default=None,
        help=(
            "Optional per-process BLAS/OpenMP thread cap written to each "
            "policy's runtime_env"
        ),
    )
    parser.add_argument("--output-dir", default="configs/generated_balanced")
    parser.add_argument("--run-root", default="outputs")
    args = parser.parse_args()

    if args.cpu_threads is not None and args.cpu_threads <= 0:
        parser.error("--cpu-threads must be positive")
    try:
        gpu_map = _validate_gpu_map(args.seeds, args.gpus)
    except ValueError as exc:
        parser.error(str(exc))

    template = yaml.safe_load(Path(args.template).read_text(encoding="utf-8"))
    if not isinstance(template, dict):
        parser.error("template must contain a YAML mapping")

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    for seed, gpu in zip(args.seeds, gpu_map):
        policy = copy.deepcopy(template)
        policy["base_config"] = args.base_config
        policy["output_root"] = f"{args.run_root}/hf_opt125m_balanced_seed{seed}"
        policy.setdefault("base_overrides", {})["seed"] = seed
        policy["base_overrides"]["output_dir"] = (
            f"{args.run_root}/_unused_core_seed{seed}"
        )

        runtime_env = {
            str(key): str(value)
            for key, value in dict(policy.get("runtime_env", {})).items()
        }
        if gpu is not None:
            runtime_env["CUDA_VISIBLE_DEVICES"] = gpu
        if args.cpu_threads is not None:
            for key in _THREAD_ENV_KEYS:
                runtime_env[key] = str(args.cpu_threads)
        if runtime_env:
            policy["runtime_env"] = runtime_env

        target = out / f"hf_opt125m_balanced_seed{seed}.yaml"
        target.write_text(
            yaml.safe_dump(policy, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
        print(target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
