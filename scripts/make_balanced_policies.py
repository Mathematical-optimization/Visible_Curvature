#!/usr/bin/env python3
from __future__ import annotations
import argparse
import copy
from pathlib import Path
import yaml


def main() -> int:
    parser = argparse.ArgumentParser(description="Create seed-specific balanced reliability policies")
    parser.add_argument("--template", default="configs/hf_opt125m_balanced_reliability.yaml")
    parser.add_argument("--base-config", required=True, help="Exact-block confirmatory YAML")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--output-dir", default="configs/generated_balanced")
    parser.add_argument("--run-root", default="outputs")
    args = parser.parse_args()
    template = yaml.safe_load(Path(args.template).read_text())
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    for seed in args.seeds:
        policy = copy.deepcopy(template)
        policy["base_config"] = args.base_config
        policy["output_root"] = f"{args.run_root}/hf_opt125m_balanced_seed{seed}"
        policy.setdefault("base_overrides", {})["seed"] = seed
        policy["base_overrides"]["output_dir"] = f"{args.run_root}/_unused_core_seed{seed}"
        target = out / f"hf_opt125m_balanced_seed{seed}.yaml"
        target.write_text(yaml.safe_dump(policy, sort_keys=False, allow_unicode=True))
        print(target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
