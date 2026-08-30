from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def test_seed_policy_generator_can_pin_gpu_and_cpu_threads(tmp_path: Path):
    output_dir = tmp_path / "policies"
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "make_balanced_policies.py"),
            "--template",
            str(ROOT / "configs" / "hf_opt125m_balanced_reliability.yaml"),
            "--base-config",
            "configs/exact_blocks.yaml",
            "--seeds",
            "0",
            "1",
            "--gpus",
            "2",
            "3",
            "--cpu-threads",
            "6",
            "--output-dir",
            str(output_dir),
            "--run-root",
            str(tmp_path / "runs"),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    seed0 = yaml.safe_load(
        (output_dir / "hf_opt125m_balanced_seed0.yaml").read_text(
            encoding="utf-8"
        )
    )
    seed1 = yaml.safe_load(
        (output_dir / "hf_opt125m_balanced_seed1.yaml").read_text(
            encoding="utf-8"
        )
    )

    assert seed0["runtime_env"]["CUDA_VISIBLE_DEVICES"] == "2"
    assert seed1["runtime_env"]["CUDA_VISIBLE_DEVICES"] == "3"
    for policy in (seed0, seed1):
        assert policy["runtime_env"]["OMP_NUM_THREADS"] == "6"
        assert policy["runtime_env"]["MKL_NUM_THREADS"] == "6"
        assert policy["runtime_env"]["OPENBLAS_NUM_THREADS"] == "6"
        assert policy["runtime_env"]["NUMEXPR_NUM_THREADS"] == "6"
