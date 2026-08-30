from __future__ import annotations

from pathlib import Path

from visible_curvature.runtime_bootstrap import (
    apply_runtime_env_from_policy_file,
)


def test_policy_runtime_env_is_applied_before_heavy_imports(
    tmp_path: Path,
    monkeypatch,
):
    policy = tmp_path / "policy.yaml"
    policy.write_text(
        "runtime_env:\n"
        "  CUDA_VISIBLE_DEVICES: 3\n"
        "  OMP_NUM_THREADS: 6\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.delenv("OMP_NUM_THREADS", raising=False)

    applied = apply_runtime_env_from_policy_file(policy)

    assert applied == {
        "CUDA_VISIBLE_DEVICES": "3",
        "OMP_NUM_THREADS": "6",
    }
    assert __import__("os").environ["CUDA_VISIBLE_DEVICES"] == "3"
    assert __import__("os").environ["OMP_NUM_THREADS"] == "6"
