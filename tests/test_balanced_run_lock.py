from __future__ import annotations

from pathlib import Path

import pytest

from visible_curvature.run_lock import OutputLockError, exclusive_output_lock


def test_output_lock_rejects_a_concurrent_writer(tmp_path: Path):
    output_root = tmp_path / "seed0"
    policy_path = tmp_path / "policy.yaml"
    policy_path.write_text("output_root: seed0\n", encoding="utf-8")

    with exclusive_output_lock(output_root, policy_path=policy_path):
        with pytest.raises(OutputLockError, match="already locked"):
            with exclusive_output_lock(output_root, policy_path=policy_path):
                pass


def test_output_lock_is_released_after_context_exit(tmp_path: Path):
    output_root = tmp_path / "seed0"
    policy_path = tmp_path / "policy.yaml"
    policy_path.write_text("output_root: seed0\n", encoding="utf-8")

    with exclusive_output_lock(output_root, policy_path=policy_path):
        pass

    with exclusive_output_lock(output_root, policy_path=policy_path):
        lock_path = output_root / ".balanced_run.lock"
        assert lock_path.exists()
        assert "policy.yaml" in lock_path.read_text(encoding="utf-8")


def test_balanced_orchestrator_acquires_the_output_lock():
    import inspect

    from visible_curvature.reliability_balanced import run_balanced_policy

    source = inspect.getsource(run_balanced_policy)
    assert "exclusive_output_lock" in source
