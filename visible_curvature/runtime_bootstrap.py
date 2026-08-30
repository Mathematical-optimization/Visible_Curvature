"""Apply lightweight runtime policy settings before NumPy/Torch imports."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping, MutableMapping

import yaml


class RuntimeEnvironmentError(RuntimeError):
    """Raised when a balanced policy declares an invalid runtime environment."""


def runtime_env_from_policy(policy: Mapping[str, Any]) -> dict[str, str]:
    """Normalize a policy's optional ``runtime_env`` mapping to strings."""
    value = policy.get("runtime_env", {})
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise RuntimeEnvironmentError("runtime_env must be a mapping")
    return {str(key): str(item) for key, item in value.items()}


def apply_runtime_env_from_policy_file(
    policy_path: str | Path,
    *,
    environ: MutableMapping[str, str] | None = None,
) -> dict[str, str]:
    """Load and apply ``runtime_env`` before importing numerical libraries."""
    source = Path(policy_path)
    policy = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(policy, Mapping):
        raise RuntimeEnvironmentError(
            f"expected a YAML mapping in balanced policy: {source}"
        )
    runtime_env = runtime_env_from_policy(policy)
    target = os.environ if environ is None else environ
    target.update(runtime_env)
    return runtime_env
