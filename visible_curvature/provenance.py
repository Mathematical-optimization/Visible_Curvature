from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

from . import __version__


_IGNORED_DIRECTORIES = {
    ".git",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".tox",
    ".venv",
    "__pycache__",
    "build",
    "dist",
    "outputs",
    "data",
    "validation_artifacts",
    ".worktrees",
    "worktrees",
}
_IGNORED_SUFFIXES = {".pyc", ".pyo", ".mmap", ".pt"}


def _source_files(root: Path) -> Iterable[Path]:
    for path in sorted(root.rglob("*"), key=lambda value: value.as_posix()):
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        if any(part in _IGNORED_DIRECTORIES for part in relative.parts):
            continue
        if path.suffix.lower() in _IGNORED_SUFFIXES:
            continue
        yield path


def source_tree_digest(root: str | Path) -> str:
    """Hash deterministic package/source inputs while excluding runtime artifacts."""
    base = Path(root).expanduser().resolve()
    if not base.exists() or not base.is_dir():
        raise ValueError(f"source tree root is not a directory: {base}")
    digest = hashlib.sha256()
    for path in _source_files(base):
        relative = path.relative_to(base).as_posix().encode("utf-8")
        payload = path.read_bytes()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def _distribution_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _git_state(root: Path) -> dict[str, Any]:
    def run(*args: str) -> str:
        try:
            completed = subprocess.run(
                ["git", *args],
                cwd=root,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                timeout=5,
            )
        except (OSError, subprocess.SubprocessError):
            return ""
        return completed.stdout.strip()

    commit = run("rev-parse", "HEAD")
    status = run("status", "--porcelain")
    return {
        "commit": commit or None,
        "dirty": bool(status) if commit else None,
    }


def _gpu_records() -> list[dict[str, Any]]:
    if not torch.cuda.is_available():
        return []
    records: list[dict[str, Any]] = []
    for index in range(torch.cuda.device_count()):
        properties = torch.cuda.get_device_properties(index)
        records.append(
            {
                "index": index,
                "name": properties.name,
                "total_memory_bytes": int(properties.total_memory),
                "compute_capability": [
                    int(properties.major),
                    int(properties.minor),
                ],
                "multi_processor_count": int(properties.multi_processor_count),
            }
        )
    return records


def runtime_environment_digest(payload: dict[str, Any]) -> str:
    """Hash the reproducibility-relevant software and hardware environment."""
    execution = dict(payload.get("execution", {}))
    environment = dict(execution.get("environment", {}))
    # Device-slot assignment is an orchestration detail.  Seed policies may
    # intentionally use CUDA_VISIBLE_DEVICES=0,1,2 while running on identical
    # GPU architectures; recording it is useful, but treating it as a
    # scientific incompatibility would make valid multi-GPU aggregation fail.
    environment.pop("CUDA_VISIBLE_DEVICES", None)
    execution["environment"] = environment
    software = dict(payload.get("software", {}))
    # Installation paths vary across otherwise equivalent virtual environments.
    # Version/build metadata remains in the digest; the executable path is
    # retained in the provenance record only.
    software.pop("python_executable", None)
    stable = {
        "software": software,
        "hardware": {
            key: value
            for key, value in dict(payload.get("hardware", {})).items()
            if key != "gpus"
        },
        "gpu_architectures": [
            {
                "name": gpu.get("name"),
                "compute_capability": gpu.get("compute_capability"),
                "total_memory_bytes": gpu.get("total_memory_bytes"),
            }
            for gpu in payload.get("hardware", {}).get("gpus", [])
        ],
        "execution": execution,
        "source": {
            "tree_sha256": payload.get("source", {}).get("tree_sha256"),
            "git": payload.get("source", {}).get("git", {}),
        },
    }
    encoded = json.dumps(
        stable, sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def runtime_provenance(root: str | Path | None = None) -> dict[str, Any]:
    """Collect machine-readable source, software, hardware, and execution provenance."""
    source_root = (
        Path(root).expanduser().resolve()
        if root is not None
        else Path(__file__).resolve().parents[1]
    )
    try:
        blas_config = getattr(np.__config__, "CONFIG", None)
        if blas_config is not None:
            json.dumps(blas_config, default=str)
    except Exception:  # pragma: no cover - NumPy build-specific metadata
        blas_config = None

    cudnn_version = torch.backends.cudnn.version()
    payload: dict[str, Any] = {
        "schema_version": "1.0",
        "source": {
            "root": str(source_root),
            "tree_sha256": source_tree_digest(source_root),
            "git": _git_state(source_root),
        },
        "software": {
            "python": platform.python_version(),
            "python_implementation": platform.python_implementation(),
            "python_executable": sys.executable,
            "torch": str(torch.__version__),
            "torch_cuda_build": torch.version.cuda,
            "numpy": str(np.__version__),
            "scipy": _distribution_version("scipy"),
            "pandas": _distribution_version("pandas"),
            "matplotlib": _distribution_version("matplotlib"),
            "pyyaml": _distribution_version("PyYAML"),
            "transformers": _distribution_version("transformers"),
            "datasets": _distribution_version("datasets"),
            "package": _distribution_version("visible-curvature-experiments"),
            "package_source_version": __version__,
            "cudnn": int(cudnn_version) if cudnn_version is not None else None,
        },
        "hardware": {
            "platform": platform.platform(),
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "cpu_count": os.cpu_count(),
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_device_count": int(torch.cuda.device_count()),
            "gpus": _gpu_records(),
            "numpy_build_config": blas_config,
        },
        "execution": {
            "deterministic_algorithms_enabled": bool(
                torch.are_deterministic_algorithms_enabled()
            ),
            "deterministic_algorithms_warn_only": bool(
                torch.is_deterministic_algorithms_warn_only_enabled()
            ),
            "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
            "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
            "environment": {
                key: os.environ.get(key)
                for key in (
                    "CUDA_VISIBLE_DEVICES",
                    "OMP_NUM_THREADS",
                    "MKL_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS",
                    "NUMEXPR_NUM_THREADS",
                    "CUBLAS_WORKSPACE_CONFIG",
                )
            },
        },
    }
    payload["runtime_environment_sha256"] = runtime_environment_digest(payload)
    return payload
