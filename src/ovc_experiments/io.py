from __future__ import annotations

from dataclasses import asdict, is_dataclass
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any

import numpy as np
import pandas as pd
import torch


def _json_sanitize(value: Any) -> Any:
    if is_dataclass(value):
        return _json_sanitize(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return _json_sanitize(value.detach().cpu().tolist())
    if isinstance(value, np.ndarray):
        return _json_sanitize(value.tolist())
    if isinstance(value, (np.integer, np.floating)):
        return _json_sanitize(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(key): _json_sanitize(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_json_sanitize(item) for item in value]
    if isinstance(value, list):
        return [_json_sanitize(item) for item in value]
    return value


def atomic_write_json(payload: Any, path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(
            _json_sanitize(payload),
            handle,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        handle.write("\n")
    os.replace(temporary, destination)
    return destination


def write_dataframe(frame: pd.DataFrame, path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, destination)
    return destination


def append_dataframe_row(row: dict[str, Any], path: str | Path) -> Path:
    """Atomically append one record while preserving the union of CSV fields.

    This helper is intentionally single-writer. Parallel workers should emit
    per-worker shards and merge them after completion rather than sharing one
    CSV file.
    """

    destination = Path(path)
    if destination.exists() and destination.stat().st_size > 0:
        existing = pd.read_csv(destination)
        records = existing.to_dict(orient="records")
        records.append(row)
        frame = pd.DataFrame.from_records(records)
    else:
        frame = pd.DataFrame.from_records([row])
    return write_dataframe(frame, destination)


def append_jsonl_strict(payload: Any, path: str | Path) -> Path:
    """Append one strict-JSON record, replacing non-finite values by null."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(
        _json_sanitize(payload),
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
    )
    with destination.open("a", encoding="utf-8") as handle:
        handle.write(encoded)
        handle.write("\n")
    return destination


def save_tensor_bundle(payload: dict[str, Any], path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, destination)
    return destination


def load_tensor_bundle(path: str | Path, *, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    payload = torch.load(Path(path), map_location=map_location, weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a dictionary tensor bundle: {path}")
    return payload


def file_sha256(path: str | Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_revision(directory: Path) -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=directory,
            check=True,
            capture_output=True,
            text=True,
        )
        return completed.stdout.strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None


def environment_manifest(*, project_dir: str | Path, config_path: str | Path | None = None) -> dict[str, Any]:
    project = Path(project_dir).resolve()
    manifest: dict[str, Any] = {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "device_cuda_available": torch.cuda.is_available(),
        "git_revision": _git_revision(project),
    }
    if config_path is not None and Path(config_path).exists():
        manifest["config_sha256"] = file_sha256(config_path)
    return manifest


def safe_block_name(name: str) -> str:
    return name.replace(".", "__").replace("/", "_").replace("\\", "_")
