from __future__ import annotations

import contextlib
import hashlib
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, Iterator, Mapping

import numpy as np
import torch


def seed_everything(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        if hasattr(torch.backends.cuda.matmul, "allow_tf32"):
            torch.backends.cuda.matmul.allow_tf32 = False
        if hasattr(torch.backends.cudnn, "allow_tf32"):
            torch.backends.cudnn.allow_tf32 = False
        torch.use_deterministic_algorithms(True, warn_only=True)


def stable_int_hash(text: str, mod: int = 2**31 - 1) -> int:
    h = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return int(h[:16], 16) % mod


def get_device(name: str | None = None) -> torch.device:
    if name is None or name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def get_dtype(name: str | None) -> torch.dtype:
    name = (name or "float32").lower()
    mapping = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float64": torch.float64,
        "fp64": torch.float64,
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    if name not in mapping:
        raise ValueError(f"Unsupported dtype: {name}")
    return mapping[name]


def json_dump(obj: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=_json_default)


def _json_default(x: Any) -> Any:
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().tolist()
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    raise TypeError(type(x).__name__)


def tensor_to_numpy(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().numpy()


def finite_or_nan(x: float | torch.Tensor) -> float:
    if isinstance(x, torch.Tensor):
        x = float(x.detach().cpu())
    return float(x) if math.isfinite(float(x)) else float("nan")


@contextlib.contextmanager
def only_parameter_requires_grad(model: torch.nn.Module, param: torch.nn.Parameter) -> Iterator[None]:
    old = [(p, p.requires_grad) for p in model.parameters()]
    try:
        for p, _ in old:
            p.requires_grad_(p is param)
        yield
    finally:
        for p, flag in old:
            p.requires_grad_(flag)


@contextlib.contextmanager
def eval_mode(model: torch.nn.Module) -> Iterator[None]:
    was_training = model.training
    model.eval()
    try:
        yield
    finally:
        model.train(was_training)


def move_batch_to_device(batch: Any, device: torch.device) -> Any:
    if torch.is_tensor(batch):
        return batch.to(device)
    if isinstance(batch, Mapping):
        return {k: move_batch_to_device(v, device) for k, v in batch.items()}
    if isinstance(batch, (list, tuple)):
        vals = [move_batch_to_device(v, device) for v in batch]
        return type(batch)(vals) if isinstance(batch, tuple) else vals
    return batch


def now_seconds() -> float:
    return time.perf_counter()


def memory_stats(device: torch.device) -> Dict[str, float]:
    if device.type != "cuda":
        return {"allocated_mb": 0.0, "reserved_mb": 0.0, "max_allocated_mb": 0.0}
    return {
        "allocated_mb": torch.cuda.memory_allocated(device) / 2**20,
        "reserved_mb": torch.cuda.memory_reserved(device) / 2**20,
        "max_allocated_mb": torch.cuda.max_memory_allocated(device) / 2**20,
    }


def sanitize_filename(name: str) -> str:
    keep = []
    for ch in name:
        if ch.isalnum() or ch in "._-":
            keep.append(ch)
        else:
            keep.append("_")
    return "".join(keep)


def canonical_config_hash(cfg: Mapping[str, Any], length: int = 16) -> str:
    """Hash scientific settings while excluding run-location bookkeeping."""
    def clean(value: Any) -> Any:
        if isinstance(value, Mapping):
            return {
                str(k): clean(v)
                for k, v in sorted(value.items(), key=lambda item: str(item[0]))
                if not str(k).startswith("_")
                and str(k) not in {"output_dir", "shift_overrides_path"}
            }
        if isinstance(value, (list, tuple)):
            return [clean(v) for v in value]
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        return str(value)
    payload = json.dumps(clean(cfg), sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[: int(length)]


def protocol_config_hash(cfg: Mapping[str, Any], length: int = 16) -> str:
    """Hash analysis protocol while allowing paired seeds/checkpoints/models."""
    protocol = {
        key: cfg.get(key)
        for key in ("data", "blocks", "analysis")
        if key in cfg
    }
    return canonical_config_hash(protocol, length=length)
