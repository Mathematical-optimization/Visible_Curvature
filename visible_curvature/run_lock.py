"""Process-lifetime lock for balanced output roots.

Balanced runs write and sometimes replace multiple files under one output root.
Two writers targeting the same root can therefore corrupt scientific artifacts.
This module uses a non-blocking POSIX advisory lock that is released by the
kernel when the owning process exits, including abnormal termination.
"""
from __future__ import annotations

import json
import os
import socket
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, TextIO

try:
    import fcntl
except ImportError as exc:  # pragma: no cover - CUDA workflows are POSIX-only.
    raise RuntimeError(
        "balanced output locking requires a POSIX platform with fcntl"
    ) from exc


class OutputLockError(RuntimeError):
    """Raised when another process already owns a balanced output root."""


def _read_owner(handle: TextIO) -> str:
    handle.seek(0)
    content = handle.read().strip()
    return content if content else "owner metadata unavailable"


@contextmanager
def exclusive_output_lock(
    output_root: str | Path,
    *,
    policy_path: str | Path | None = None,
) -> Iterator[Path]:
    """Exclusively lock ``output_root`` for the lifetime of the context.

    The lock file remains as provenance after release, but the kernel lock does
    not. A second live process targeting the same root fails immediately rather
    than racing on diagnostic or final output files.
    """
    root = Path(output_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    lock_path = root / ".balanced_run.lock"
    handle = lock_path.open("a+", encoding="utf-8")
    acquired = False
    try:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            acquired = True
        except BlockingIOError as exc:
            owner = _read_owner(handle)
            raise OutputLockError(
                f"balanced output root is already locked: {root}\n{owner}"
            ) from exc

        metadata = {
            "hostname": socket.gethostname(),
            "pid": os.getpid(),
            "policy_path": (
                str(Path(policy_path).resolve())
                if policy_path is not None
                else None
            ),
            "started_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        handle.seek(0)
        handle.truncate(0)
        json.dump(metadata, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
        yield lock_path
    finally:
        if acquired:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()
