from __future__ import annotations
from pathlib import Path


def resolve_output_dir(
    output_dir: str | Path,
    *,
    config_path: str | Path | None = None,
    create: bool = True,
) -> Path:
    """Resolve a relative output path against the config file, not process CWD."""
    path = Path(output_dir).expanduser()
    if not path.is_absolute():
        base = Path(config_path).expanduser().resolve().parent if config_path is not None else Path.cwd()
        path = base / path
    path = path.resolve()
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path
