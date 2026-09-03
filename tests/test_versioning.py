from __future__ import annotations

from pathlib import Path
import tomllib

import ovc_experiments


def test_package_versions_are_consistent() -> None:
    root = Path(__file__).resolve().parents[1]
    project_version = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))[
        "project"
    ]["version"]
    hardened_version = (root / "VERSION_HARDENED").read_text(encoding="utf-8").strip()

    assert project_version == "1.2.0"
    assert ovc_experiments.__version__ == project_version
    assert hardened_version == project_version
