from __future__ import annotations

import json
from pathlib import Path

from visible_curvature.provenance import runtime_provenance, source_tree_digest


def test_source_tree_digest_is_stable_and_ignores_runtime_outputs(tmp_path: Path):
    (tmp_path / "a.py").write_text("x = 1\n", encoding="utf-8")
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "note.md").write_text("note\n", encoding="utf-8")
    first = source_tree_digest(tmp_path)
    (tmp_path / "outputs").mkdir()
    (tmp_path / "outputs" / "run.csv").write_text("debug\n", encoding="utf-8")
    assert source_tree_digest(tmp_path) == first
    (tmp_path / "a.py").write_text("x = 2\n", encoding="utf-8")
    assert source_tree_digest(tmp_path) != first


def test_runtime_provenance_is_json_serializable_and_records_environment(tmp_path: Path):
    (tmp_path / "module.py").write_text("pass\n", encoding="utf-8")
    payload = runtime_provenance(tmp_path)
    json.dumps(payload, sort_keys=True)
    assert payload["source"]["tree_sha256"] == source_tree_digest(tmp_path)
    assert payload["software"]["python"]
    assert payload["software"]["torch"]
    assert "cuda_available" in payload["hardware"]
    assert "deterministic_algorithms_enabled" in payload["execution"]


def test_runtime_environment_digest_ignores_source_checkout_path():
    from visible_curvature.provenance import runtime_environment_digest

    base = {
        "software": {"python": "3.13", "torch": "2.8"},
        "hardware": {"platform": "linux", "gpus": []},
        "execution": {"deterministic_algorithms_enabled": True},
        "source": {
            "root": "/checkout/a",
            "tree_sha256": "tree",
            "git": {"commit": "commit", "dirty": False},
        },
    }
    moved = json.loads(json.dumps(base))
    moved["source"]["root"] = "/checkout/b"
    assert runtime_environment_digest(base) == runtime_environment_digest(moved)


def test_runtime_environment_digest_ignores_gpu_slot_assignment():
    import copy

    from visible_curvature.provenance import runtime_environment_digest

    base = {
        "software": {"python": "3.13", "torch": "2.8"},
        "hardware": {
            "platform": "linux",
            "gpus": [
                {
                    "index": 0,
                    "name": "same-gpu",
                    "compute_capability": [8, 0],
                    "total_memory_bytes": 1,
                }
            ],
        },
        "execution": {
            "deterministic_algorithms_enabled": True,
            "environment": {"CUDA_VISIBLE_DEVICES": "0", "OMP_NUM_THREADS": "8"},
        },
        "source": {"tree_sha256": "tree", "git": {"commit": "c", "dirty": False}},
    }
    other = copy.deepcopy(base)
    other["execution"]["environment"]["CUDA_VISIBLE_DEVICES"] = "2"
    assert runtime_environment_digest(base) == runtime_environment_digest(other)


def test_runtime_environment_digest_ignores_python_executable_path():
    import copy

    from visible_curvature.provenance import runtime_environment_digest

    base = {
        "software": {
            "python": "3.13",
            "python_implementation": "CPython",
            "python_executable": "/venv/a/bin/python",
            "torch": "2.8",
        },
        "hardware": {"platform": "linux", "gpus": []},
        "execution": {"deterministic_algorithms_enabled": True, "environment": {}},
        "source": {"tree_sha256": "tree", "git": {"commit": "c", "dirty": False}},
    }
    moved = copy.deepcopy(base)
    moved["software"]["python_executable"] = "/venv/b/bin/python"
    assert runtime_environment_digest(base) == runtime_environment_digest(moved)


def test_runtime_provenance_records_source_package_version(tmp_path):
    from visible_curvature import __version__
    from visible_curvature.provenance import runtime_provenance

    (tmp_path / "source.py").write_text("x = 1\n", encoding="utf-8")
    payload = runtime_provenance(tmp_path)
    assert payload["software"]["package_source_version"] == __version__
