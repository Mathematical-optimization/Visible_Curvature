from __future__ import annotations

import json
import math
from pathlib import Path

import torch

from visible_curvature.curvature import LinearMatrixOperator, stabilize_curvature
from visible_curvature.reliability_balanced import build_stage_specs, prepare_core_config


def _diagonal_operator(values: list[float]) -> LinearMatrixOperator:
    matrix = torch.diag(torch.tensor(values, dtype=torch.float64))
    return LinearMatrixOperator(
        (1, len(values)),
        lambda value: (matrix @ value.reshape(-1)).reshape(1, -1),
        torch.device("cpu"),
        torch.float64,
    )


def test_shift_override_bypasses_stage_dependent_estimation():
    result = stabilize_curvature(
        _diagonal_operator([-2.0, 10.0]),
        psd_mode="shift",
        ridge=1.0e-2,
        ridge_mode="relative_max",
        lanczos_steps=2,
        lanczos_starts=1,
        seed=0,
        shift_override=0.75,
    )
    assert math.isclose(result.shift, 0.75, rel_tol=0.0, abs_tol=1.0e-12)
    assert result.shift_source == "override"
    vector = torch.tensor([1.0, 0.0], dtype=torch.float64)
    assert torch.allclose(result.operator.matvec(vector), torch.tensor([-1.25, 0.0], dtype=torch.float64))


def test_balanced_policy_sets_real_relative_max_ridge_schema(tmp_path: Path):
    base = {
        "scientific_run": True,
        "analysis": {
            "compute_tier": "confirmatory",
            "curvature": {
                "lanczos_steps": 64,
                "lanczos_starts": 2,
                "partial_trace_probes": 32,
                "stabilize_lanczos_steps": 16,
                "stabilize_lanczos_starts": 1,
                "ridge": 1.0e-6,
                "ridge_mode": "absolute",
            },
            "bootstrap": {"reps": 100, "minimum_reps": 100},
            "interventions": {"enabled": True},
            "alpha_sweep": {"enabled": True},
            "damping_sweep": {"enabled": True},
        },
    }
    stage = build_stage_specs(
        {"reliability": {"endpoint_steps": [96], "endpoint_starts": [3], "partial_trace_probes": [64]}}
    )[0]
    cfg, changed = prepare_core_config(
        base,
        stage=stage,
        output_dir=tmp_path / "stage",
        diagnostic=True,
        policy={"reliability": {"fixed_relative_ridge": 2.0e-5}},
    )
    assert cfg["analysis"]["curvature"]["ridge"] == 2.0e-5
    assert cfg["analysis"]["curvature"]["ridge_mode"] == "relative_max"
    assert changed["analysis.curvature.ridge"] == 2.0e-5
    assert changed["analysis.curvature.ridge_mode"] == "relative_max"


def test_prepare_core_config_injects_shift_override_path(tmp_path: Path):
    override_path = tmp_path / "curvature_shift_overrides.json"
    override_path.write_text(json.dumps({"block": 0.25}), encoding="utf-8")
    base = {
        "analysis": {
            "compute_tier": "debug",
            "curvature": {
                "lanczos_steps": 8,
                "lanczos_starts": 1,
                "partial_trace_probes": 4,
                "stabilize_lanczos_steps": 4,
                "stabilize_lanczos_starts": 1,
                "ridge": 1.0e-5,
                "ridge_mode": "relative_max",
            },
            "bootstrap": {"reps": 0, "minimum_reps": 0},
            "interventions": {"enabled": False},
            "alpha_sweep": {"enabled": False},
            "damping_sweep": {"enabled": False},
        }
    }
    stage = build_stage_specs(
        {"reliability": {"endpoint_steps": [8], "endpoint_starts": [1], "partial_trace_probes": [4]}}
    )[0]
    cfg, changed = prepare_core_config(
        base,
        stage=stage,
        output_dir=tmp_path / "stage",
        diagnostic=True,
        policy={"reliability": {}},
        shift_overrides_path=override_path,
    )
    assert cfg["analysis"]["curvature"]["shift_overrides_path"] == str(override_path.resolve())
    assert changed["analysis.curvature.shift_overrides_path"] == str(override_path.resolve())


def test_core_loader_accepts_versioned_override_document(tmp_path: Path):
    from visible_curvature.analysis_runner import load_curvature_shift_overrides

    path = tmp_path / "overrides.json"
    path.write_text(
        json.dumps({"schema_version": "1.0", "blocks": {"layer.weight": 0.125}}),
        encoding="utf-8",
    )
    values, digest = load_curvature_shift_overrides({"shift_overrides_path": str(path)})
    assert values == {"layer.weight": 0.125}
    assert len(digest) == 64


def test_extract_shift_overrides_uses_one_value_per_block(tmp_path: Path):
    from visible_curvature.reliability_balanced import extract_shift_overrides

    calibration = tmp_path / "calibration"
    calibration.mkdir()
    pd = __import__("pandas")
    pd.DataFrame(
        [
            {"block_name": "a", "covariance_moment": "centered", "curvature_shift": 0.25},
            {"block_name": "a", "covariance_moment": "uncentered", "curvature_shift": 0.25},
            {"block_name": "b", "covariance_moment": "centered", "curvature_shift": 0.5},
        ]
    ).to_csv(calibration / "block_metrics.csv", index=False)
    destination = tmp_path / "curvature_shift_overrides.json"
    result = extract_shift_overrides(calibration, destination)
    assert result == {"a": 0.25, "b": 0.5}
    payload = json.loads(destination.read_text(encoding="utf-8"))
    assert payload["blocks"] == result
