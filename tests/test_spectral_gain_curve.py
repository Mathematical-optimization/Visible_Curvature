from __future__ import annotations

import math

import pytest

from visible_curvature.analysis_runner import build_spectral_gain_rows
from visible_curvature.config import validate_config


def _spectrum(minimum: float, maximum: float) -> dict[str, float]:
    return {
        "min_ritz": minimum,
        "max_ritz": maximum,
        "min_ritz_residual": 0.0,
        "max_ritz_residual": 0.0,
        "steps": 8,
        "starts": 1,
    }


def test_tau_rows_preserve_comparison_identity():
    rows = build_spectral_gain_rows(
        {"block_name": "layer.weight", "seed": 3},
        _spectrum(1.0e-4, 10.0),
        _spectrum(2.0e-4, 8.0),
        taus=[1.0e-3, 1.0e-4],
        covariance_moment="centered",
        assignment="observed",
        alpha=0.25,
        sweep_mode="primary",
        damping_coefficient=0.01,
        adam_damping_coefficient=0.02,
        shampoo_damping_coefficient=0.01,
    )
    assert [row["tau"] for row in rows] == [1.0e-3, 1.0e-4]
    assert all(row["assignment"] == "observed" for row in rows)
    assert all(row["covariance_moment"] == "centered" for row in rows)
    assert all(row["alpha"] == 0.25 for row in rows)
    assert all(row["sweep_mode"] == "primary" for row in rows)
    assert rows[0]["adam_condition_saturated"] is True
    assert rows[0]["shampoo_condition_saturated"] is True
    assert math.isfinite(rows[0]["delta_g"])


def _scientific_config(*, num_batches: int, group_size: int) -> dict:
    commit = "a" * 40
    return {
        "scientific_run": True,
        "model": {
            "backend": "hf_causal_lm",
            "revision": commit,
        },
        "data": {
            "backend": "hf_text",
            "revision": commit,
            "tokenizer_revision": commit,
            "order_seed": 0,
        },
        "analysis": {
            "compute_tier": "confirmatory",
            "covariance": {
                "skip_batches": 0,
                "num_batches": num_batches,
                "group_size": group_size,
            },
            "curvature": {"skip_batches": num_batches, "num_batches": 1},
            "bootstrap": {"reps": 100, "diagnostics": "delta_only"},
        },
    }


def test_scientific_grouped_bootstrap_requires_equal_group_sizes():
    with pytest.raises(ValueError, match="divisible"):
        validate_config(_scientific_config(num_batches=10, group_size=4), "frozen")


def test_scientific_grouped_bootstrap_accepts_equal_group_sizes():
    cfg = validate_config(_scientific_config(num_batches=12, group_size=4), "frozen")
    assert cfg["analysis"]["covariance"]["group_size"] == 4
