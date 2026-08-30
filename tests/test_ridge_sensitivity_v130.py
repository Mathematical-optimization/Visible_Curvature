from __future__ import annotations

import pytest

from visible_curvature.analysis_runner import ridge_sensitivity_plan
from visible_curvature.config import validate_config


def test_ridge_sensitivity_plan_uses_relative_raw_max_and_compensates_negative_tail():
    rows = ridge_sensitivity_plan(
        raw_min_ritz=-0.2,
        raw_max_ritz=10.0,
        coefficients=[1.0e-4, 1.0e-3],
    )
    assert [row["ridge_coefficient"] for row in rows] == [1.0e-4, 1.0e-3]
    assert rows[0]["target_ridge"] == pytest.approx(1.0e-3)
    assert rows[1]["target_ridge"] == pytest.approx(1.0e-2)
    assert rows[0]["nominal_shift"] == pytest.approx(0.201)
    assert rows[1]["nominal_shift"] == pytest.approx(0.21)


def test_ridge_sensitivity_config_rejects_negative_or_duplicate_coefficients():
    base = {
        "analysis": {
            "covariance": {"num_batches": 2, "group_size": 1},
            "curvature": {"skip_batches": 2, "num_batches": 1},
            "ridge_sensitivity": {"enabled": True, "coefficients": [1.0e-5]},
        }
    }
    assert validate_config(base, "frozen")["analysis"]["ridge_sensitivity"]["enabled"] is True

    negative = {
        **base,
        "analysis": {
            **base["analysis"],
            "ridge_sensitivity": {"enabled": True, "coefficients": [-1.0e-5]},
        },
    }
    with pytest.raises(ValueError, match="ridge_sensitivity.*nonnegative"):
        validate_config(negative, "frozen")

    duplicate = {
        **base,
        "analysis": {
            **base["analysis"],
            "ridge_sensitivity": {
                "enabled": True,
                "coefficients": [1.0e-5, 1.0e-5],
            },
        },
    }
    with pytest.raises(ValueError, match="ridge_sensitivity.*unique"):
        validate_config(duplicate, "frozen")
