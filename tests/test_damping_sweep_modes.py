from __future__ import annotations

import torch

from visible_curvature.analysis_runner import _resolve_dampings, damping_sweep_plan
from visible_curvature.covariance import CovarianceEstimate


def _covariance() -> CovarianceEstimate:
    return CovarianceEstimate(
        count=4,
        mean=torch.zeros(2, 2, dtype=torch.float64),
        diag=torch.tensor([[1.0, 4.0], [9.0, 16.0]], dtype=torch.float64),
        left=torch.diag(torch.tensor([2.0, 8.0], dtype=torch.float64)),
        right=torch.diag(torch.tensor([3.0, 12.0], dtype=torch.float64)),
    )


def _config() -> dict:
    return {
        "adam": {
            "damping_coefficient": 0.2,
            "damping_statistic": "median",
            "damping_min": 0.0,
        },
        "shampoo": {
            "damping_coefficient": 0.3,
            "damping_statistic": "lambda_max",
            "damping_min": 0.0,
        },
    }


def test_separate_damping_overrides_keep_adam_fixed_for_shampoo_only():
    baseline = _resolve_dampings(_covariance(), _config())
    changed = _resolve_dampings(
        _covariance(),
        _config(),
        shampoo_coefficient_override=1.0,
    )
    assert changed["adam"] == baseline["adam"]
    assert changed["adam_coefficient"] == baseline["adam_coefficient"] == 0.2
    assert changed["left"] != baseline["left"]
    assert changed["right"] != baseline["right"]
    assert changed["shampoo_coefficient"] == 1.0


def test_damping_sweep_plan_separates_joint_and_mechanistic_modes():
    plan = damping_sweep_plan(
        _config(),
        {"modes": ["joint", "shampoo_only"], "coefficients": [0.0, 1.0]},
    )
    assert plan == [
        {
            "sweep_mode": "joint",
            "damping_coefficient": 0.0,
            "adam_damping_coefficient": 0.0,
            "shampoo_damping_coefficient": 0.0,
        },
        {
            "sweep_mode": "joint",
            "damping_coefficient": 1.0,
            "adam_damping_coefficient": 1.0,
            "shampoo_damping_coefficient": 1.0,
        },
        {
            "sweep_mode": "shampoo_only",
            "damping_coefficient": 0.0,
            "adam_damping_coefficient": 0.2,
            "shampoo_damping_coefficient": 0.0,
        },
        {
            "sweep_mode": "shampoo_only",
            "damping_coefficient": 1.0,
            "adam_damping_coefficient": 0.2,
            "shampoo_damping_coefficient": 1.0,
        },
    ]
