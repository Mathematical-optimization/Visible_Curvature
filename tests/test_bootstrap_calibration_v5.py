from __future__ import annotations

import math

import numpy as np

from visible_curvature.reliability import calibrated_bootstrap_interval


def test_calibrated_bootstrap_interval_is_centered_on_high_budget_point():
    low_reference = 2.0
    low_bootstrap = np.array([1.0, 2.0, 3.0, 2.5, 1.5])
    low, high = calibrated_bootstrap_interval(
        point=10.0,
        bootstrap_values=low_bootstrap,
        reference=low_reference,
        alpha=0.4,
        minimum_reps=5,
    )
    errors = low_bootstrap - low_reference
    expected_low = 10.0 - np.quantile(errors, 0.8)
    expected_high = 10.0 - np.quantile(errors, 0.2)
    assert np.isclose(low, expected_low)
    assert np.isclose(high, expected_high)


def test_calibrated_bootstrap_interval_requires_declared_minimum_replicates():
    low, high = calibrated_bootstrap_interval(
        point=1.0,
        bootstrap_values=[0.9, 1.1],
        reference=1.0,
        minimum_reps=3,
    )
    assert math.isnan(low)
    assert math.isnan(high)


def test_calibrated_bootstrap_interval_discards_nonfinite_replicates():
    low, high = calibrated_bootstrap_interval(
        point=1.0,
        bootstrap_values=[0.8, 1.0, 1.2, float("nan")],
        reference=1.0,
        alpha=0.5,
        minimum_reps=3,
    )
    assert low < 1.0 < high
