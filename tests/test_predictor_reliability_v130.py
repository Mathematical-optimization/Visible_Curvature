import math

from visible_curvature.analysis_runner import _factor_reliability
from visible_curvature.diagnostics import predicted_delta_g, predicted_delta_g_components


def test_predicted_delta_g_components_separate_baseline_and_consumption():
    components = predicted_delta_g_components(
        r_left=1.5,
        width_left=1.0,
        r_right=-0.5,
        width_right=2.0,
        r_adam=0.25,
        width_adam=5.0,
        factor_exponent=0.25,
    )
    expected_baseline = 5.0 - (1.0 + 2.0)
    expected_consumption = 0.25 * (1.5 * 1.0 - 0.5 * 2.0) - 0.5 * 0.25 * 5.0
    assert math.isclose(components["baseline_width_mismatch"], expected_baseline)
    assert math.isclose(components["delta_g_predicted_consumption"], expected_consumption)
    assert math.isclose(
        components["delta_g_predicted_full_proxy"],
        expected_baseline + expected_consumption,
    )
    assert math.isclose(
        predicted_delta_g(
            r_left=1.5,
            width_left=1.0,
            r_right=-0.5,
            width_right=2.0,
            r_adam=0.25,
            width_adam=5.0,
            factor_exponent=0.25,
        ),
        components["delta_g_predicted_full_proxy"],
    )


def _reliable_metrics():
    return {
        "r2_left": 0.9,
        "r2_right": 0.9,
        "r2_adam": 0.9,
        "commutator_left": 0.1,
        "commutator_right": 0.1,
        "elasticity_eigen_max_residual_left": 1.0e-4,
        "elasticity_eigen_max_residual_right": 1.0e-4,
        "partial_trace_negative_spectral_mass_left": 0.0,
        "partial_trace_negative_spectral_mass_right": 0.0,
        "num_factor_modes_left": 8,
        "num_factor_modes_right": 8,
        "num_coordinate_modes_adam": 64,
        "curvature_log_width_left": 1.0,
        "curvature_log_width_right": 1.0,
        "curvature_log_width_adam": 2.0,
        "delta_g_predicted_full_proxy": 0.2,
        "adam_floored_fraction": 0.0,
        "shampoo_left_floored_fraction": 0.0,
        "shampoo_right_floored_fraction": 0.0,
    }


def _reliability_cfg():
    return {
        "min_r2": 0.5,
        "max_commutator": 0.5,
        "max_factor_eigen_residual": 0.01,
        "max_factor_negative_mass": 0.05,
        "min_elasticity_modes": 4,
        "min_curvature_log_width": 0.1,
        "max_preconditioner_floored_fraction": 0.25,
    }


def test_factor_reliability_requires_adam_regression_quality():
    metrics = _reliable_metrics()
    metrics["r2_adam"] = 0.1
    reliable, reasons = _factor_reliability(metrics, _reliability_cfg())
    assert not reliable
    assert "r2_adam" in reasons.split(",")


def test_factor_reliability_reports_degenerate_factor_and_floor_dominance():
    metrics = _reliable_metrics()
    metrics["num_factor_modes_left"] = 2
    metrics["curvature_log_width_right"] = 1.0e-5
    metrics["shampoo_left_floored_fraction"] = 0.5
    reliable, reasons = _factor_reliability(metrics, _reliability_cfg())
    reason_set = set(reasons.split(","))
    assert not reliable
    assert "insufficient_left_modes" in reason_set
    assert "degenerate_right_curvature_width" in reason_set
    assert "floor_dominated_preconditioner" in reason_set


def test_factor_reliability_rejects_nonfinite_full_proxy():
    metrics = _reliable_metrics()
    metrics["delta_g_predicted_full_proxy"] = float("nan")
    reliable, reasons = _factor_reliability(metrics, _reliability_cfg())
    assert not reliable
    assert "predictor_nonfinite" in reasons.split(",")
