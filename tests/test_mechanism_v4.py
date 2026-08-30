import math

import torch

from visible_curvature.covariance import CovarianceEstimate
from visible_curvature.mechanism import mean_gradient_contamination, summarize_sign_prediction
from visible_curvature.diagnostics import visible_elasticity


def test_mean_gradient_contamination_reports_adam_and_factor_ratios():
    mean = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    centered = CovarianceEstimate(
        count=4,
        mean=mean,
        diag=torch.ones(2, 2),
        left=2.0 * torch.eye(2),
        right=2.0 * torch.eye(2),
    )
    result = mean_gradient_contamination(centered)
    assert math.isclose(result["mean_fraction_adam_over_centered"], 0.5)
    assert math.isclose(result["mean_fraction_left_over_centered"], 0.5)
    assert math.isclose(result["mean_fraction_right_over_centered"], 0.5)
    assert math.isclose(result["mean_fraction_adam_of_uncentered"], 1.0 / 3.0)


def test_sign_prediction_summary_reports_balanced_accuracy_and_spearman():
    summary = summarize_sign_prediction(
        actual=[2.0, 1.0, -1.0, -2.0, float("nan")],
        predicted=[3.0, 0.5, -0.5, -4.0, 1.0],
    )
    assert summary["n_pairs"] == 4
    assert summary["sign_balanced_accuracy"] == 1.0
    assert summary["sign_accuracy"] == 1.0
    assert summary["spearman"] > 0.9
    assert summary["true_positive"] == 2
    assert summary["true_negative"] == 2


def test_visible_elasticity_floor_is_scale_invariant_below_unit_scale():
    H = torch.diag(torch.tensor([1.0e-4, 1.0e-3, 1.0e-2], dtype=torch.float64))
    C = H.square()
    fit_small, _ = visible_elasticity(H, C, damping=0.0, curvature_rel_floor=1.0e-3)
    fit_scaled, _ = visible_elasticity(100.0 * H, 10000.0 * C, damping=0.0, curvature_rel_floor=1.0e-3)
    assert fit_small.n == 3
    assert fit_scaled.n == 3
    assert math.isclose(fit_small.slope, 2.0, rel_tol=1e-6)
    assert math.isclose(fit_scaled.slope, 2.0, rel_tol=1e-6)
