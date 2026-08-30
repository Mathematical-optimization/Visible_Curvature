import math

import torch

from visible_curvature.curvature import LinearMatrixOperator, estimate_partial_traces_and_diagonal
from visible_curvature.diagnostics import adam_coordinate_elasticity, predicted_delta_g, predicted_delta_g_components


def test_hutchinson_diagonal_is_exact_for_diagonal_operator():
    diagonal = torch.tensor([[1.0, 2.0], [3.0, 5.0]], dtype=torch.float64)
    op = LinearMatrixOperator(
        (2, 2),
        lambda V: diagonal * V,
        torch.device("cpu"),
        torch.float64,
    )
    _, _, diag = estimate_partial_traces_and_diagonal(op, num_probes=3, seed=11)
    assert torch.allclose(diag, diagonal)


def test_adam_coordinate_elasticity_recovers_power_coupling():
    h = torch.logspace(0, 3, 64, dtype=torch.float64).reshape(8, 8)
    r = 1.4
    q = h.pow(r)
    fit, aux = adam_coordinate_elasticity(h, q, damping=0.0)
    assert math.isclose(fit.slope, r, rel_tol=1e-10, abs_tol=1e-10)
    assert fit.r2 > 0.999999
    assert aux["curvature_log_width"] > 0


def test_predicted_delta_g_includes_baseline_width_mismatch():
    kwargs = dict(
        r_left=2.0,
        width_left=2.0,
        r_right=1.0,
        width_right=1.0,
        r_adam=1.5,
        width_adam=5.0,
    )
    got = predicted_delta_g(**kwargs)
    baseline = 5.0 - (2.0 + 1.0)
    consumption = 0.25 * (2.0 * 2.0 + 1.0 * 1.0) - 0.5 * 1.5 * 5.0
    assert math.isclose(got, baseline + consumption)
    components = predicted_delta_g_components(**kwargs)
    assert math.isclose(components["baseline_width_mismatch"], baseline)
    assert math.isclose(components["delta_g_predicted_consumption"], consumption)
