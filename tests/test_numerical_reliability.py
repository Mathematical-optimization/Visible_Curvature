import math

import torch

from visible_curvature.linear_algebra import (
    condition_from_spectrum,
    constant_step_hitting_time,
    make_lanczos_starts,
    multi_start_lanczos,
)


def test_condition_number_is_scale_invariant():
    expected = condition_from_spectrum(1.0, 10.0, rel_floor=1e-8)
    assert expected == 10.0
    for scale in (1e-12, 1e-8, 1e-4, 1.0, 1e4):
        got = condition_from_spectrum(scale, 10.0 * scale, rel_floor=1e-8)
        assert math.isclose(got, expected, rel_tol=1e-12)


def test_truncated_condition_number_uses_relative_floor():
    got = condition_from_spectrum(1e-12, 1.0, truncation_tau=1e-4)
    assert got == 1e4


def test_constant_step_hitting_time_matches_endpoint_formula():
    K = 10.0
    eps = 1e-3
    t = constant_step_hitting_time(K, eps)
    q = (K - 1.0) / (K + 1.0)
    assert q ** (2 * (t - 1)) <= eps
    assert t == 1 or q ** (2 * (t - 2)) > eps


def test_lanczos_accepts_shared_starts_and_reports_residuals():
    diag = torch.tensor([1.0, 2.0, 4.0, 8.0])
    starts = make_lanczos_starts(dim=4, starts=2, device=torch.device("cpu"), dtype=torch.float64, seed=7)
    result = multi_start_lanczos(
        lambda x: diag * x,
        dim=4,
        steps=4,
        device=torch.device("cpu"),
        dtype=torch.float64,
        starts=starts,
    )
    assert math.isclose(result["min_ritz"], 1.0, rel_tol=1e-10)
    assert math.isclose(result["max_ritz"], 8.0, rel_tol=1e-10)
    assert len(result["runs"]) == 2
    assert all("min_residual" in run and "max_residual" in run for run in result["runs"])
