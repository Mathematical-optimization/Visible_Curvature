from __future__ import annotations

import pytest
import torch

from visible_curvature.diagnostics import visible_elasticity


def test_visible_elasticity_reports_full_spectrum_scope_and_negative_mass():
    curvature = torch.diag(torch.tensor([-1.0, 1.0, 4.0], dtype=torch.float64))
    covariance = torch.diag(torch.tensor([1.0, 2.0, 8.0], dtype=torch.float64))
    _, aux = visible_elasticity(
        curvature,
        covariance,
        damping=0.1,
        max_full_eigh_dim=8,
        approx_k=2,
    )
    assert aux["eigenspectrum_scope"] == "full"
    assert aux["eigen_modes_returned"] == 3
    assert aux["eigen_mode_fraction"] == 1.0
    assert aux["negative_mode_fraction"] == pytest.approx(1 / 3)
    assert aux["negative_spectral_mass"] > 0
    assert aux["eigen_max_relative_residual"] < 1e-10


def test_visible_elasticity_reports_spectral_tail_scope():
    diagonal = torch.linspace(1.0, 20.0, 20, dtype=torch.float64)
    curvature = torch.diag(diagonal)
    covariance = torch.diag(diagonal.square())
    _, aux = visible_elasticity(
        curvature,
        covariance,
        damping=0.1,
        max_full_eigh_dim=8,
        approx_k=4,
    )
    assert aux["eigenspectrum_scope"] == "spectral_tails"
    assert 0 < aux["eigen_modes_returned"] < 20
    assert 0 < aux["eigen_mode_fraction"] < 1
    assert aux["eigen_max_relative_residual"] < 1e-4

from visible_curvature.analysis_runner import _endpoint_numerical_reliability


def _spec(min_value=1.0, max_value=10.0, min_residual=1e-3, max_residual=1e-3):
    return {
        "min_ritz": min_value,
        "max_ritz": max_value,
        "min_ritz_residual": min_residual,
        "max_ritz_residual": max_residual,
    }


def test_endpoint_numerical_reliability_uses_minimum_and_maximum_ritz_scales():
    cfg = {
        "max_min_ritz_residual_over_min": 0.01,
        "max_max_ritz_residual_over_max": 0.01,
    }
    good = _endpoint_numerical_reliability([_spec()], cfg)
    assert good["numerically_reliable"] is True
    bad = _endpoint_numerical_reliability([_spec(min_value=1e-4, min_residual=1e-3)], cfg)
    assert bad["numerically_reliable"] is False
    assert "ritz_residual" in bad["numerical_reliability_reasons"]

from visible_curvature.analysis_runner import _pair_numerical_reliability


def test_pair_numerical_reliability_requires_tau_sign_stability():
    cfg = {
        "max_min_ritz_residual_over_min": 0.1,
        "max_max_ritz_residual_over_max": 0.1,
        "sign_zero_tol": 1e-12,
    }
    result = _pair_numerical_reliability(
        _spec(min_value=1e-3, max_value=10.0, min_residual=1e-5),
        _spec(min_value=0.1, max_value=20.0, min_residual=1e-4),
        cfg,
        tau_sweep=[1e-1, 1e-3],
        rel_floor=1e-12,
        curvature_shift_ok=True,
    )
    assert result["tau_sign_stable"] is False
    assert result["numerically_reliable"] is False
    assert "tau_sign" in result["numerical_reliability_reasons"]
