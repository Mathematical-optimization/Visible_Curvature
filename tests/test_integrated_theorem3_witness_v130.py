from __future__ import annotations

import math

from visible_curvature.synthetic_theory import integrated_theorem3_witness


def test_integrated_theorem3_witness_has_common_initialization_and_three_certificates():
    row = integrated_theorem3_witness(
        kappa=8.0,
        r=2.0,
        alpha=0.25,
        rho=1.0,
        degree=3,
        tolerance=1.0e-9,
    )
    assert row["factor_dimension"] < 12 * (3 + 1)
    assert row["full_dimension"] < 144 * (3 + 1) ** 2
    assert row["reciprocal_closure_error"] < 1.0e-9
    assert row["covariance_spectrum_error"] < 1.0e-9
    assert row["covariance_diagonal_error"] < 1.0e-9
    assert row["adam_operator_is_scalar"] is True
    assert math.isclose(row["total_initial_energy"], 0.5, rel_tol=1.0e-9)
    for group in ("scalar", "aligned", "reversed"):
        assert math.isclose(
            row[f"{group}_initial_energy"], 1.0 / 6.0, rel_tol=1.0e-8
        )
        assert math.isclose(
            row[f"{group}_certificate_ratio"], 1.0, rel_tol=1.0e-8
        )
    assert row["simultaneous_certificates_passed"] is True
    assert row["check_passed"] is True
