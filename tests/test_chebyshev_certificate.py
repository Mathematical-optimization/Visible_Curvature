import math

import numpy as np
import pytest

from visible_curvature.chebyshev import weighted_chebyshev_certificate


@pytest.mark.parametrize(
    ("condition_number", "degree"),
    [(4.0, 1), (16.0, 2), (64.0, 4), (256.0, 8)],
)
def test_weighted_chebyshev_certificate_matches_exact_barrier(condition_number, degree):
    certificate = weighted_chebyshev_certificate(condition_number, degree)

    nodes = np.asarray(certificate["nodes"], dtype=np.float64)
    weights = np.asarray(certificate["weights"], dtype=np.float64)
    lagrange = np.asarray(certificate["lagrange_at_zero"], dtype=np.float64)
    chebyshev_values = np.asarray(certificate["chebyshev_values"], dtype=np.float64)

    assert len(nodes) == degree + 1
    assert np.all(np.diff(nodes) > 0.0)
    assert math.isclose(nodes[0], 1.0, rel_tol=0.0, abs_tol=1.0e-12)
    assert math.isclose(nodes[-1], condition_number, rel_tol=0.0, abs_tol=1.0e-10)
    assert np.all(weights > 0.0)
    assert math.isclose(float(weights.sum()), 1.0, rel_tol=1.0e-10, abs_tol=1.0e-12)
    assert np.array_equal(np.sign(lagrange), (-1.0) ** np.arange(degree + 1))
    assert np.allclose(
        chebyshev_values,
        ((-1.0) ** np.arange(degree + 1)) * certificate["delta_T"],
        rtol=1.0e-9,
        atol=1.0e-12,
    )
    assert math.isclose(
        certificate["quadratic_optimum"],
        certificate["C_T"],
        rel_tol=1.0e-8,
        abs_tol=1.0e-12,
    )
    assert math.isclose(
        certificate["chebyshev_energy"],
        certificate["C_T"],
        rel_tol=1.0e-8,
        abs_tol=1.0e-12,
    )
    assert math.isclose(
        certificate["three_group_lower_bound"],
        certificate["C_T"] / 3.0,
        rel_tol=1.0e-12,
        abs_tol=1.0e-15,
    )
    assert certificate["all_checks_passed"] is True


def test_two_endpoint_polynomial_does_not_evade_weighted_degree_two_grid():
    certificate = weighted_chebyshev_certificate(25.0, 2)
    nodes = np.asarray(certificate["nodes"], dtype=np.float64)
    weights = np.asarray(certificate["weights"], dtype=np.float64)

    values = (1.0 - nodes) * (1.0 - nodes / 25.0)
    weighted_energy = float(np.dot(weights, values**2))

    assert math.isclose(values[0], 0.0, abs_tol=1.0e-14)
    assert math.isclose(values[-1], 0.0, abs_tol=1.0e-12)
    assert weighted_energy > 0.0
    assert weighted_energy >= certificate["C_T"] * (1.0 - 1.0e-10)


@pytest.mark.parametrize(
    ("condition_number", "degree"),
    [(1.0, 2), (0.5, 2), (4.0, 0)],
)
def test_weighted_chebyshev_certificate_rejects_invalid_inputs(condition_number, degree):
    with pytest.raises(ValueError):
        weighted_chebyshev_certificate(condition_number, degree)
