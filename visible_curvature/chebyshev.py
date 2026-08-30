from __future__ import annotations

import math
from typing import Any

import numpy as np


def _sech(value: float) -> float:
    """Evaluate sech(value) without overflowing for large positive values."""
    magnitude = abs(float(value))
    if magnitude < 20.0:
        return 1.0 / math.cosh(magnitude)
    tail = math.exp(-magnitude)
    return 2.0 * tail / (1.0 + tail * tail)


def _lagrange_values_at_zero(nodes: np.ndarray) -> np.ndarray:
    """Return the Lagrange basis values ell_j(0) for positive distinct nodes."""
    count = int(nodes.size)
    values = np.empty(count, dtype=np.float64)
    log_nodes = np.log(nodes)
    total_log_nodes = float(log_nodes.sum())
    for index in range(count):
        differences = nodes[index] - np.delete(nodes, index)
        if np.any(differences == 0.0):
            raise ValueError("Chebyshev nodes must be distinct")
        log_abs = total_log_nodes - float(log_nodes[index]) - float(np.log(np.abs(differences)).sum())
        values[index] = ((-1.0) ** index) * math.exp(log_abs)
    return values


def _constrained_chebyshev_energy(
    nodes: np.ndarray,
    weights: np.ndarray,
    condition_number: float,
    degree: int,
) -> tuple[float, float]:
    """Solve the weighted polynomial problem in a Chebyshev basis.

    The polynomial is represented as sum_k c_k T_k(x(z)), where
    x(z)=(K+1-2z)/(K-1). The constraint p(0)=1 is imposed exactly through
    the corresponding Chebyshev basis values at x(0)>1.
    """
    angles = np.arange(degree + 1, dtype=np.float64) * (math.pi / degree)
    basis_at_nodes = np.cos(np.outer(angles, np.arange(degree + 1, dtype=np.float64)))

    x_zero = (condition_number + 1.0) / (condition_number - 1.0)
    acosh_zero = math.acosh(x_zero)
    constraint = np.cosh(np.arange(degree + 1, dtype=np.float64) * acosh_zero)

    gram = basis_at_nodes.T @ (weights[:, None] * basis_at_nodes)
    solution = np.linalg.solve(gram, constraint)
    denominator = float(constraint @ solution)
    if not math.isfinite(denominator) or denominator <= 0.0:
        raise RuntimeError("Weighted Chebyshev quadratic program is not positive definite")
    coefficients = solution / denominator
    values = basis_at_nodes @ coefficients
    energy = float(np.dot(weights, values * values))
    constraint_value = float(constraint @ coefficients)
    return energy, constraint_value


def weighted_chebyshev_certificate(condition_number: float, degree: int) -> dict[str, Any]:
    """Compute a numerical certificate for the weighted Chebyshev barrier.

    For ``K > 1`` and integer ``T >= 1``, this constructs the ``T+1``
    extremal nodes and positive Lagrange weights from Lemma 2 of the paper.
    It verifies that both the constrained weighted least-squares optimum and
    the scaled Chebyshev polynomial attain ``C_T(K)``.
    """
    K = float(condition_number)
    T = int(degree)
    if not math.isfinite(K) or K <= 1.0:
        raise ValueError("condition_number must be finite and greater than one")
    if isinstance(degree, bool) or T != degree or T < 1:
        raise ValueError("degree must be a positive integer")

    indices = np.arange(T + 1, dtype=np.float64)
    nodes = 0.5 * (K + 1.0 - (K - 1.0) * np.cos(indices * math.pi / T))
    lagrange = _lagrange_values_at_zero(nodes)

    gamma = math.log((math.sqrt(K) + 1.0) / (math.sqrt(K) - 1.0))
    delta = _sech(T * gamma)
    barrier = delta * delta
    weights = delta * np.abs(lagrange)

    mapped_nodes = (K + 1.0 - 2.0 * nodes) / (K - 1.0)
    mapped_nodes = np.clip(mapped_nodes, -1.0, 1.0)
    chebyshev_values = np.cos(T * np.arccos(mapped_nodes)) * delta
    chebyshev_energy = float(np.dot(weights, chebyshev_values * chebyshev_values))

    quadratic_optimum, polynomial_constraint = _constrained_chebyshev_energy(
        nodes,
        weights,
        K,
        T,
    )
    value_space_optimum = float(1.0 / np.sum((lagrange * lagrange) / weights))

    weight_sum = float(weights.sum())
    expected_signs = (-1.0) ** np.arange(T + 1, dtype=np.float64)
    scale = max(barrier, np.finfo(np.float64).tiny)
    checks = {
        "nodes_strictly_increasing": bool(np.all(np.diff(nodes) > 0.0)),
        "endpoint_nodes_match": bool(
            math.isclose(float(nodes[0]), 1.0, rel_tol=0.0, abs_tol=1.0e-12)
            and math.isclose(float(nodes[-1]), K, rel_tol=1.0e-12, abs_tol=1.0e-12)
        ),
        "weights_positive": bool(np.all(weights > 0.0)),
        "weights_sum_to_one": bool(math.isclose(weight_sum, 1.0, rel_tol=2.0e-10, abs_tol=2.0e-12)),
        "lagrange_signs_alternate": bool(np.array_equal(np.sign(lagrange), expected_signs)),
        "chebyshev_alternates": bool(
            np.allclose(
                chebyshev_values,
                expected_signs * delta,
                rtol=2.0e-10,
                atol=2.0e-12,
            )
        ),
        "quadratic_optimum_matches": bool(
            abs(quadratic_optimum - barrier) <= max(2.0e-9 * scale, 2.0e-13)
        ),
        "value_space_optimum_matches": bool(
            abs(value_space_optimum - barrier) <= max(2.0e-10 * scale, 2.0e-13)
        ),
        "chebyshev_energy_matches": bool(
            abs(chebyshev_energy - barrier) <= max(2.0e-10 * scale, 2.0e-13)
        ),
        "polynomial_constraint_matches": bool(
            math.isclose(polynomial_constraint, 1.0, rel_tol=2.0e-10, abs_tol=2.0e-12)
        ),
    }

    return {
        "condition_number": K,
        "degree": T,
        "gamma_K": gamma,
        "delta_T": delta,
        "C_T": barrier,
        "nodes": nodes.tolist(),
        "lagrange_at_zero": lagrange.tolist(),
        "weights": weights.tolist(),
        "weight_sum": weight_sum,
        "weight_sum_error": abs(weight_sum - 1.0),
        "chebyshev_values": chebyshev_values.tolist(),
        "chebyshev_energy": chebyshev_energy,
        "chebyshev_energy_relative_error": abs(chebyshev_energy - barrier) / scale,
        "quadratic_optimum": quadratic_optimum,
        "quadratic_optimum_relative_error": abs(quadratic_optimum - barrier) / scale,
        "value_space_optimum": value_space_optimum,
        "value_space_optimum_relative_error": abs(value_space_optimum - barrier) / scale,
        "polynomial_constraint_value": polynomial_constraint,
        "three_group_lower_bound": barrier / 3.0,
        "factor_dimension_strict_upper_bound": 12 * (T + 1),
        "full_dimension_strict_upper_bound": 144 * (T + 1) ** 2,
        **checks,
        "all_checks_passed": bool(all(checks.values())),
    }
