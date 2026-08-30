from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import torch

from .chebyshev import weighted_chebyshev_certificate
from .config import ensure_output_dir, save_resolved_config
from .curvature import LinearMatrixOperator
from .preconditioners import AdamFormPreconditioner, ScalarPreconditioner, ShampooFormPreconditioner
from .utils import json_dump


def _hadamard(n: int, *, dtype: torch.dtype = torch.float64) -> torch.Tensor:
    if n <= 0 or n & (n - 1):
        raise ValueError("Hadamard dimension must be a positive power of two")
    matrix = torch.ones(1, 1, dtype=dtype)
    while matrix.shape[0] < n:
        matrix = torch.cat(
            [torch.cat([matrix, matrix], dim=1), torch.cat([matrix, -matrix], dim=1)],
            dim=0,
        )
    return matrix / math.sqrt(n)


def _spectral_power(matrix: torch.Tensor, exponent: float) -> torch.Tensor:
    values, vectors = torch.linalg.eigh(0.5 * (matrix + matrix.T))
    if float(values.min()) <= 0.0:
        raise ValueError("Synthetic theorem factors must be positive definite")
    return (vectors * values.pow(float(exponent)).unsqueeze(0)) @ vectors.T


def _dense_effective(operator: LinearMatrixOperator, preconditioner: Any) -> torch.Tensor:
    basis = torch.eye(operator.dim, dtype=operator.dtype, device=operator.device)
    columns: list[torch.Tensor] = []
    for index in range(operator.dim):
        vector = basis[:, index].reshape(operator.shape)
        half = preconditioner.apply_half(vector)
        columns.append(preconditioner.apply_half(operator.matvec_matrix(half)).reshape(-1))
    matrix = torch.stack(columns, dim=1)
    return 0.5 * (matrix + matrix.T)


def _condition(matrix: torch.Tensor) -> float:
    values = torch.linalg.eigvalsh(0.5 * (matrix + matrix.T))
    if float(values[0]) <= 0.0:
        return float("inf")
    return float((values[-1] / values[0]).detach().cpu())


def _relative_error(observed: float, expected: float) -> float:
    return abs(float(observed) - float(expected)) / max(abs(float(expected)), 1.0e-300)


def _theorem1_rows(section: Mapping[str, Any], tolerance: float) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    chi_values = [float(x) for x in section.get("chi_values", [4.0, 16.0])]
    lambda_values = [float(x) for x in section.get("lambda_values", [0.0, 1.0])]
    for chi in chi_values:
        if chi <= 1.0:
            raise ValueError("Theorem 1 requires chi > 1")
        covariance = torch.tensor([[chi**2], [1.0]], dtype=torch.float64)
        for damping in lambda_values:
            if damping < 0.0:
                raise ValueError("Theorem 1 damping must be nonnegative")
            preconditioners = {
                "aligned": (
                    AdamFormPreconditioner(covariance, damping=damping),
                    torch.tensor([[chi], [1.0]], dtype=torch.float64),
                    chi * math.sqrt((1.0 + damping) / (chi**2 + damping)),
                ),
                "scalar": (
                    ScalarPreconditioner(),
                    torch.tensor([[chi], [1.0]], dtype=torch.float64),
                    chi,
                ),
                "reversed": (
                    AdamFormPreconditioner(covariance, damping=damping),
                    torch.tensor([[1.0], [chi]], dtype=torch.float64),
                    chi * math.sqrt((chi**2 + damping) / (1.0 + damping)),
                ),
            }
            numeric: dict[str, float] = {}
            for assignment, (preconditioner, diagonal, analytic) in preconditioners.items():
                operator = LinearMatrixOperator(
                    (2, 1),
                    lambda vector, diagonal=diagonal: diagonal * vector,
                    device=torch.device("cpu"),
                    dtype=torch.float64,
                )
                value = _condition(_dense_effective(operator, preconditioner))
                numeric[assignment] = value
                rows.append(
                    {
                        "experiment": "theorem1",
                        "assignment": assignment,
                        "chi": chi,
                        "kappa": float("nan"),
                        "r": float("nan"),
                        "alpha": float("nan"),
                        "rho": float("nan"),
                        "damping": damping,
                        "condition_numeric": value,
                        "condition_analytic": analytic,
                        "relative_condition_error": _relative_error(value, analytic),
                        "delta_g": math.log(chi) - math.log(value),
                        "covariance_spectrum_error": 0.0,
                        "covariance_diagonal_error": 0.0,
                        "adam_operator_error": 0.0,
                        "invariants_passed": True,
                        "ordering_passed": False,
                    }
                )
            ordering = numeric["aligned"] < numeric["scalar"] < numeric["reversed"]
            for row in rows[-3:]:
                row["ordering_passed"] = bool(ordering)
                row["check_passed"] = bool(
                    ordering and row["relative_condition_error"] <= tolerance
                )
    return rows


def _flat_kronecker_rows(section: Mapping[str, Any], tolerance: float) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    kappa_values = [float(x) for x in section.get("kappa_values", [4.0, 16.0])]
    r_values = [float(x) for x in section.get("r_values", [1.0, 2.0])]
    alpha_values = [float(x) for x in section.get("alpha_values", [0.25, 0.5])]
    rho_values = [float(x) for x in section.get("rho_values", [0.0, 1.0, 256.0])]
    Q = _hadamard(4)

    for kappa in kappa_values:
        if kappa <= 1.0:
            raise ValueError("Theorem 3 requires kappa > 1")
        factor_eigenvalues = torch.tensor([1.0, kappa, 1.0, kappa], dtype=torch.float64)
        B = Q @ torch.diag(factor_eigenvalues) @ Q.T
        operator = LinearMatrixOperator(
            (4, 4),
            lambda vector, B=B: B @ vector @ B,
            device=torch.device("cpu"),
            dtype=torch.float64,
        )
        for r in r_values:
            if r < 0.0:
                raise ValueError("r must be nonnegative")
            C_plus = _spectral_power(B, r)
            C_minus = (kappa**r) * _spectral_power(B, -r)
            spectrum_plus = torch.linalg.eigvalsh(C_plus)
            spectrum_minus = torch.linalg.eigvalsh(C_minus)
            spectrum_error = float(
                torch.linalg.vector_norm(torch.sort(spectrum_plus).values - torch.sort(spectrum_minus).values)
                / torch.linalg.vector_norm(spectrum_plus).clamp_min(torch.finfo(torch.float64).tiny)
            )
            sigma_diag_plus = torch.outer(torch.diag(C_plus), torch.diag(C_plus))
            sigma_diag_minus = torch.outer(torch.diag(C_minus), torch.diag(C_minus))
            diagonal_error = float(
                torch.linalg.vector_norm(sigma_diag_plus - sigma_diag_minus)
                / torch.linalg.vector_norm(sigma_diag_plus).clamp_min(torch.finfo(torch.float64).tiny)
            )
            adam_plus = AdamFormPreconditioner(sigma_diag_plus, damping=0.0)
            adam_minus = AdamFormPreconditioner(sigma_diag_minus, damping=0.0)
            adam_matrix_plus = _dense_effective(operator, adam_plus)
            adam_matrix_minus = _dense_effective(operator, adam_minus)
            adam_operator_error = float(
                torch.linalg.vector_norm(adam_matrix_plus - adam_matrix_minus)
                / torch.linalg.vector_norm(adam_matrix_plus).clamp_min(torch.finfo(torch.float64).tiny)
            )
            adam_condition = _condition(adam_matrix_plus)
            invariants_passed = (
                spectrum_error <= tolerance
                and diagonal_error <= tolerance
                and adam_operator_error <= tolerance
            )

            for alpha in alpha_values:
                if not 0.0 < alpha <= 0.5:
                    raise ValueError("alpha must lie in (0, 0.5]")
                if r > 1.0 / alpha + 1.0e-12:
                    raise ValueError("Synthetic monotone aligned map requires r <= 1/alpha")
                for rho in rho_values:
                    if rho < 0.0:
                        raise ValueError("rho must be nonnegative")
                    S = ((kappa**r + rho) / (1.0 + rho)) ** alpha
                    analytic = {
                        "aligned": kappa**2 / S**2,
                        "adam": kappa**2,
                        "reversed": kappa**2 * S**2,
                    }
                    preconditioners = {
                        "aligned": ShampooFormPreconditioner(
                            C_plus,
                            C_plus,
                            damping=rho,
                            factor_exponent=alpha,
                        ),
                        "adam": adam_plus,
                        "reversed": ShampooFormPreconditioner(
                            C_minus,
                            C_minus,
                            damping=rho,
                            factor_exponent=alpha,
                        ),
                    }
                    numeric = {
                        name: _condition(_dense_effective(operator, preconditioner))
                        for name, preconditioner in preconditioners.items()
                    }
                    ordering = numeric["aligned"] <= numeric["adam"] <= numeric["reversed"]
                    for assignment in ("aligned", "adam", "reversed"):
                        value = numeric[assignment]
                        expected = analytic[assignment]
                        row = {
                            "experiment": "theorem3",
                            "assignment": assignment,
                            "chi": float("nan"),
                            "kappa": kappa,
                            "r": r,
                            "alpha": alpha,
                            "rho": rho,
                            "damping": rho,
                            "condition_numeric": value,
                            "condition_analytic": expected,
                            "relative_condition_error": _relative_error(value, expected),
                            "delta_g": math.log(adam_condition) - math.log(value),
                            "covariance_spectrum_error": spectrum_error,
                            "covariance_diagonal_error": diagonal_error,
                            "adam_operator_error": adam_operator_error,
                            "invariants_passed": bool(invariants_passed),
                            "ordering_passed": bool(ordering),
                        }
                        row["check_passed"] = bool(
                            invariants_passed
                            and ordering
                            and row["relative_condition_error"] <= tolerance
                        )
                        rows.append(row)
    return rows



def _normalized_signed_effective_map(
    s: float,
    *,
    kappa: float,
    r: float,
    alpha: float,
    rho: float,
    sigma: int,
) -> float:
    value = float(s)
    if sigma == 1:
        factor_statistic = value ** (0.5 * r)
    elif sigma == -1:
        factor_statistic = (kappa**r) * value ** (-0.5 * r)
    else:
        raise ValueError("sigma must be +1 or -1")
    raw = value / (factor_statistic + rho) ** (2.0 * alpha)
    if sigma == 1:
        at_one = 1.0 / (1.0 + rho) ** (2.0 * alpha)
    else:
        at_one = 1.0 / (kappa**r + rho) ** (2.0 * alpha)
    return raw / at_one


def _invert_monotone_map(
    target: float,
    *,
    kappa: float,
    r: float,
    alpha: float,
    rho: float,
    sigma: int,
) -> float:
    lower = 1.0
    upper = kappa**2
    target_value = float(target)
    lower_value = _normalized_signed_effective_map(
        lower, kappa=kappa, r=r, alpha=alpha, rho=rho, sigma=sigma
    )
    upper_value = _normalized_signed_effective_map(
        upper, kappa=kappa, r=r, alpha=alpha, rho=rho, sigma=sigma
    )
    scale = max(abs(target_value), 1.0)
    if target_value < lower_value - 1.0e-12 * scale or target_value > upper_value + 1.0e-12 * scale:
        raise ValueError(
            f"target {target_value} lies outside monotone image "
            f"[{lower_value}, {upper_value}]"
        )
    if math.isclose(target_value, lower_value, rel_tol=0.0, abs_tol=1.0e-14 * scale):
        return lower
    if math.isclose(target_value, upper_value, rel_tol=0.0, abs_tol=1.0e-14 * scale):
        return upper
    for _ in range(120):
        midpoint = 0.5 * (lower + upper)
        mapped = _normalized_signed_effective_map(
            midpoint, kappa=kappa, r=r, alpha=alpha, rho=rho, sigma=sigma
        )
        if mapped < target_value:
            lower = midpoint
        else:
            upper = midpoint
    return 0.5 * (lower + upper)


def _weighted_value_space_optimum(nodes: np.ndarray, weights: np.ndarray) -> float:
    count = int(nodes.size)
    lagrange = np.empty(count, dtype=np.float64)
    for index in range(count):
        numerator = float(np.prod(-np.delete(nodes, index)))
        denominator = float(
            np.prod(nodes[index] - np.delete(nodes, index))
        )
        lagrange[index] = numerator / denominator
    return float(1.0 / np.sum((lagrange * lagrange) / weights))


def integrated_theorem3_witness(
    *,
    kappa: float,
    r: float,
    alpha: float,
    rho: float,
    degree: int,
    tolerance: float = 1.0e-8,
) -> dict[str, Any]:
    """Instantiate the full three-group, reciprocal-closed Theorem-3 witness.

    The validator constructs the scalar, aligned, and reversed pulled-back
    Chebyshev groups inside one flat Kronecker factor and one common
    initialization. It verifies the paired invariants, one-third energy split,
    dimension bounds, and all three simultaneous lower certificates.
    """
    kappa = float(kappa)
    r = float(r)
    alpha = float(alpha)
    rho = float(rho)
    T = int(degree)
    tolerance = float(tolerance)
    if not math.isfinite(kappa) or kappa <= 1.0:
        raise ValueError("kappa must be finite and greater than one")
    if not math.isfinite(r) or r < 0.0:
        raise ValueError("r must be finite and nonnegative")
    if not math.isfinite(alpha) or not 0.0 < alpha <= 0.5:
        raise ValueError("alpha must lie in (0, 0.5]")
    if r > 1.0 / alpha + 1.0e-12:
        raise ValueError("the aligned effective map requires r <= 1/alpha")
    if not math.isfinite(rho) or rho < 0.0:
        raise ValueError("rho must be finite and nonnegative")
    if T < 1 or T != degree:
        raise ValueError("degree must be a positive integer")
    if not math.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("tolerance must be positive")

    strength = ((kappa**r + rho) / (1.0 + rho)) ** alpha
    K_scalar = kappa**2
    K_aligned = K_scalar / strength**2
    K_reversed = K_scalar * strength**2
    if K_aligned <= 1.0 + 1.0e-12:
        raise ValueError(
            "integrated finite-degree certificate excludes the exact-inverse aligned endpoint"
        )

    certificates = {
        "scalar": weighted_chebyshev_certificate(K_scalar, T),
        "aligned": weighted_chebyshev_certificate(K_aligned, T),
        "reversed": weighted_chebyshev_certificate(K_reversed, T),
    }
    scalar_nodes = np.asarray(certificates["scalar"]["nodes"], dtype=np.float64)
    aligned_targets = np.asarray(
        certificates["aligned"]["nodes"], dtype=np.float64
    )
    reversed_targets = np.asarray(
        certificates["reversed"]["nodes"], dtype=np.float64
    )
    aligned_nodes = np.asarray(
        [
            _invert_monotone_map(
                value,
                kappa=kappa,
                r=r,
                alpha=alpha,
                rho=rho,
                sigma=1,
            )
            for value in aligned_targets
        ],
        dtype=np.float64,
    )
    reversed_nodes = np.asarray(
        [
            _invert_monotone_map(
                value,
                kappa=kappa,
                r=r,
                alpha=alpha,
                rho=rho,
                sigma=-1,
            )
            for value in reversed_targets
        ],
        dtype=np.float64,
    )

    reserved = np.concatenate(
        [np.sqrt(scalar_nodes), np.sqrt(aligned_nodes), np.sqrt(reversed_nodes)]
    )
    eigenvalues = np.concatenate([reserved, kappa / reserved])
    factor_dimension = 1
    while factor_dimension < int(eigenvalues.size):
        factor_dimension *= 2
    if factor_dimension > int(eigenvalues.size):
        eigenvalues = np.concatenate(
            [
                eigenvalues,
                np.full(
                    factor_dimension - int(eigenvalues.size),
                    math.sqrt(kappa),
                    dtype=np.float64,
                ),
            ]
        )

    reciprocal_sorted = np.sort(kappa / eigenvalues)
    eigen_sorted = np.sort(eigenvalues)
    reciprocal_error = float(
        np.linalg.norm(eigen_sorted - reciprocal_sorted)
        / max(np.linalg.norm(eigen_sorted), np.finfo(np.float64).tiny)
    )

    Q = _hadamard(factor_dimension)
    b = torch.as_tensor(eigenvalues, dtype=torch.float64)
    B = Q @ torch.diag(b) @ Q.T
    c_plus_values = b.pow(r)
    c_minus_values = (kappa**r) * b.pow(-r)
    C_plus = Q @ torch.diag(c_plus_values) @ Q.T
    C_minus = Q @ torch.diag(c_minus_values) @ Q.T

    factor_spectrum_error = float(
        torch.linalg.vector_norm(
            torch.sort(c_plus_values).values - torch.sort(c_minus_values).values
        )
        / torch.linalg.vector_norm(c_plus_values).clamp_min(
            torch.finfo(torch.float64).tiny
        )
    )
    covariance_plus_spectrum = torch.sort(
        torch.outer(c_plus_values, c_plus_values).reshape(-1)
    ).values
    covariance_minus_spectrum = torch.sort(
        torch.outer(c_minus_values, c_minus_values).reshape(-1)
    ).values
    covariance_spectrum_error = float(
        torch.linalg.vector_norm(
            covariance_plus_spectrum - covariance_minus_spectrum
        )
        / torch.linalg.vector_norm(covariance_plus_spectrum).clamp_min(
            torch.finfo(torch.float64).tiny
        )
    )
    covariance_plus_diagonal = torch.outer(torch.diag(C_plus), torch.diag(C_plus))
    covariance_minus_diagonal = torch.outer(
        torch.diag(C_minus), torch.diag(C_minus)
    )
    covariance_diagonal_error = float(
        torch.linalg.vector_norm(
            covariance_plus_diagonal - covariance_minus_diagonal
        )
        / torch.linalg.vector_norm(covariance_plus_diagonal).clamp_min(
            torch.finfo(torch.float64).tiny
        )
    )
    flat_diagonal_spread = max(
        float(torch.diag(C_plus).max() - torch.diag(C_plus).min()),
        float(torch.diag(C_minus).max() - torch.diag(C_minus).min()),
    ) / max(float(torch.diag(C_plus).abs().max()), torch.finfo(torch.float64).tiny)

    group_slices = {
        "scalar": slice(0, T + 1),
        "aligned": slice(T + 1, 2 * (T + 1)),
        "reversed": slice(2 * (T + 1), 3 * (T + 1)),
    }
    group_nodes = {
        "scalar": scalar_nodes,
        "aligned": aligned_nodes,
        "reversed": reversed_nodes,
    }
    group_weights = {
        name: np.asarray(certificate["weights"], dtype=np.float64)
        for name, certificate in certificates.items()
    }

    x_groups: dict[str, torch.Tensor] = {}
    group_energies: dict[str, float] = {}
    for name, group_slice in group_slices.items():
        matrix = torch.zeros(
            factor_dimension, factor_dimension, dtype=torch.float64
        )
        nodes = group_nodes[name]
        weights = group_weights[name]
        for local_index, factor_index in enumerate(
            range(group_slice.start, group_slice.stop)
        ):
            q = Q[:, factor_index]
            coefficient = math.sqrt(
                float(weights[local_index]) / (3.0 * float(nodes[local_index]))
            )
            matrix.add_(coefficient * torch.outer(q, q))
        x_groups[name] = matrix
        group_energies[name] = float(
            0.5 * torch.sum(matrix * (B @ matrix @ B))
        )
    x0 = sum(x_groups.values(), torch.zeros_like(B))
    total_energy = float(0.5 * torch.sum(x0 * (B @ x0 @ B)))

    effective_actual = {
        "scalar": scalar_nodes,
        "aligned": np.asarray(
            [
                _normalized_signed_effective_map(
                    value,
                    kappa=kappa,
                    r=r,
                    alpha=alpha,
                    rho=rho,
                    sigma=1,
                )
                for value in aligned_nodes
            ],
            dtype=np.float64,
        ),
        "reversed": np.asarray(
            [
                _normalized_signed_effective_map(
                    value,
                    kappa=kappa,
                    r=r,
                    alpha=alpha,
                    rho=rho,
                    sigma=-1,
                )
                for value in reversed_nodes
            ],
            dtype=np.float64,
        ),
    }
    target_nodes = {
        "scalar": scalar_nodes,
        "aligned": aligned_targets,
        "reversed": reversed_targets,
    }

    row: dict[str, Any] = {
        "experiment": "integrated_theorem3_witness",
        "kappa": kappa,
        "r": r,
        "alpha": alpha,
        "rho": rho,
        "degree": T,
        "strength": strength,
        "K_scalar": K_scalar,
        "K_aligned": K_aligned,
        "K_reversed": K_reversed,
        "factor_dimension": factor_dimension,
        "full_dimension": factor_dimension**2,
        "factor_dimension_bound": 12 * (T + 1),
        "full_dimension_bound": 144 * (T + 1) ** 2,
        "reciprocal_closure_error": reciprocal_error,
        "factor_spectrum_error": factor_spectrum_error,
        "covariance_spectrum_error": covariance_spectrum_error,
        "covariance_diagonal_error": covariance_diagonal_error,
        "flat_diagonal_relative_spread": flat_diagonal_spread,
        "adam_operator_is_scalar": bool(flat_diagonal_spread <= tolerance),
        "total_initial_energy": total_energy,
    }

    certificate_passes: list[bool] = []
    node_errors: list[float] = []
    energy_errors: list[float] = []
    for name in ("scalar", "aligned", "reversed"):
        actual_nodes = effective_actual[name]
        expected_nodes = target_nodes[name]
        node_error = float(
            np.linalg.norm(actual_nodes - expected_nodes)
            / max(np.linalg.norm(expected_nodes), np.finfo(np.float64).tiny)
        )
        optimum = _weighted_value_space_optimum(actual_nodes, group_weights[name])
        expected_barrier = float(certificates[name]["C_T"])
        certificate_ratio = optimum / expected_barrier
        energy = group_energies[name]
        energy_error = abs(energy - 1.0 / 6.0) / (1.0 / 6.0)
        row[f"{name}_node_error"] = node_error
        row[f"{name}_initial_energy"] = energy
        row[f"{name}_energy_relative_error"] = energy_error
        row[f"{name}_weighted_optimum"] = optimum
        row[f"{name}_C_T"] = expected_barrier
        row[f"{name}_three_group_lower_bound"] = expected_barrier / 3.0
        row[f"{name}_certificate_ratio"] = certificate_ratio
        node_errors.append(node_error)
        energy_errors.append(energy_error)
        certificate_passes.append(
            node_error <= 100.0 * tolerance
            and energy_error <= 100.0 * tolerance
            and abs(certificate_ratio - 1.0) <= 100.0 * tolerance
            and bool(certificates[name]["all_checks_passed"])
        )

    orthogonality_error = 0.0
    names = list(x_groups)
    for left_index, left_name in enumerate(names):
        for right_name in names[left_index + 1 :]:
            orthogonality_error = max(
                orthogonality_error,
                abs(float(torch.sum(x_groups[left_name] * x_groups[right_name]))),
            )
    row["group_orthogonality_error"] = orthogonality_error
    row["dimension_bounds_passed"] = bool(
        factor_dimension < 12 * (T + 1)
        and factor_dimension**2 < 144 * (T + 1) ** 2
    )
    row["paired_invariants_passed"] = bool(
        reciprocal_error <= 100.0 * tolerance
        and factor_spectrum_error <= 100.0 * tolerance
        and covariance_spectrum_error <= 100.0 * tolerance
        and covariance_diagonal_error <= 100.0 * tolerance
        and flat_diagonal_spread <= 100.0 * tolerance
    )
    row["common_initialization_passed"] = bool(
        abs(total_energy - 0.5) <= 100.0 * tolerance
        and max(energy_errors) <= 100.0 * tolerance
        and orthogonality_error <= 100.0 * tolerance
    )
    row["simultaneous_certificates_passed"] = bool(all(certificate_passes))
    row["check_passed"] = bool(
        row["dimension_bounds_passed"]
        and row["paired_invariants_passed"]
        and row["common_initialization_passed"]
        and row["simultaneous_certificates_passed"]
    )
    return row


def _integrated_theorem3_rows(
    section: Mapping[str, Any], tolerance: float
) -> list[dict[str, Any]]:
    kappa_values = [float(value) for value in section.get("kappa_values", [8.0])]
    r_values = [float(value) for value in section.get("r_values", [2.0])]
    alpha_values = [float(value) for value in section.get("alpha_values", [0.25])]
    rho_values = [float(value) for value in section.get("rho_values", [1.0])]
    degrees = [int(value) for value in section.get("T_values", [3])]
    rows: list[dict[str, Any]] = []
    for kappa in kappa_values:
        for r in r_values:
            for alpha in alpha_values:
                for rho in rho_values:
                    strength = ((kappa**r + rho) / (1.0 + rho)) ** alpha
                    if kappa**2 / strength**2 <= 1.0 + 1.0e-12:
                        continue
                    for degree in degrees:
                        rows.append(
                            integrated_theorem3_witness(
                                kappa=kappa,
                                r=r,
                                alpha=alpha,
                                rho=rho,
                                degree=degree,
                                tolerance=tolerance,
                            )
                        )
    if not rows:
        raise ValueError("integrated Theorem-3 grid contains no non-endpoint witness")
    return rows


def _chebyshev_rows(section: Mapping[str, Any]) -> list[dict[str, Any]]:
    condition_numbers = [float(value) for value in section.get("K_values", [4.0, 16.0, 64.0])]
    degrees = [int(value) for value in section.get("T_values", [2, 3, 4, 8])]
    tolerance = float(section.get("tolerance", 1.0e-8))
    if tolerance <= 0.0:
        raise ValueError("Chebyshev certificate tolerance must be positive")

    rows: list[dict[str, Any]] = []
    array_fields = {"nodes", "lagrange_at_zero", "weights", "chebyshev_values"}
    for condition_number in condition_numbers:
        for degree in degrees:
            certificate = weighted_chebyshev_certificate(condition_number, degree)
            row: dict[str, Any] = {}
            for key, value in certificate.items():
                if key in array_fields:
                    row[f"{key}_json"] = json.dumps(value, separators=(",", ":"))
                else:
                    row[key] = value
            row["configured_tolerance"] = tolerance
            row["configured_tolerance_passed"] = bool(
                float(certificate["weight_sum_error"]) <= tolerance
                and float(certificate["quadratic_optimum_relative_error"]) <= tolerance
                and float(certificate["value_space_optimum_relative_error"]) <= tolerance
                and float(certificate["chebyshev_energy_relative_error"]) <= tolerance
            )
            row["all_checks_passed"] = bool(
                certificate["all_checks_passed"] and row["configured_tolerance_passed"]
            )
            rows.append(row)
    return rows


def run_synthetic_theory(cfg: Mapping[str, Any]) -> Path:
    """Run dense conditioning checks and weighted-Chebyshev certificates."""
    output_dir = ensure_output_dir(cfg)
    section = dict(cfg.get("synthetic", {}))
    tolerance = float(section.get("tolerance", 1.0e-8))

    conditioning_rows = _theorem1_rows(section.get("theorem1", {}), tolerance)
    conditioning_rows.extend(_flat_kronecker_rows(section, tolerance))
    frame = pd.DataFrame(conditioning_rows)
    frame.to_csv(output_dir / "theory_results.csv", index=False)
    frame[frame["experiment"] == "theorem1"].to_csv(
        output_dir / "theorem1_conditioning_results.csv", index=False
    )
    theorem3 = frame[frame["experiment"] == "theorem3"].copy()
    theorem3.to_csv(output_dir / "flat_kronecker_conditioning_results.csv", index=False)

    chebyshev_frame = pd.DataFrame(_chebyshev_rows(section.get("chebyshev", {})))
    chebyshev_frame.to_csv(output_dir / "chebyshev_certificates.csv", index=False)

    integrated_frame = pd.DataFrame(
        _integrated_theorem3_rows(
            section.get("integrated_theorem3", {}), tolerance
        )
    )
    integrated_frame.to_csv(
        output_dir / "integrated_theorem3_witness.csv", index=False
    )

    zero = theorem3[theorem3["rho"] == 0.0]
    alpha_doubling_passed = True
    for (_, _, assignment), group in zero.groupby(["kappa", "r", "assignment"]):
        if assignment == "adam":
            continue
        indexed = group.set_index("alpha")
        if 0.25 in indexed.index and 0.5 in indexed.index:
            alpha_doubling_passed &= math.isclose(
                float(indexed.loc[0.5, "delta_g"]),
                2.0 * float(indexed.loc[0.25, "delta_g"]),
                rel_tol=max(10.0 * tolerance, 1.0e-10),
                abs_tol=max(10.0 * tolerance, 1.0e-10),
            )

    damping_attenuation_passed = True
    for _, group in theorem3[theorem3["assignment"] != "adam"].groupby(
        ["kappa", "r", "alpha", "assignment"]
    ):
        group = group.sort_values("rho")
        magnitudes = group["delta_g"].abs().to_numpy()
        damping_attenuation_passed &= bool(
            (magnitudes[1:] <= magnitudes[:-1] + 10.0 * tolerance).all()
        )

    chebyshev_all_checks = bool(
        not chebyshev_frame.empty and chebyshev_frame["all_checks_passed"].all()
    )
    integrated_all_checks = bool(
        not integrated_frame.empty and integrated_frame["check_passed"].all()
    )
    all_checks = bool(
        frame["check_passed"].all()
        and alpha_doubling_passed
        and damping_attenuation_passed
        and chebyshev_all_checks
        and integrated_all_checks
    )
    summary = {
        "all_checks_passed": all_checks,
        "row_count": int(len(frame)),
        "theorem1_rows": int((frame["experiment"] == "theorem1").sum()),
        "theorem3_rows": int((frame["experiment"] == "theorem3").sum()),
        "maximum_relative_condition_error": float(frame["relative_condition_error"].max()),
        "maximum_covariance_spectrum_error": float(frame["covariance_spectrum_error"].max()),
        "maximum_covariance_diagonal_error": float(frame["covariance_diagonal_error"].max()),
        "maximum_adam_operator_error": float(frame["adam_operator_error"].max()),
        "alpha_doubling_passed": bool(alpha_doubling_passed),
        "damping_attenuation_passed": bool(damping_attenuation_passed),
        "chebyshev_all_checks_passed": chebyshev_all_checks,
        "chebyshev_certificate_count": int(len(chebyshev_frame)),
        "integrated_theorem3_all_checks_passed": integrated_all_checks,
        "integrated_theorem3_witness_count": int(len(integrated_frame)),
        "maximum_integrated_theorem3_covariance_spectrum_error": float(
            integrated_frame["covariance_spectrum_error"].max()
        ),
        "maximum_integrated_theorem3_covariance_diagonal_error": float(
            integrated_frame["covariance_diagonal_error"].max()
        ),
        "maximum_chebyshev_quadratic_optimum_relative_error": float(
            chebyshev_frame["quadratic_optimum_relative_error"].max()
        ),
        "maximum_chebyshev_energy_relative_error": float(
            chebyshev_frame["chebyshev_energy_relative_error"].max()
        ),
        "maximum_chebyshev_weight_sum_error": float(
            chebyshev_frame["weight_sum_error"].max()
        ),
    }
    json_dump(summary, output_dir / "theory_summary.json")
    save_resolved_config(cfg, output_dir)
    if not all_checks:
        raise RuntimeError(f"Synthetic theorem checks failed; inspect {output_dir}")
    return output_dir
