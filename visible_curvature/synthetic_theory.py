from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping

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
    all_checks = bool(
        frame["check_passed"].all()
        and alpha_doubling_passed
        and damping_attenuation_passed
        and chebyshev_all_checks
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
