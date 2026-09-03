from __future__ import annotations

import math

import torch

from ovc_experiments.operators import DenseOperator, FunctionOperator, ShiftedOperator, materialize
from ovc_experiments.spectral import estimate_condition, lanczos, slq_spectrum


def test_operator_materialization_and_shift() -> None:
    matrix = torch.diag(torch.tensor([1.0, 3.0, 7.0], dtype=torch.float64))
    op = DenseOperator(matrix)
    shifted = ShiftedOperator(op, 2.0)
    recovered = materialize(shifted)
    assert torch.allclose(recovered, matrix + 2.0 * torch.eye(3, dtype=torch.float64))


def test_lanczos_recovers_rotated_spd_spectrum_and_residuals() -> None:
    generator = torch.Generator().manual_seed(4)
    q, _ = torch.linalg.qr(torch.randn(6, 6, dtype=torch.float64, generator=generator))
    eigenvalues = torch.tensor([1.0, 1.5, 2.0, 4.0, 7.0, 12.0], dtype=torch.float64)
    matrix = q @ torch.diag(eigenvalues) @ q.T
    result = lanczos(DenseOperator(matrix), steps=6, seed=3, reorthogonalize=True)

    assert torch.allclose(result.ritz_values, eigenvalues, atol=1e-9, rtol=1e-9)
    assert torch.max(result.residual_norms).item() < 1e-8
    gram = result.ritz_vectors.T @ result.ritz_vectors
    assert torch.allclose(gram, torch.eye(6, dtype=torch.float64), atol=1e-8)


def test_condition_estimator_exact_and_censored_cases() -> None:
    spd = DenseOperator(torch.diag(torch.tensor([2.0, 5.0, 20.0], dtype=torch.float64)))
    estimate = estimate_condition(spd, exact_max_dim=8)
    assert not estimate.censored
    assert math.isclose(estimate.condition_number or 0.0, 10.0, rel_tol=1e-12)

    zero = FunctionOperator(
        dimension=3,
        matvec_fn=lambda x: torch.zeros_like(x),
        dtype=torch.float64,
        device=torch.device("cpu"),
        name="zero",
    )
    censored = estimate_condition(zero, exact_max_dim=8, positive_threshold=1e-12)
    assert censored.censored
    assert censored.condition_number is None
    assert censored.censor_reason == "no_positive_eigenvalue"


def test_slq_weighted_quantiles_cover_diagonal_spectrum() -> None:
    values = torch.arange(1, 17, dtype=torch.float64)
    result = slq_spectrum(
        DenseOperator(torch.diag(values)), probes=32, steps=16, seed=12
    )
    assert abs(result.quantile(0.5) - 8.5) < 1.0
    assert abs(result.quantile(0.1) - 2.0) < 1.5
    assert abs(result.quantile(0.9) - 15.0) < 1.5
    assert math.isclose(float(result.weights.sum()), 1.0, rel_tol=1e-12)
