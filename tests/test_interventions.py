from __future__ import annotations

import math

import torch

from ovc_experiments.blocks import MatrixLayout
from ovc_experiments.interventions import (
    alpha_sweep,
    damping_sweep,
    finite_sample_sweep,
    reassign_factor_spectrum,
)
from ovc_experiments.operators import DenseOperator


def _paired_problem() -> tuple[DenseOperator, torch.Tensor, torch.Tensor, MatrixLayout]:
    kappa = 10.0
    r = 2.0
    b = torch.diag(torch.tensor([1.0, kappa], dtype=torch.float64))
    hessian = DenseOperator(torch.kron(b, b))
    factor = torch.diag(torch.tensor([1.0, kappa**r], dtype=torch.float64))
    layout = MatrixLayout.from_shape((2, 2))
    return hessian, factor, b, layout


def test_reassignment_preserves_spectrum_and_orders_against_curvature_proxy() -> None:
    factor = torch.diag(torch.tensor([1.0, 3.0, 11.0], dtype=torch.float64))
    angle = 0.41
    rotation = torch.tensor(
        [
            [math.cos(angle), -math.sin(angle), 0.0],
            [math.sin(angle), math.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float64,
    )
    proxy = rotation @ torch.diag(torch.tensor([2.0, 5.0, 9.0], dtype=torch.float64)) @ rotation.T

    aligned = reassign_factor_spectrum(factor, proxy, mode="aligned")
    reversed_factor = reassign_factor_spectrum(factor, proxy, mode="reversed")
    random_factor = reassign_factor_spectrum(factor, proxy, mode="random", seed=7)

    expected_spectrum = torch.linalg.eigvalsh(factor)
    for candidate in (aligned, reversed_factor, random_factor):
        assert torch.allclose(torch.linalg.eigvalsh(candidate), expected_spectrum)

    _, proxy_vectors = torch.linalg.eigh(proxy)
    aligned_diagonal = torch.diag(proxy_vectors.T @ aligned @ proxy_vectors)
    reversed_diagonal = torch.diag(proxy_vectors.T @ reversed_factor @ proxy_vectors)
    assert torch.all(aligned_diagonal[1:] >= aligned_diagonal[:-1])
    assert torch.all(reversed_diagonal[1:] <= reversed_diagonal[:-1])


def test_alpha_sweep_recovers_aligned_and_reversed_fan() -> None:
    hessian, factor, proxy, layout = _paired_problem()
    aligned = reassign_factor_spectrum(factor, proxy, mode="aligned")
    reversed_factor = reassign_factor_spectrum(factor, proxy, mode="reversed")
    alphas = [0.0, 0.125, 0.25, 0.5]

    aligned_results = alpha_sweep(
        hessian,
        aligned,
        aligned,
        layout=layout,
        alphas=alphas,
        damping=0.0,
        exact_max_dim=16,
    )
    reversed_results = alpha_sweep(
        hessian,
        reversed_factor,
        reversed_factor,
        layout=layout,
        alphas=alphas,
        damping=0.0,
        exact_max_dim=16,
    )

    aligned_conditions = [result.condition_number for result in aligned_results]
    reversed_conditions = [result.condition_number for result in reversed_results]
    assert aligned_conditions == sorted(aligned_conditions, reverse=True)
    assert reversed_conditions == sorted(reversed_conditions)
    assert math.isclose(aligned_conditions[2], 10.0, rel_tol=1e-10)
    assert math.isclose(reversed_conditions[2], 1000.0, rel_tol=1e-10)


def test_dimensionless_damping_contracts_both_branches_toward_scalar() -> None:
    hessian, factor, proxy, layout = _paired_problem()
    aligned = reassign_factor_spectrum(factor, proxy, mode="aligned")
    reversed_factor = reassign_factor_spectrum(factor, proxy, mode="reversed")
    ratios = [0.0, 1e-2, 1.0, 1e4]

    plus = damping_sweep(
        hessian,
        aligned,
        aligned,
        layout=layout,
        alpha=0.25,
        normalized_ratios=ratios,
        exact_max_dim=16,
    )
    minus = damping_sweep(
        hessian,
        reversed_factor,
        reversed_factor,
        layout=layout,
        alpha=0.25,
        normalized_ratios=ratios,
        exact_max_dim=16,
    )

    assert plus[0].condition_number < plus[-1].condition_number
    assert minus[0].condition_number > minus[-1].condition_number
    assert abs(plus[-1].condition_number - 100.0) < 1.0
    assert abs(minus[-1].condition_number - 100.0) < 1.0
    assert plus[-1].rho_over_min == ratios[-1]
    assert plus[-1].rho_over_max < plus[-1].rho_over_min


def test_finite_sample_sweep_is_deterministic_and_uses_requested_sizes() -> None:
    generator = torch.Generator().manual_seed(12)
    gradients = torch.randn(20, 2, 3, generator=generator, dtype=torch.float64)
    layout = MatrixLayout.from_shape((2, 3))
    first = finite_sample_sweep(gradients, layout=layout, sample_sizes=[4, 8, 16], seed=5)
    second = finite_sample_sweep(gradients, layout=layout, sample_sizes=[4, 8, 16], seed=5)

    assert [item.sample_size for item in first] == [4, 8, 16]
    for left, right in zip(first, second):
        assert torch.allclose(left.moments.centered_left, right.moments.centered_left)
        assert torch.equal(left.indices, right.indices)
