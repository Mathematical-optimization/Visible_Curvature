from __future__ import annotations

import math

import torch

from ovc_experiments.blocks import MatrixLayout
from ovc_experiments.diagnostics import (
    curvature_factor_proxies_from_ritz,
    eigenspace_overlap,
    matched_response,
    nci,
    projected_commutator,
)
from ovc_experiments.geometry import measure_frozen_geometry
from ovc_experiments.operators import DenseOperator
from ovc_experiments.preconditioners import AdamPreconditioner, ScaledPreconditioner


def test_geometry_gain_recovers_exact_inverse_case() -> None:
    h = torch.tensor([1.0, 4.0, 9.0, 16.0], dtype=torch.float64)
    curvature = DenseOperator(torch.diag(h))
    layout = MatrixLayout.from_shape((2, 2))
    preconditioner = AdamPreconditioner(h.square(), layout=layout)

    geometry = measure_frozen_geometry(
        curvature,
        preconditioner,
        exact_max_dim=16,
    )

    assert geometry.curvature_condition.condition_number == 16.0
    assert math.isclose(geometry.effective_condition.condition_number or 0.0, 1.0, rel_tol=1e-12)
    assert math.isclose(geometry.gain or 0.0, math.log(16.0), rel_tol=1e-12)


def test_matched_response_recovers_power_law_slope() -> None:
    h = torch.tensor([1.0, 2.0, 4.0, 8.0], dtype=torch.float64)
    curvature = DenseOperator(torch.diag(h))
    statistic = DenseOperator(torch.diag(h.pow(1.75)))
    vectors = torch.eye(4, dtype=torch.float64)

    response = matched_response(curvature, statistic, vectors)

    assert math.isclose(response.slope, 1.75, rel_tol=1e-10)
    assert math.isclose(response.spearman, 1.0, rel_tol=1e-12)
    assert response.valid_directions == 4


def test_projected_commutator_distinguishes_commuting_and_rotated_statistics() -> None:
    h = DenseOperator(torch.diag(torch.tensor([1.0, 2.0, 4.0], dtype=torch.float64)))
    q_commuting = DenseOperator(torch.diag(torch.tensor([2.0, 3.0, 9.0], dtype=torch.float64)))
    angle = 0.37
    rotation = torch.tensor(
        [
            [math.cos(angle), -math.sin(angle), 0.0],
            [math.sin(angle), math.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float64,
    )
    q_rotated = DenseOperator(rotation @ q_commuting.matrix @ rotation.T)
    basis = torch.eye(3, dtype=torch.float64)

    assert projected_commutator(h, q_commuting, basis) < 1e-12
    assert projected_commutator(h, q_rotated, basis) > 1e-3


def test_eigenspace_overlap_is_one_for_identical_top_subspaces() -> None:
    a = DenseOperator(torch.diag(torch.tensor([1.0, 2.0, 8.0, 9.0], dtype=torch.float64)))
    b = DenseOperator(torch.diag(torch.tensor([0.5, 1.5, 7.0, 12.0], dtype=torch.float64)))
    result = eigenspace_overlap(a, b, rank=2, exact_max_dim=8)
    assert math.isclose(result.affinity, 1.0, rel_tol=1e-12)
    assert torch.allclose(result.squared_overlap, torch.eye(2, dtype=torch.float64))


def test_nci_is_invariant_to_positive_scalar_grafting() -> None:
    layout = MatrixLayout.from_shape((2, 2))
    h = torch.diag(torch.tensor([1.0, 2.0, 5.0, 11.0], dtype=torch.float64))
    sigma = torch.diag(torch.tensor([2.0, 3.0, 7.0, 13.0], dtype=torch.float64))
    base = AdamPreconditioner(torch.tensor([1.0, 4.0, 9.0, 16.0], dtype=torch.float64), layout=layout)
    scaled = ScaledPreconditioner(base, 19.0)

    assert math.isclose(nci(base, h, sigma), nci(scaled, h, sigma), rel_tol=1e-11)


def test_curvature_factor_proxies_recover_partial_traces_from_full_eigendecomposition() -> None:
    left = torch.tensor([[3.0, 0.2], [0.2, 1.5]], dtype=torch.float64)
    right = torch.tensor([[2.0, 0.1], [0.1, 4.0]], dtype=torch.float64)
    # Row-major matrix action L X R corresponds to L kron R for symmetric factors.
    hessian = torch.kron(left, right)
    eigenvalues, eigenvectors = torch.linalg.eigh(hessian)
    layout = MatrixLayout.from_shape((2, 2))

    left_proxy, right_proxy = curvature_factor_proxies_from_ritz(
        eigenvalues, eigenvectors, layout
    )

    assert torch.allclose(left_proxy, left * torch.trace(right), atol=1e-9, rtol=1e-9)
    assert torch.allclose(right_proxy, right * torch.trace(left), atol=1e-9, rtol=1e-9)
