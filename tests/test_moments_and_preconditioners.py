from __future__ import annotations

import math

import torch

from ovc_experiments.blocks import MatrixLayout
from ovc_experiments.moments import estimate_moments_from_gradients
from ovc_experiments.operators import CongruenceOperator, DenseOperator
from ovc_experiments.preconditioners import (
    AdamPreconditioner,
    ScaledPreconditioner,
    ShampooPreconditioner,
    TiledShampooPreconditioner,
)
from ovc_experiments.spectral import estimate_condition


def test_centered_and_uncentered_moments_are_computed_separately() -> None:
    gradients = torch.tensor(
        [
            [[1.0, 0.0], [0.0, 1.0]],
            [[3.0, 0.0], [0.0, -1.0]],
        ],
        dtype=torch.float64,
    )
    moments = estimate_moments_from_gradients(
        gradients, MatrixLayout.from_shape((2, 2))
    )

    expected_mean = torch.tensor([[2.0, 0.0], [0.0, 0.0]], dtype=torch.float64)
    assert torch.allclose(moments.mean_matrix, expected_mean)
    assert torch.allclose(
        moments.centered_left,
        torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float64),
    )
    assert torch.allclose(
        moments.centered_right,
        torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float64),
    )
    assert torch.allclose(
        moments.centered_diag,
        torch.tensor([1.0, 0.0, 0.0, 1.0], dtype=torch.float64),
    )
    assert torch.allclose(
        moments.uncentered_diag,
        torch.tensor([5.0, 0.0, 0.0, 1.0], dtype=torch.float64),
    )


def test_adam_preconditioner_applies_full_and_square_root_powers() -> None:
    layout = MatrixLayout.from_shape((2, 2))
    statistic = torch.tensor([1.0, 4.0, 9.0, 16.0], dtype=torch.float64)
    preconditioner = AdamPreconditioner(statistic, layout=layout, damping=0.0)
    vector = torch.ones(4, dtype=torch.float64)

    assert torch.allclose(
        preconditioner.apply(vector),
        torch.tensor([1.0, 0.5, 1.0 / 3.0, 0.25], dtype=torch.float64),
    )
    assert torch.allclose(
        preconditioner.apply_sqrt(vector),
        statistic.pow(-0.25),
    )


def test_shampoo_preconditioner_matches_two_sided_matrix_action() -> None:
    layout = MatrixLayout.from_shape((2, 2))
    left = torch.diag(torch.tensor([1.0, 16.0], dtype=torch.float64))
    right = torch.diag(torch.tensor([1.0, 81.0], dtype=torch.float64))
    preconditioner = ShampooPreconditioner(
        left, right, layout=layout, alpha=0.25, damping_left=0.0, damping_right=0.0
    )
    matrix = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float64)
    expected = torch.diag(torch.diag(left).pow(-0.25)) @ matrix @ torch.diag(torch.diag(right).pow(-0.25))

    assert torch.allclose(
        layout.to_matrix(layout.unflatten(preconditioner.apply(layout.flatten(matrix)))),
        expected,
    )
    expected_sqrt = torch.diag(torch.diag(left).pow(-0.125)) @ matrix @ torch.diag(torch.diag(right).pow(-0.125))
    assert torch.allclose(
        layout.to_matrix(layout.unflatten(preconditioner.apply_sqrt(layout.flatten(matrix)))),
        expected_sqrt,
    )


def test_full_tile_matches_untiled_shampoo_and_scalar_tiles_are_coordinatewise() -> None:
    gradients = torch.tensor(
        [
            [[1.0, 2.0], [3.0, 4.0]],
            [[2.0, 1.0], [4.0, 3.0]],
            [[1.5, 1.0], [2.0, 5.0]],
        ],
        dtype=torch.float64,
    )
    layout = MatrixLayout.from_shape((2, 2))
    moments = estimate_moments_from_gradients(gradients, layout)
    untiled = ShampooPreconditioner(
        moments.centered_left,
        moments.centered_right,
        layout=layout,
        alpha=0.25,
        damping_left=1e-3,
        damping_right=1e-3,
    )
    full_tile = TiledShampooPreconditioner.from_per_example_gradients(
        gradients,
        layout=layout,
        alpha=0.25,
        damping=1e-3,
        tile_rows=2,
        tile_cols=2,
        centered=True,
    )
    vector = torch.arange(1, 5, dtype=torch.float64)
    assert torch.allclose(full_tile.apply(vector), untiled.apply(vector), atol=1e-10)

    scalar_tiles = TiledShampooPreconditioner.from_per_example_gradients(
        gradients,
        layout=layout,
        alpha=0.25,
        damping=1e-3,
        tile_rows=1,
        tile_cols=1,
        centered=True,
    )
    output = scalar_tiles.apply(vector)
    assert output.shape == vector.shape
    assert torch.isfinite(output).all()


def test_positive_scalar_grafting_leaves_effective_condition_number_invariant() -> None:
    layout = MatrixLayout.from_shape((2, 2))
    statistic = torch.tensor([1.0, 4.0, 9.0, 16.0], dtype=torch.float64)
    base = AdamPreconditioner(statistic, layout=layout)
    scaled = ScaledPreconditioner(base, 37.0)
    hessian = DenseOperator(
        torch.diag(torch.tensor([1.0, 2.0, 8.0, 32.0], dtype=torch.float64))
    )
    base_effective = CongruenceOperator(hessian, base.apply_sqrt)
    scaled_effective = CongruenceOperator(hessian, scaled.apply_sqrt)

    k_base = estimate_condition(base_effective, exact_max_dim=8).condition_number
    k_scaled = estimate_condition(scaled_effective, exact_max_dim=8).condition_number
    assert k_base is not None and k_scaled is not None
    assert math.isclose(k_base, k_scaled, rel_tol=1e-12)
