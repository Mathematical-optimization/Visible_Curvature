from __future__ import annotations

import math

import torch

from ovc_experiments.blocks import MatrixLayout
from ovc_experiments.interventions import damping_sweep
from ovc_experiments.operators import DenseOperator
from ovc_experiments.spectrum_utils import factorwise_damping, positive_spectrum_summary


def test_positive_spectrum_summary_is_scale_aware() -> None:
    base = torch.diag(torch.tensor([1.0e-12, 1.0], dtype=torch.float64))

    summaries = [
        positive_spectrum_summary(base * scale, relative_threshold=1e-10)
        for scale in (1.0e-12, 1.0, 1.0e12)
    ]

    assert [summary.active_rank for summary in summaries] == [1, 1, 1]
    assert [summary.numerical_null_rank for summary in summaries] == [1, 1, 1]
    ratios = [summary.maximum_active / summary.minimum_active for summary in summaries]
    assert max(ratios) / min(ratios) < 1.0 + 1e-12


def test_factorwise_damping_uses_each_factor_active_minimum() -> None:
    left = torch.diag(torch.tensor([1.0, 4.0], dtype=torch.float64))
    right = torch.diag(torch.tensor([100.0, 400.0], dtype=torch.float64))

    damping = factorwise_damping(left, right, normalized_ratio=0.1)

    assert math.isclose(damping.left, 0.1, rel_tol=1e-12)
    assert math.isclose(damping.right, 10.0, rel_tol=1e-12)
    assert math.isclose(damping.left_over_min, 0.1, rel_tol=1e-12)
    assert math.isclose(damping.right_over_min, 0.1, rel_tol=1e-12)
    assert math.isclose(damping.left_over_max, 0.025, rel_tol=1e-12)
    assert math.isclose(damping.right_over_max, 0.025, rel_tol=1e-12)


def test_damping_sweep_records_left_and_right_normalization_separately() -> None:
    layout = MatrixLayout.from_shape((2, 2))
    curvature = DenseOperator(torch.eye(4, dtype=torch.float64))
    left = torch.diag(torch.tensor([1.0, 4.0], dtype=torch.float64))
    right = torch.diag(torch.tensor([100.0, 400.0], dtype=torch.float64))

    point = damping_sweep(
        curvature,
        left,
        right,
        layout=layout,
        alpha=0.25,
        normalized_ratios=[0.1],
        exact_max_dim=8,
    )[0]

    assert math.isclose(point.damping_left or 0.0, 0.1, rel_tol=1e-12)
    assert math.isclose(point.damping_right or 0.0, 10.0, rel_tol=1e-12)
    assert math.isclose(point.rho_left_over_min or 0.0, 0.1, rel_tol=1e-12)
    assert math.isclose(point.rho_right_over_min or 0.0, 0.1, rel_tol=1e-12)


def test_zero_damping_on_singular_factor_is_censored_not_finite() -> None:
    layout = MatrixLayout.from_shape((2, 1))
    curvature = DenseOperator(torch.eye(2, dtype=torch.float64))
    left = torch.diag(torch.tensor([1.0, 0.0], dtype=torch.float64))
    right = torch.ones((1, 1), dtype=torch.float64)

    point = damping_sweep(
        curvature,
        left,
        right,
        layout=layout,
        alpha=0.25,
        normalized_ratios=[0.0],
        exact_max_dim=8,
    )[0]

    assert point.censored
    assert math.isnan(point.condition_number)
    assert point.geometry is not None
    assert point.geometry.effective_condition.censor_reason.startswith("invalid_preconditioner:")


def test_tiled_shampoo_normalizes_damping_within_each_tile() -> None:
    from ovc_experiments.preconditioners import TiledShampooPreconditioner

    layout = MatrixLayout.from_shape((2, 2))
    gradients = torch.tensor(
        [
            [[1.0, 10.0], [100.0, 1000.0]],
            [[2.0, 20.0], [200.0, 2000.0]],
        ],
        dtype=torch.float64,
    )

    tiled = TiledShampooPreconditioner.from_per_example_gradients(
        gradients,
        layout=layout,
        alpha=0.25,
        damping=0.0,
        damping_ratio=0.1,
        relative_eigenvalue_floor=1e-12,
        tile_rows=1,
        tile_cols=1,
        centered=False,
    )

    assert len(tiled.tiles) == 4
    for tile in tiled.tiles:
        preconditioner = tile.preconditioner
        left_min = float(torch.linalg.eigvalsh(preconditioner.left_factor).min().item())
        right_min = float(torch.linalg.eigvalsh(preconditioner.right_factor).min().item())
        assert math.isclose(preconditioner.damping_left / left_min, 0.1, rel_tol=1e-12)
        assert math.isclose(preconditioner.damping_right / right_min, 0.1, rel_tol=1e-12)


def test_hardened_streaming_runner_uses_factorwise_damping() -> None:
    from ovc_experiments.hardened_runner import HardenedBlockConfig, analyze_block_streaming
    from ovc_experiments.safe_operators import DiagonalOperator

    generator = torch.Generator().manual_seed(17)
    gradients = [
        torch.randn((2, 2), generator=generator, dtype=torch.float64)
        * torch.tensor([[1.0, 100.0], [2.0, 200.0]], dtype=torch.float64)
        for _ in range(12)
    ]
    curvature = DiagonalOperator(torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64))

    result = analyze_block_streaming(
        curvature_operator=curvature,
        gradient_factory=lambda: iter(gradients),
        rows=2,
        cols=2,
        example_count=len(gradients),
        config=HardenedBlockConfig(
            centered_moments=False,
            shampoo_damping_ratio=0.1,
            exact_condition_max_dim=16,
        ),
    )

    assert math.isclose(float(result.row["left_rho_over_m"]), 0.1, rel_tol=1e-12)
    assert math.isclose(float(result.row["right_rho_over_m"]), 0.1, rel_tol=1e-12)
    assert not math.isclose(
        float(result.row["shampoo_damping_left"]),
        float(result.row["shampoo_damping_right"]),
        rel_tol=1e-6,
    )
