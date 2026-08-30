import math

import torch

from visible_curvature.covariance import CovarianceState
from visible_curvature.preconditioners import ShampooFormPreconditioner, resolve_relative_damping


def test_shampoo_accepts_distinct_left_and_right_damping():
    left = torch.diag(torch.tensor([1.0, 4.0], dtype=torch.float64))
    right = torch.diag(torch.tensor([9.0, 16.0, 25.0], dtype=torch.float64))
    p = ShampooFormPreconditioner(left, right, damping=(0.5, 2.0))
    assert p.left_damping == 0.5
    assert p.right_damping == 2.0
    assert math.isclose(p.floor_lambda, 1.0)
    v = torch.ones(2, 3, dtype=torch.float64)
    assert p.apply(v).shape == v.shape


def test_relative_damping_uses_declared_block_scale():
    diag = torch.tensor([1.0, 2.0, 100.0])
    got = resolve_relative_damping(diag, coefficient=1e-2, statistic="median")
    assert math.isclose(got, 0.02)


def test_covariance_can_return_uncentered_second_moment():
    state = CovarianceState.zeros((2, 2), dtype=torch.float64)
    state.update(torch.tensor([[1.0, 0.0], [0.0, 0.0]], dtype=torch.float64))
    state.update(torch.tensor([[3.0, 0.0], [0.0, 0.0]], dtype=torch.float64))
    centered = state.finalize(dtype=torch.float64)
    uncentered = centered.uncentered()
    assert torch.allclose(uncentered.diag, centered.diag + centered.mean.square())
    assert torch.allclose(uncentered.left, centered.left + centered.mean @ centered.mean.T)
