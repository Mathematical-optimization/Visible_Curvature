from __future__ import annotations

import math

import torch

from ovc_experiments.blocks import BlockSpec, MatrixLayout
from ovc_experiments.dynamics import (
    run_chebyshev,
    run_conjugate_gradient,
    run_frozen_block_continuation,
    run_gradient_descent,
)
from ovc_experiments.functional import FunctionalBlockModel
from ovc_experiments.operators import CongruenceOperator, DenseOperator
from ovc_experiments.preconditioners import AdamPreconditioner, IdentityPreconditioner, ScaledPreconditioner
from ovc_experiments.tasks import ClassificationTask


def test_optimal_constant_step_has_endpoint_contraction() -> None:
    operator = DenseOperator(torch.diag(torch.tensor([1.0, 3.0], dtype=torch.float64)))
    initial = torch.tensor([1.0, 2.0], dtype=torch.float64)
    result = run_gradient_descent(operator, initial, steps=4, eigen_min=1.0, eigen_max=3.0)
    expected = torch.tensor([1.0, 0.25, 0.25**2, 0.25**3, 0.25**4], dtype=torch.float64)
    assert torch.allclose(result.relative_objective, expected, atol=1e-12, rtol=1e-12)


def test_chebyshev_terminal_error_obeys_minimax_envelope() -> None:
    values = torch.linspace(1.0, 25.0, 20, dtype=torch.float64)
    operator = DenseOperator(torch.diag(values))
    initial = torch.ones(20, dtype=torch.float64)
    steps = 6
    result = run_chebyshev(operator, initial, steps=steps, eigen_min=1.0, eigen_max=25.0)
    gamma = math.acosh((25.0 + 1.0) / (25.0 - 1.0))
    envelope = 1.0 / math.cosh(steps * gamma) ** 2
    assert result.relative_objective[-1].item() <= envelope + 1e-10


def test_conjugate_gradient_terminates_in_number_of_distinct_eigenvalues() -> None:
    operator = DenseOperator(torch.diag(torch.tensor([1.0, 4.0, 9.0], dtype=torch.float64)))
    initial = torch.tensor([1.0, -2.0, 3.0], dtype=torch.float64)
    result = run_conjugate_gradient(operator, initial, steps=3, tolerance=1e-14)
    assert result.relative_objective[-1].item() < 1e-20


def test_scalar_grafting_gives_same_optimal_gd_curve() -> None:
    layout = MatrixLayout.from_shape((2, 2))
    hessian = DenseOperator(torch.diag(torch.tensor([1.0, 2.0, 8.0, 32.0], dtype=torch.float64)))
    base = AdamPreconditioner(torch.tensor([1.0, 4.0, 9.0, 16.0], dtype=torch.float64), layout=layout)
    scaled = ScaledPreconditioner(base, 13.0)
    effective_base = CongruenceOperator(hessian, base.apply_sqrt)
    effective_scaled = CongruenceOperator(hessian, scaled.apply_sqrt)
    initial = torch.arange(1, 5, dtype=torch.float64)

    first = run_gradient_descent(effective_base, initial, steps=5)
    second = run_gradient_descent(effective_scaled, initial, steps=5)
    assert torch.allclose(first.relative_objective, second.relative_objective, atol=1e-10, rtol=1e-10)


def test_frozen_block_continuation_decreases_fixed_batch_loss() -> None:
    model = torch.nn.Linear(2, 2, bias=False, dtype=torch.float64)
    with torch.no_grad():
        model.weight.zero_()
    batch = {
        "inputs": torch.tensor([[2.0, 0.0], [0.0, 2.0]], dtype=torch.float64),
        "targets": torch.tensor([0, 1], dtype=torch.long),
    }
    block = BlockSpec("weight", (2, 2), MatrixLayout.from_shape((2, 2)), 4)
    functional = FunctionalBlockModel(model, block, ClassificationTask())
    preconditioner = IdentityPreconditioner(block.layout, dtype=torch.float64, device="cpu")
    result = run_frozen_block_continuation(
        functional,
        batch,
        preconditioner,
        steps=10,
        step_size=0.2,
    )
    assert result.losses[-1] < result.losses[0]
    assert len(result.losses) == 11
