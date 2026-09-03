from __future__ import annotations

import torch

from ovc_experiments.blocks import BlockSpec, MatrixLayout
from ovc_experiments.curvature import (
    CrossEntropyGGNOperator,
    EmpiricalFisherOperator,
    ExactHessianOperator,
)
from ovc_experiments.functional import FunctionalBlockModel, collect_per_example_gradients
from ovc_experiments.operators import materialize
from ovc_experiments.tasks import ClassificationTask


def _linear_problem() -> tuple[torch.nn.Module, dict[str, torch.Tensor]]:
    model = torch.nn.Linear(2, 3, bias=False, dtype=torch.float64)
    with torch.no_grad():
        model.weight.copy_(
            torch.tensor(
                [[0.3, -0.2], [0.1, 0.4], [-0.5, 0.2]], dtype=torch.float64
            )
        )
    batch = {
        "inputs": torch.tensor([[1.0, 2.0], [-1.0, 0.5], [0.2, -0.7]], dtype=torch.float64),
        "targets": torch.tensor([0, 2, 1], dtype=torch.long),
    }
    return model, batch


def test_per_example_gradient_loop_and_vmap_match() -> None:
    model, batch = _linear_problem()
    task = ClassificationTask()
    block = BlockSpec(
        name="weight",
        shape=(3, 2),
        layout=MatrixLayout.from_shape((3, 2)),
        numel=6,
    )
    functional = FunctionalBlockModel(model, block, task)

    loop = collect_per_example_gradients(functional, batch, backend="loop")
    vectorized = collect_per_example_gradients(functional, batch, backend="vmap")

    assert loop.shape == (3, 3, 2)
    assert torch.allclose(loop, vectorized, atol=1e-10, rtol=1e-10)
    assert torch.allclose(loop.mean(dim=0), model.weight.grad if model.weight.grad is not None else loop.mean(dim=0))


def test_empirical_fisher_matches_gradient_gram_matrix() -> None:
    model, batch = _linear_problem()
    task = ClassificationTask()
    block = BlockSpec("weight", (3, 2), MatrixLayout.from_shape((3, 2)), 6)
    functional = FunctionalBlockModel(model, block, task)
    gradients = collect_per_example_gradients(functional, batch, backend="loop")
    operator = EmpiricalFisherOperator(gradients, block.layout)

    flat = gradients.reshape(gradients.shape[0], -1)
    expected = flat.T @ flat / flat.shape[0]
    assert torch.allclose(materialize(operator), expected, atol=1e-10, rtol=1e-10)


def test_cross_entropy_ggn_matches_explicit_loss_hessian_for_linear_model() -> None:
    model, batch = _linear_problem()
    task = ClassificationTask()
    block = BlockSpec("weight", (3, 2), MatrixLayout.from_shape((3, 2)), 6)
    functional = FunctionalBlockModel(model, block, task)

    ggn = CrossEntropyGGNOperator(functional, batch)
    hessian = ExactHessianOperator(functional, batch)
    ggn_matrix = materialize(ggn)
    hessian_matrix = materialize(hessian)

    assert torch.allclose(ggn_matrix, hessian_matrix, atol=1e-9, rtol=1e-9)
    assert torch.linalg.eigvalsh(ggn_matrix).min().item() > -1e-9


def test_exact_hessian_matches_torch_func_hessian() -> None:
    model, batch = _linear_problem()
    task = ClassificationTask()
    block = BlockSpec("weight", (3, 2), MatrixLayout.from_shape((3, 2)), 6)
    functional = FunctionalBlockModel(model, block, task)
    operator = ExactHessianOperator(functional, batch)

    explicit_tensor = torch.func.hessian(lambda p: functional.mean_loss(p, batch))(
        functional.block_parameter
    )
    expected = explicit_tensor.reshape(6, 6)
    assert torch.allclose(materialize(operator), expected, atol=1e-9, rtol=1e-9)
