import torch
from ovc_experiments.safe_operators import DiagonalOperator


def test_diagonal_operator_does_not_materialize_quadratic_storage():
    diagonal = torch.linspace(1.0, 2.0, 589_824, dtype=torch.float64)
    operator = DiagonalOperator(diagonal)
    vector = torch.ones_like(diagonal)
    assert operator.diagonal.numel() == diagonal.numel()
    assert torch.allclose(operator.matvec(vector), diagonal)
    try:
        operator.to_dense(max_dim=1024)
    except MemoryError:
        pass
    else:
        raise AssertionError('large dense materialization must be refused')
