import torch
from ovc_experiments.safe_operators import DiagonalOperator
from ovc_experiments.same_error_dynamics import compare_same_original_error


def test_dynamics_share_identical_original_error():
    H = DiagonalOperator(torch.tensor([1., 4.], dtype=torch.float64))
    P = {'a': DiagonalOperator(torch.ones(2, dtype=torch.float64)), 'b': DiagonalOperator(torch.tensor([1., .25], dtype=torch.float64))}
    e0 = torch.tensor([2., 3.], dtype=torch.float64)
    traces = compare_same_original_error(H, P, e0, steps=1, step_sizes={'a': .1, 'b': .1})
    assert torch.equal(traces['a'].errors[0], e0)
    assert torch.equal(traces['b'].errors[0], e0)
