import torch

from visible_curvature.curvature import LinearMatrixOperator, estimate_partial_traces


def test_partial_trace_hutchinson_on_kronecker_operator():
    torch.manual_seed(0)
    L = torch.tensor([[2.0, 0.3], [0.3, 1.0]])
    R = torch.tensor([[1.5, 0.2, 0.0], [0.2, 1.0, 0.1], [0.0, 0.1, 0.7]])
    # Operator H(V)=L V R. Partial traces are tr(R)L and tr(L)R.
    op = LinearMatrixOperator((2, 3), lambda V: L @ V @ R, torch.device('cpu'), torch.float32)
    HL, HR = estimate_partial_traces(op, num_probes=5000, seed=0)
    assert torch.allclose(HL, torch.trace(R) * L, atol=0.08, rtol=0.05)
    assert torch.allclose(HR, torch.trace(L) * R, atol=0.08, rtol=0.05)
