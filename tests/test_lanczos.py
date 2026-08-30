import torch

from visible_curvature.linear_algebra import lanczos, condition_from_spectrum


def test_lanczos_diagonal_extrema():
    vals = torch.tensor([1.0, 2.0, 5.0, 9.0])
    def mv(v):
        return vals * v
    out = lanczos(mv, dim=4, steps=4, device=torch.device('cpu'), seed=0)
    assert abs(out.min_ritz - 1.0) < 1e-4
    assert abs(out.max_ritz - 9.0) < 1e-4
    assert abs(condition_from_spectrum(out.min_ritz, out.max_ritz) - 9.0) < 1e-3
