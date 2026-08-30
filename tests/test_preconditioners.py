import torch

from visible_curvature.preconditioners import AdamFormPreconditioner, ShampooFormPreconditioner, kronecker_consistent_adam_diag


def test_adam_and_shampoo_shapes():
    torch.manual_seed(0)
    V = torch.randn(4, 3)
    diag = torch.rand(4, 3)
    Pad = AdamFormPreconditioner(diag, damping=1e-3)
    assert Pad.apply(V).shape == V.shape
    L = torch.randn(4, 4); L = L @ L.T + 0.1 * torch.eye(4)
    R = torch.randn(3, 3); R = R @ R.T + 0.1 * torch.eye(3)
    Psh = ShampooFormPreconditioner(L, R, damping=1e-3)
    assert Psh.apply(V).shape == V.shape
    diag_kron = kronecker_consistent_adam_diag(L, R)
    assert diag_kron.shape == V.shape
