import torch

from visible_curvature.interventions import reassign_factor


def test_supported_interventions_preserve_factor_spectrum():
    torch.manual_seed(0)
    matrix = torch.randn(5, 5)
    covariance = matrix @ matrix.T + 0.1 * torch.eye(5)
    matrix = torch.randn(5, 5)
    curvature = matrix @ matrix.T + 0.2 * torch.eye(5)
    reference = torch.sort(torch.linalg.eigvalsh(covariance)).values
    for mode in ("observed", "aligned", "reversed"):
        reassigned = reassign_factor(covariance, curvature, mode=mode)
        observed = torch.sort(torch.linalg.eigvalsh(reassigned)).values
        assert torch.allclose(observed, reference, atol=1e-5, rtol=1e-5)
