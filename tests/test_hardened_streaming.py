import torch
from ovc_experiments.streaming_moments import accumulate_matrix_moments


def test_streaming_moments_match_batch_formulas():
    torch.manual_seed(0)
    gradients = [torch.randn(3, 2, dtype=torch.float64) for _ in range(17)]
    stats = accumulate_matrix_moments(iter(gradients), 3, 2)
    batch = torch.stack(gradients)
    mean = batch.mean(0)
    centered = batch - mean
    assert torch.allclose(stats.mean, mean, atol=1e-12, rtol=1e-12)
    assert torch.allclose(stats.adam_diagonal_centered, centered.square().mean(0).reshape(-1), atol=1e-12, rtol=1e-12)
    assert torch.allclose(stats.left_centered, torch.einsum('nrc,nsc->rs', centered, centered) / len(gradients), atol=1e-12, rtol=1e-12)
    assert torch.allclose(stats.right_centered, torch.einsum('nrc,nrd->cd', centered, centered) / len(gradients), atol=1e-12, rtol=1e-12)
