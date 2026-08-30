import torch

from visible_curvature.covariance import CovarianceState, merge_states


def test_streaming_covariance_matches_direct():
    torch.manual_seed(0)
    xs = [torch.randn(4, 3) for _ in range(7)]
    s = CovarianceState.zeros((4, 3), dtype=torch.float64)
    for x in xs:
        s.update(x)
    est = s.finalize(ddof=0, dtype=torch.float64)
    X = torch.stack(xs).double()
    mu = X.mean(0)
    Z = X - mu
    diag = (Z * Z).mean(0)
    left = torch.einsum('nij,nkj->ik', Z, Z) / X.shape[0]
    right = torch.einsum('nji,njk->ik', Z, Z) / X.shape[0]
    assert torch.allclose(est.mean, mu, atol=1e-10)
    assert torch.allclose(est.diag, diag, atol=1e-10)
    assert torch.allclose(est.left, left, atol=1e-10)
    assert torch.allclose(est.right, right, atol=1e-10)


def test_merge_states_matches_single_pass():
    torch.manual_seed(1)
    xs = [torch.randn(3, 2) for _ in range(8)]
    groups = []
    for part in [xs[:3], xs[3:6], xs[6:]]:
        s = CovarianceState.zeros((3, 2), dtype=torch.float64)
        for x in part:
            s.update(x)
        groups.append(s)
    merged = merge_states(groups).finalize(dtype=torch.float64)
    direct = CovarianceState.zeros((3, 2), dtype=torch.float64)
    for x in xs:
        direct.update(x)
    direct = direct.finalize(dtype=torch.float64)
    assert torch.allclose(merged.mean, direct.mean, atol=1e-10)
    assert torch.allclose(merged.diag, direct.diag, atol=1e-10)
    assert torch.allclose(merged.left, direct.left, atol=1e-10)
    assert torch.allclose(merged.right, direct.right, atol=1e-10)
