import math
import torch

from visible_curvature.curvature import LinearMatrixOperator, stabilize_curvature


def test_relative_ridge_scales_with_raw_top_eigenvalue():
    H = torch.diag(torch.tensor([-2.0, 10.0], dtype=torch.float64))
    op = LinearMatrixOperator((1, 2), lambda V: (H @ V.reshape(-1)).reshape(1, 2), torch.device("cpu"), torch.float64)
    result = stabilize_curvature(
        op,
        psd_mode="shift",
        ridge=1e-2,
        ridge_mode="relative_max",
        lanczos_steps=2,
        lanczos_starts=1,
        seed=0,
    )
    # target positive floor is 0.01 * lambda_max = 0.1, so shift is 2.1.
    assert math.isclose(result.target_ridge, 0.1, rel_tol=1e-10)
    assert math.isclose(result.shift, 2.1, rel_tol=1e-10)
