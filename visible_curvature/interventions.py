from __future__ import annotations

from typing import Literal

import torch

Tensor = torch.Tensor
AssignmentMode = Literal["observed", "aligned", "reversed"]


def reassign_factor(
    covariance_factor: Tensor,
    curvature_factor: Tensor,
    mode: AssignmentMode,
    seed: int = 0,
    curvature_eigendecomp: tuple[Tensor, Tensor] | None = None,
) -> Tensor:
    """Preserve factor eigenvalues while changing only curvature assignment."""
    del seed
    C = 0.5 * (covariance_factor + covariance_factor.T)
    H = 0.5 * (curvature_factor + curvature_factor.T)
    if mode == "observed":
        return C.clone()
    if mode not in {"aligned", "reversed"}:
        raise ValueError("mode must be observed, aligned, or reversed")

    q_sorted = torch.sort(torch.linalg.eigvalsh(C.double())).values.to(device=C.device, dtype=C.dtype)
    if curvature_eigendecomp is None:
        hvals, vectors = torch.linalg.eigh(H.double())
        vectors = vectors.to(device=C.device, dtype=C.dtype)
    else:
        hvals, vectors = curvature_eigendecomp
        hvals = hvals.to(device=C.device)
        vectors = vectors.to(device=C.device, dtype=C.dtype)
    order = torch.argsort(hvals)
    basis = vectors[:, order]
    assigned = q_sorted if mode == "aligned" else torch.flip(q_sorted, dims=[0])
    result = (basis * assigned.unsqueeze(0)) @ basis.T
    return 0.5 * (result + result.T)


def build_factor_intervention(
    left: Tensor,
    right: Tensor,
    H_left: Tensor,
    H_right: Tensor,
    mode: AssignmentMode,
    seed: int = 0,
    left_curvature_eigendecomp: tuple[Tensor, Tensor] | None = None,
    right_curvature_eigendecomp: tuple[Tensor, Tensor] | None = None,
) -> tuple[Tensor, Tensor]:
    return (
        reassign_factor(
            left,
            H_left,
            mode=mode,
            seed=seed,
            curvature_eigendecomp=left_curvature_eigendecomp,
        ),
        reassign_factor(
            right,
            H_right,
            mode=mode,
            seed=seed + 1,
            curvature_eigendecomp=right_curvature_eigendecomp,
        ),
    )
