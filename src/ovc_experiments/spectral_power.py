from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

Tensor = torch.Tensor
SubspacePolicy = Literal['strict_spd', 'positive_active', 'pseudoinverse']


@dataclass(frozen=True)
class SpectralPowerDiagnostics:
    maximum_eigenvalue: float
    numerical_floor: float
    active_rank: int
    null_rank: int
    negative_rank: int
    policy: str


def symmetric_matrix_power(
    matrix: Tensor,
    exponent: float,
    *,
    damping: float = 0.0,
    relative_eigenvalue_floor: float = 1e-12,
    absolute_eigenvalue_floor: float = 0.0,
    subspace_policy: SubspacePolicy = 'strict_spd',
    return_diagnostics: bool = False,
) -> Tensor | tuple[Tensor, SpectralPowerDiagnostics]:
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError('matrix must be square')
    if damping < 0:
        raise ValueError('damping must be non-negative')
    if subspace_policy not in {'strict_spd', 'positive_active', 'pseudoinverse'}:
        raise ValueError(f'unknown subspace policy: {subspace_policy}')
    sym = (matrix + matrix.mT) * 0.5
    if damping:
        sym = sym + damping * torch.eye(sym.shape[0], dtype=sym.dtype, device=sym.device)
    eigenvalues, eigenvectors = torch.linalg.eigh(sym)
    maximum = max(float(eigenvalues.abs().max().item()), 0.0)
    eps = torch.finfo(eigenvalues.dtype).eps
    floor = max(
        float(absolute_eigenvalue_floor),
        float(relative_eigenvalue_floor) * maximum,
        64.0 * eps * matrix.shape[0] * max(maximum, torch.finfo(eigenvalues.dtype).tiny),
    )
    negative = eigenvalues < -floor
    numerical_null = eigenvalues <= 0
    null = eigenvalues.abs() <= floor
    positive = eigenvalues > floor
    if negative.any() and exponent != int(exponent):
        raise ValueError('fractional power of a matrix with negative active eigenvalues is undefined')
    if exponent < 0 and subspace_policy == 'strict_spd' and numerical_null.any():
        raise ValueError('negative powers require SPD input in strict_spd mode; add damping or choose an explicit subspace policy')
    powered = torch.empty_like(eigenvalues)
    if exponent < 0:
        # Tiny positive values are clamped in strict mode; they are never
        # silently deleted. Exact numerical nulls are handled by the selected
        # explicit policy.
        safe = eigenvalues.clamp_min(floor)
        powered.copy_(safe.pow(exponent))
        if subspace_policy in {'positive_active', 'pseudoinverse'}:
            powered[~positive] = 0.0
    else:
        powered.copy_(eigenvalues.clamp_min(0).pow(exponent))
        if exponent == 0:
            powered.fill_(1.0)
    result = (eigenvectors * powered.unsqueeze(0)) @ eigenvectors.mT
    result = (result + result.mT) * 0.5
    diagnostics = SpectralPowerDiagnostics(
        maximum_eigenvalue=maximum,
        numerical_floor=floor,
        active_rank=int(positive.sum().item()),
        null_rank=int(null.sum().item()),
        negative_rank=int(negative.sum().item()),
        policy=subspace_policy,
    )
    return (result, diagnostics) if return_diagnostics else result
