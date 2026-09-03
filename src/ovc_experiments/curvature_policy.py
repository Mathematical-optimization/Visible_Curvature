from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Iterator, Literal

import torch

from .safe_operators import FunctionOperator

CurvatureKind = Literal['ggn', 'hessian', 'empirical_fisher']


def normalize_curvature_kind(kind: str) -> CurvatureKind:
    normalized = kind.strip().lower().replace('-', '_')
    aliases = {'fisher': 'empirical_fisher', 'ef': 'empirical_fisher', 'generalized_gauss_newton': 'ggn'}
    normalized = aliases.get(normalized, normalized)
    if normalized not in {'ggn', 'hessian', 'empirical_fisher'}:
        raise ValueError(f'unknown curvature kind: {kind}')
    return normalized  # type: ignore[return-value]


@dataclass(frozen=True)
class CurvaturePolicy:
    kind: CurvatureKind = 'ggn'
    primary_analysis: bool = True
    allow_empirical_fisher_primary: bool = False

    def validate(self) -> 'CurvaturePolicy':
        if self.kind == 'empirical_fisher' and self.primary_analysis and not self.allow_empirical_fisher_primary:
            raise ValueError(
                'empirical Fisher shares the per-example-gradient Gram object with the optimizer statistic; '
                'use GGN/Hessian for primary assignment claims or mark this panel control-only'
            )
        return self


def validate_curvature_policy(
    kind: str,
    *,
    primary_analysis: bool = True,
    allow_empirical_fisher_primary: bool = False,
) -> CurvaturePolicy:
    return CurvaturePolicy(
        kind=normalize_curvature_kind(kind),
        primary_analysis=primary_analysis,
        allow_empirical_fisher_primary=allow_empirical_fisher_primary,
    ).validate()


def empirical_fisher_operator(
    gradient_factory: Callable[[], Iterable[torch.Tensor]],
    dimension: int,
    *,
    count: int,
    mean_gradient: torch.Tensor | None = None,
    centered: bool = False,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str = 'cpu',
) -> FunctionOperator:
    """Memory-safe empirical-Fisher operator using a replayable gradient stream."""
    if count <= 0:
        raise ValueError('count must be positive')
    dev = torch.device(device)
    mean = None if mean_gradient is None else mean_gradient.reshape(-1).to(device=dev, dtype=dtype)
    if centered and mean is None:
        raise ValueError('centered empirical Fisher requires mean_gradient')

    def matvec(vector: torch.Tensor) -> torch.Tensor:
        v = vector.reshape(-1).to(device=dev, dtype=dtype)
        out = torch.zeros_like(v)
        observed = 0
        for gradient in gradient_factory():
            g = gradient.detach().reshape(-1).to(device=dev, dtype=dtype)
            if g.numel() != dimension:
                raise ValueError('gradient dimension changed between stream replays')
            if centered:
                g = g - mean
            out.add_(g, alpha=float(torch.dot(g, v).item()))
            observed += 1
        if observed != count:
            raise ValueError(f'expected {count} gradients, observed {observed}')
        return out / count

    return FunctionOperator(dimension=dimension, function=matvec, dtype=dtype, device=dev, name='empirical_fisher_control')
