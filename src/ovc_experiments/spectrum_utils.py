from __future__ import annotations

from dataclasses import dataclass

import torch

Tensor = torch.Tensor


@dataclass(frozen=True)
class SpectrumSummary:
    """Scale-aware summary of a symmetric matrix spectrum."""

    minimum_active: float | None
    maximum_active: float | None
    threshold: float
    active_rank: int
    numerical_null_rank: int
    negative_rank: int
    dimension: int

    @property
    def has_active_spectrum(self) -> bool:
        return self.minimum_active is not None and self.maximum_active is not None

    @property
    def is_numerically_spd(self) -> bool:
        return self.negative_rank == 0 and self.numerical_null_rank == 0


@dataclass(frozen=True)
class FactorwiseDamping:
    left: float
    right: float
    left_over_min: float
    left_over_max: float
    right_over_min: float
    right_over_max: float
    left_summary: SpectrumSummary
    right_summary: SpectrumSummary


def _threshold(
    *,
    scale: float,
    dimension: int,
    dtype: torch.dtype,
    relative_threshold: float,
    absolute_threshold: float,
) -> float:
    if relative_threshold < 0 or absolute_threshold < 0:
        raise ValueError("spectral thresholds must be nonnegative")
    if not dtype.is_floating_point:
        dtype = torch.float64
    finfo = torch.finfo(dtype)
    return max(
        float(absolute_threshold),
        float(relative_threshold) * max(scale, 0.0),
        64.0 * finfo.eps * dimension * max(scale, finfo.tiny),
    )


def positive_spectrum_summary(
    matrix: Tensor,
    *,
    relative_threshold: float = 1e-10,
    absolute_threshold: float = 0.0,
) -> SpectrumSummary:
    """Return active positive bounds without an absolute-scale fallback.

    Eigenvalues at or below the scale-aware threshold are reported as
    numerically unresolved rather than being replaced by an arbitrary value.
    """

    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"expected a square matrix, got {tuple(matrix.shape)}")
    if matrix.numel() == 0:
        raise ValueError("matrix must be nonempty")
    sym = 0.5 * (matrix + matrix.mT)
    eigenvalues = torch.linalg.eigvalsh(sym)
    if not torch.isfinite(eigenvalues).all():
        raise ValueError("matrix spectrum contains non-finite values")
    scale = float(eigenvalues.abs().max().item())
    threshold = _threshold(
        scale=scale,
        dimension=matrix.shape[0],
        dtype=eigenvalues.dtype,
        relative_threshold=relative_threshold,
        absolute_threshold=absolute_threshold,
    )
    active = eigenvalues[eigenvalues > threshold]
    minimum = float(active.min().item()) if active.numel() else None
    maximum = float(active.max().item()) if active.numel() else None
    return SpectrumSummary(
        minimum_active=minimum,
        maximum_active=maximum,
        threshold=threshold,
        active_rank=int(active.numel()),
        numerical_null_rank=int((eigenvalues.abs() <= threshold).sum().item()),
        negative_rank=int((eigenvalues < -threshold).sum().item()),
        dimension=int(eigenvalues.numel()),
    )


def factorwise_damping(
    left: Tensor,
    right: Tensor,
    *,
    normalized_ratio: float,
    relative_threshold: float = 1e-10,
    absolute_threshold: float = 0.0,
) -> FactorwiseDamping:
    """Scale left and right damping by each factor's own active minimum."""

    ratio = float(normalized_ratio)
    if ratio < 0:
        raise ValueError("normalized damping ratio must be nonnegative")
    left_summary = positive_spectrum_summary(
        left,
        relative_threshold=relative_threshold,
        absolute_threshold=absolute_threshold,
    )
    right_summary = positive_spectrum_summary(
        right,
        relative_threshold=relative_threshold,
        absolute_threshold=absolute_threshold,
    )
    if not left_summary.has_active_spectrum:
        raise ValueError("left factor has no resolved positive active eigenvalue")
    if not right_summary.has_active_spectrum:
        raise ValueError("right factor has no resolved positive active eigenvalue")
    assert left_summary.minimum_active is not None
    assert left_summary.maximum_active is not None
    assert right_summary.minimum_active is not None
    assert right_summary.maximum_active is not None
    left_damping = ratio * left_summary.minimum_active
    right_damping = ratio * right_summary.minimum_active
    return FactorwiseDamping(
        left=left_damping,
        right=right_damping,
        left_over_min=ratio,
        left_over_max=left_damping / left_summary.maximum_active,
        right_over_min=ratio,
        right_over_max=right_damping / right_summary.maximum_active,
        left_summary=left_summary,
        right_summary=right_summary,
    )
