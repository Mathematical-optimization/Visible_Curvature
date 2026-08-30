from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from .linear_algebra import symmetric_eigen_powers_with_metadata

Tensor = torch.Tensor


@dataclass
class SymmetricSpectrum:
    """Reusable float64 eigendecomposition for a symmetric factor."""

    values: Tensor
    vectors: Tensor
    output_dtype: torch.dtype

    @classmethod
    def from_matrix(cls, matrix: Tensor) -> "SymmetricSpectrum":
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError("matrix must be square")
        work = (0.5 * (matrix + matrix.T)).to(dtype=torch.float64)
        values, vectors = torch.linalg.eigh(work)
        return cls(values=values, vectors=vectors, output_dtype=matrix.dtype)

    def powers(
        self,
        powers: tuple[float, ...],
        *,
        damping: float,
        eig_floor: float,
        relative_eig_floor: float,
    ) -> tuple[dict[float, Tensor], dict[str, float]]:
        if damping < 0 or eig_floor < 0 or relative_eig_floor < 0:
            raise ValueError("damping and eigenvalue floors must be nonnegative")
        shifted = self.values + float(damping)
        raw_min = float(shifted.min().detach().cpu()) if shifted.numel() else float("nan")
        raw_max = float(shifted.max().detach().cpu()) if shifted.numel() else float("nan")
        scale = max(
            abs(raw_min) if math.isfinite(raw_min) else 0.0,
            abs(raw_max) if math.isfinite(raw_max) else 0.0,
            torch.finfo(torch.float64).tiny,
        )
        effective_floor = max(
            float(eig_floor),
            float(relative_eig_floor) * torch.finfo(torch.float64).eps * scale,
            torch.finfo(torch.float64).tiny,
        )
        floor_mask = shifted < effective_floor
        clamped = shifted.clamp_min(effective_floor)
        matrices = {
            float(power): (
                (self.vectors * clamped.pow(float(power)).unsqueeze(0)) @ self.vectors.T
            ).to(dtype=self.output_dtype)
            for power in powers
        }
        metadata = {
            "effective_eig_floor": float(effective_floor),
            "floored_fraction": float(floor_mask.double().mean().detach().cpu()) if shifted.numel() else 0.0,
            "raw_min_eigenvalue": raw_min,
            "raw_max_eigenvalue": raw_max,
        }
        return matrices, metadata


def resolve_relative_damping(
    values: Tensor,
    coefficient: float,
    statistic: str = "median",
    minimum: float = 0.0,
) -> float:
    """Resolve a block-relative damping value from nonnegative statistics."""
    x = values.detach().double().flatten()
    x = x[torch.isfinite(x) & (x >= 0)]
    if x.numel() == 0:
        scale = 0.0
    elif statistic == "median":
        scale = float(torch.median(x).cpu())
    elif statistic in {"max", "lambda_max"}:
        scale = float(torch.max(x).cpu())
    elif statistic == "mean":
        scale = float(torch.mean(x).cpu())
    elif statistic == "trace_mean":
        scale = float(torch.sum(x).cpu()) / max(int(x.numel()), 1)
    else:
        raise ValueError(f"Unknown relative damping statistic: {statistic}")
    return max(float(minimum), float(coefficient) * scale)


def resolve_factor_damping(
    factor: Tensor,
    coefficient: float,
    statistic: str = "lambda_max",
    minimum: float = 0.0,
) -> float:
    factor = 0.5 * (factor + factor.T)
    if statistic in {"lambda_max", "max"}:
        scale = max(float(torch.linalg.eigvalsh(factor.double()).max().detach().cpu()), 0.0)
    elif statistic == "trace_mean":
        scale = max(float(torch.trace(factor.double()).detach().cpu()) / max(factor.shape[0], 1), 0.0)
    else:
        return resolve_relative_damping(torch.diag(factor), coefficient, statistic, minimum)
    return max(float(minimum), float(coefficient) * scale)


@dataclass
class AdamFormPreconditioner:
    diag_cov: Tensor
    damping: float
    relative_floor: float = 64.0
    absolute_floor: float = 0.0

    def __post_init__(self) -> None:
        if float(self.damping) < 0.0:
            raise ValueError("damping must be nonnegative")
        if float(self.relative_floor) < 0.0 or float(self.absolute_floor) < 0.0:
            raise ValueError("Adam floors must be nonnegative")
        self.diag_cov = self.diag_cov.clamp_min(0)
        base = self.diag_cov + float(self.damping)
        finite_positive = base[torch.isfinite(base) & (base > 0)]
        if finite_positive.numel():
            scale = float(torch.median(finite_positive.detach().double()).cpu())
        else:
            scale = 1.0
        dtype_eps = torch.finfo(base.dtype).eps
        self.effective_floor = max(
            float(self.absolute_floor),
            float(self.relative_floor) * dtype_eps * max(scale, torch.finfo(base.dtype).tiny),
            torch.finfo(base.dtype).tiny,
        )
        floor_mask = base < self.effective_floor
        base = base.clamp_min(self.effective_floor)
        self.floored_fraction = float(floor_mask.float().mean().detach().cpu()) if base.numel() else 0.0
        self._p = base.pow(-0.5)
        self._p_half = base.pow(-0.25)
        self.max_entry = float(self._p.max().detach().cpu()) if self._p.numel() else float("nan")

    def apply(self, V: Tensor) -> Tensor:
        return self._p * V

    def apply_half(self, V: Tensor) -> Tensor:
        return self._p_half * V

    @property
    def floor_lambda(self) -> float:
        return max(float(self.damping), float(self.effective_floor))

    @property
    def fro_norm_sq(self) -> float:
        return float(torch.sum(self._p.square()).detach().cpu())


@dataclass
class ShampooFormPreconditioner:
    left: Tensor
    right: Tensor
    damping: float | tuple[float, float]
    factor_exponent: float = 0.25
    eig_floor: float = 0.0
    relative_eig_floor: float = 64.0
    left_spectrum: SymmetricSpectrum | None = None
    right_spectrum: SymmetricSpectrum | None = None

    def __post_init__(self) -> None:
        alpha = float(self.factor_exponent)
        if not 0.0 < alpha <= 0.5:
            raise ValueError("factor_exponent must lie in (0, 0.5]")
        self.factor_exponent = alpha
        self.left = 0.5 * (self.left + self.left.T)
        self.right = 0.5 * (self.right + self.right.T)
        if isinstance(self.damping, tuple):
            self.left_damping, self.right_damping = map(float, self.damping)
        else:
            self.left_damping = self.right_damping = float(self.damping)
        if self.left_damping < 0.0 or self.right_damping < 0.0:
            raise ValueError("damping must be nonnegative")

        powers = (-alpha, -alpha / 2.0)
        if self.left_spectrum is None:
            left_powers, left_meta = symmetric_eigen_powers_with_metadata(
                self.left,
                powers,
                damping=self.left_damping,
                eig_floor=self.eig_floor,
                relative_eig_floor=self.relative_eig_floor,
            )
        else:
            left_powers, left_meta = self.left_spectrum.powers(
                powers,
                damping=self.left_damping,
                eig_floor=self.eig_floor,
                relative_eig_floor=self.relative_eig_floor,
            )
        if self.right_spectrum is None:
            right_powers, right_meta = symmetric_eigen_powers_with_metadata(
                self.right,
                powers,
                damping=self.right_damping,
                eig_floor=self.eig_floor,
                relative_eig_floor=self.relative_eig_floor,
            )
        else:
            right_powers, right_meta = self.right_spectrum.powers(
                powers,
                damping=self.right_damping,
                eig_floor=self.eig_floor,
                relative_eig_floor=self.relative_eig_floor,
            )
        self.L_apply = left_powers[-alpha]
        self.R_apply = right_powers[-alpha]
        self.L_half = left_powers[-alpha / 2.0]
        self.R_half = right_powers[-alpha / 2.0]

        # Backward-compatible names for the practical alpha=1/4 path.
        self.L_m14 = self.L_apply
        self.R_m14 = self.R_apply
        self.L_m18 = self.L_half
        self.R_m18 = self.R_half

        self.left_effective_eig_floor = float(left_meta["effective_eig_floor"])
        self.right_effective_eig_floor = float(right_meta["effective_eig_floor"])
        self.left_floored_fraction = float(left_meta["floored_fraction"])
        self.right_floored_fraction = float(right_meta["floored_fraction"])
        self.left_raw_min_eigenvalue = float(left_meta["raw_min_eigenvalue"])
        self.right_raw_min_eigenvalue = float(right_meta["raw_min_eigenvalue"])

    def apply(self, V: Tensor) -> Tensor:
        return self.L_apply @ V @ self.R_apply

    def apply_half(self, V: Tensor) -> Tensor:
        return self.L_half @ V @ self.R_half

    @property
    def floor_lambda(self) -> float:
        left = max(self.left_damping, self.left_effective_eig_floor, 0.0)
        right = max(self.right_damping, self.right_effective_eig_floor, 0.0)
        return math.sqrt(left * right)

    @property
    def fro_norm_sq(self) -> float:
        return float((torch.sum(self.L_apply.square()) * torch.sum(self.R_apply.square())).detach().cpu())


@dataclass
class ScalarPreconditioner:
    scale: float = 1.0

    def __post_init__(self) -> None:
        if float(self.scale) <= 0.0:
            raise ValueError("scale must be positive")

    def apply(self, V: Tensor) -> Tensor:
        return V * float(self.scale)

    def apply_half(self, V: Tensor) -> Tensor:
        return V * math.sqrt(float(self.scale))

    @property
    def floor_lambda(self) -> float:
        return float(self.scale)

    @property
    def fro_norm_sq(self) -> float:
        return float("nan")


def kronecker_consistent_adam_diag(left_factor: Tensor, right_factor: Tensor) -> Tensor:
    r"""Return ``diag(Sigma)`` for a covariance consistent with the factors."""
    tL = torch.trace(left_factor)
    tR = torch.trace(right_factor)
    t = 0.5 * (tL + tR)
    if float(t) <= 0:
        raise ValueError("Factors must have positive trace")
    return torch.outer(torch.diag(left_factor), torch.diag(right_factor)) / t
