from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import torch

from .blocks import MatrixLayout
from .geometry import GeometryEstimate, measure_frozen_geometry
from .moments import MomentEstimate, estimate_moments_from_gradients
from .operators import SymmetricLinearOperator
from .preconditioners import ShampooPreconditioner
from .spectrum_utils import factorwise_damping


@dataclass
class SweepPoint:
    label: str
    condition_number: float
    gain: float | None
    censored: bool
    alpha: float | None = None
    damping_left: float | None = None
    damping_right: float | None = None
    rho_over_min: float | None = None
    rho_over_max: float | None = None
    rho_left_over_min: float | None = None
    rho_left_over_max: float | None = None
    rho_right_over_min: float | None = None
    rho_right_over_max: float | None = None
    geometry: GeometryEstimate | None = None

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "label": self.label,
            "condition_number": self.condition_number,
            "gain": self.gain,
            "censored": self.censored,
            "alpha": self.alpha,
            "damping_left": self.damping_left,
            "damping_right": self.damping_right,
            "rho_over_min": self.rho_over_min,
            "rho_over_max": self.rho_over_max,
            "rho_left_over_min": self.rho_left_over_min,
            "rho_left_over_max": self.rho_left_over_max,
            "rho_right_over_min": self.rho_right_over_min,
            "rho_right_over_max": self.rho_right_over_max,
        }
        if self.geometry is not None:
            payload.update(self.geometry.to_dict())
        return payload


@dataclass
class FiniteSamplePoint:
    sample_size: int
    indices: torch.Tensor
    moments: MomentEstimate


def reassign_factor_spectrum(
    factor: torch.Tensor,
    curvature_proxy: torch.Tensor,
    *,
    mode: str,
    seed: int = 0,
) -> torch.Tensor:
    if factor.ndim != 2 or factor.shape[0] != factor.shape[1]:
        raise ValueError("factor must be square")
    if curvature_proxy.shape != factor.shape:
        raise ValueError("factor and curvature proxy shapes differ")
    factor_values = torch.linalg.eigvalsh(0.5 * (factor + factor.T))
    _, curvature_vectors = torch.linalg.eigh(0.5 * (curvature_proxy + curvature_proxy.T))
    if mode == "aligned":
        assigned = factor_values
    elif mode == "reversed":
        assigned = torch.flip(factor_values, dims=[0])
    elif mode == "random":
        generator = torch.Generator(device=factor.device.type if factor.device.type in {"cpu", "cuda"} else "cpu")
        generator.manual_seed(int(seed))
        permutation = torch.randperm(factor_values.numel(), generator=generator, device=factor.device)
        assigned = factor_values[permutation]
    else:
        raise ValueError(f"Unknown assignment mode: {mode}")
    reassigned = (curvature_vectors * assigned.unsqueeze(0)) @ curvature_vectors.T
    return 0.5 * (reassigned + reassigned.T)


def _condition_point(
    curvature: SymmetricLinearOperator,
    left_factor: torch.Tensor,
    right_factor: torch.Tensor,
    *,
    layout: MatrixLayout,
    alpha: float,
    damping_left: float,
    damping_right: float,
    label: str,
    exact_max_dim: int,
    lanczos_steps: int,
    lanczos_starts: int,
    seed: int,
    positive_threshold: float,
    residual_tolerance: float,
    rho_over_min: float | None = None,
    rho_over_max: float | None = None,
    rho_left_over_min: float | None = None,
    rho_left_over_max: float | None = None,
    rho_right_over_min: float | None = None,
    rho_right_over_max: float | None = None,
    subspace_policy: str = "strict_spd",
) -> SweepPoint:
    preconditioner = ShampooPreconditioner(
        left_factor,
        right_factor,
        layout=layout,
        alpha=alpha,
        damping_left=damping_left,
        damping_right=damping_right,
        subspace_policy=subspace_policy,
        name=f"shampoo[{label}]",
    )
    geometry = measure_frozen_geometry(
        curvature,
        preconditioner,
        exact_max_dim=exact_max_dim,
        lanczos_steps=lanczos_steps,
        lanczos_starts=lanczos_starts,
        seed=seed,
        positive_threshold=positive_threshold,
        residual_tolerance=residual_tolerance,
        subspace_policy=subspace_policy,
    )
    condition = geometry.effective_condition.condition_number
    if condition is None:
        condition = float("nan")
    return SweepPoint(
        label=label,
        condition_number=condition,
        gain=geometry.gain,
        censored=geometry.censored,
        alpha=alpha,
        damping_left=damping_left,
        damping_right=damping_right,
        rho_over_min=rho_over_min,
        rho_over_max=rho_over_max,
        rho_left_over_min=rho_left_over_min,
        rho_left_over_max=rho_left_over_max,
        rho_right_over_min=rho_right_over_min,
        rho_right_over_max=rho_right_over_max,
        geometry=geometry,
    )


def alpha_sweep(
    curvature: SymmetricLinearOperator,
    left_factor: torch.Tensor,
    right_factor: torch.Tensor,
    *,
    layout: MatrixLayout,
    alphas: Sequence[float],
    damping: float = 0.0,
    damping_left: float | None = None,
    damping_right: float | None = None,
    exact_max_dim: int = 512,
    lanczos_steps: int = 64,
    lanczos_starts: int = 2,
    seed: int = 0,
    positive_threshold: float = 1e-10,
    residual_tolerance: float = 1e-5,
    subspace_policy: str = "strict_spd",
) -> list[SweepPoint]:
    left_damping = float(damping if damping_left is None else damping_left)
    right_damping = float(damping if damping_right is None else damping_right)
    return [
        _condition_point(
            curvature,
            left_factor,
            right_factor,
            layout=layout,
            alpha=float(alpha),
            damping_left=left_damping,
            damping_right=right_damping,
            label=f"alpha={float(alpha):g}",
            exact_max_dim=exact_max_dim,
            lanczos_steps=lanczos_steps,
            lanczos_starts=lanczos_starts,
            seed=seed + index,
            positive_threshold=positive_threshold,
            residual_tolerance=residual_tolerance,
            subspace_policy=subspace_policy,
        )
        for index, alpha in enumerate(alphas)
    ]


def damping_sweep(
    curvature: SymmetricLinearOperator,
    left_factor: torch.Tensor,
    right_factor: torch.Tensor,
    *,
    layout: MatrixLayout,
    alpha: float,
    normalized_ratios: Sequence[float],
    exact_max_dim: int = 512,
    lanczos_steps: int = 64,
    lanczos_starts: int = 2,
    seed: int = 0,
    positive_threshold: float = 1e-10,
    residual_tolerance: float = 1e-5,
    subspace_policy: str = "strict_spd",
) -> list[SweepPoint]:
    results: list[SweepPoint] = []
    for index, ratio in enumerate(normalized_ratios):
        ratio_float = float(ratio)
        if ratio_float < 0:
            raise ValueError("normalized damping ratios must be nonnegative")
        damping = factorwise_damping(
            left_factor,
            right_factor,
            normalized_ratio=ratio_float,
            relative_threshold=positive_threshold,
        )
        results.append(
            _condition_point(
                curvature,
                left_factor,
                right_factor,
                layout=layout,
                alpha=alpha,
                damping_left=damping.left,
                damping_right=damping.right,
                label=f"rho/min={ratio_float:g}",
                exact_max_dim=exact_max_dim,
                lanczos_steps=lanczos_steps,
                lanczos_starts=lanczos_starts,
                seed=seed + index,
                positive_threshold=positive_threshold,
                residual_tolerance=residual_tolerance,
                rho_over_min=ratio_float,
                rho_over_max=max(damping.left_over_max, damping.right_over_max),
                rho_left_over_min=damping.left_over_min,
                rho_left_over_max=damping.left_over_max,
                rho_right_over_min=damping.right_over_min,
                rho_right_over_max=damping.right_over_max,
                subspace_policy=subspace_policy,
            )
        )
    return results


def assignment_sweep(
    curvature: SymmetricLinearOperator,
    left_factor: torch.Tensor,
    right_factor: torch.Tensor,
    left_curvature_proxy: torch.Tensor,
    right_curvature_proxy: torch.Tensor,
    *,
    layout: MatrixLayout,
    alpha: float = 0.25,
    damping: float = 0.0,
    damping_left: float | None = None,
    damping_right: float | None = None,
    random_repeats: int = 8,
    exact_max_dim: int = 512,
    lanczos_steps: int = 64,
    lanczos_starts: int = 2,
    seed: int = 0,
    positive_threshold: float = 1e-10,
    residual_tolerance: float = 1e-5,
    subspace_policy: str = "strict_spd",
) -> list[SweepPoint]:
    left_damping = float(damping if damping_left is None else damping_left)
    right_damping = float(damping if damping_right is None else damping_right)
    modes: list[tuple[str, int]] = [("aligned", seed), ("reversed", seed)]
    modes.extend((f"random-{index}", seed + 7919 * (index + 1)) for index in range(random_repeats))
    results: list[SweepPoint] = []
    for index, (label, local_seed) in enumerate(modes):
        mode = "random" if label.startswith("random") else label
        left = reassign_factor_spectrum(
            left_factor, left_curvature_proxy, mode=mode, seed=local_seed
        )
        right = reassign_factor_spectrum(
            right_factor, right_curvature_proxy, mode=mode, seed=local_seed + 1
        )
        results.append(
            _condition_point(
                curvature,
                left,
                right,
                layout=layout,
                alpha=alpha,
                damping_left=left_damping,
                damping_right=right_damping,
                label=label,
                exact_max_dim=exact_max_dim,
                lanczos_steps=lanczos_steps,
                lanczos_starts=lanczos_starts,
                seed=seed + index,
                positive_threshold=positive_threshold,
                residual_tolerance=residual_tolerance,
                subspace_policy=subspace_policy,
            )
        )
    return results


def finite_sample_sweep(
    per_example_gradients: torch.Tensor,
    *,
    layout: MatrixLayout,
    sample_sizes: Sequence[int],
    seed: int,
    accumulation_dtype: torch.dtype = torch.float64,
) -> list[FiniteSamplePoint]:
    total = int(per_example_gradients.shape[0])
    if any(size < 1 or size > total for size in sample_sizes):
        raise ValueError(f"sample sizes must lie in [1, {total}]")
    generator = torch.Generator(
        device=per_example_gradients.device.type
        if per_example_gradients.device.type in {"cpu", "cuda"}
        else "cpu"
    )
    generator.manual_seed(int(seed))
    permutation = torch.randperm(total, generator=generator, device=per_example_gradients.device)
    results: list[FiniteSamplePoint] = []
    for size in sample_sizes:
        indices = permutation[: int(size)].clone()
        moments = estimate_moments_from_gradients(
            per_example_gradients[indices],
            layout,
            accumulation_dtype=accumulation_dtype,
        )
        results.append(FiniteSamplePoint(sample_size=int(size), indices=indices, moments=moments))
    return results


def staleness_sweep(
    curvature: SymmetricLinearOperator,
    factor_snapshots: Iterable[tuple[str, torch.Tensor, torch.Tensor]],
    *,
    layout: MatrixLayout,
    alpha: float,
    damping: float = 0.0,
    damping_left: float | None = None,
    damping_right: float | None = None,
    exact_max_dim: int = 512,
    lanczos_steps: int = 64,
    lanczos_starts: int = 2,
    seed: int = 0,
    positive_threshold: float = 1e-10,
    residual_tolerance: float = 1e-5,
) -> list[SweepPoint]:
    left_damping = float(damping if damping_left is None else damping_left)
    right_damping = float(damping if damping_right is None else damping_right)
    results: list[SweepPoint] = []
    for index, (label, left, right) in enumerate(factor_snapshots):
        results.append(
            _condition_point(
                curvature,
                left,
                right,
                layout=layout,
                alpha=alpha,
                damping_left=left_damping,
                damping_right=right_damping,
                label=label,
                exact_max_dim=exact_max_dim,
                lanczos_steps=lanczos_steps,
                lanczos_starts=lanczos_starts,
                seed=seed + index,
                positive_threshold=positive_threshold,
                residual_tolerance=residual_tolerance,
            )
        )
    return results
