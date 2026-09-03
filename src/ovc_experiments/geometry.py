from __future__ import annotations

from dataclasses import dataclass
import math

from .operators import CongruenceOperator, SymmetricLinearOperator
from .preconditioners import FrozenPreconditioner
from .spectral import ConditionEstimate, estimate_condition


@dataclass
class GeometryEstimate:
    curvature_condition: ConditionEstimate
    effective_condition: ConditionEstimate
    gain: float | None
    preconditioner_name: str

    @property
    def censored(self) -> bool:
        return self.curvature_condition.censored or self.effective_condition.censored

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "preconditioner_name": self.preconditioner_name,
            "gain": self.gain,
            "censored": self.censored,
        }
        payload.update({f"curvature_{key}": value for key, value in self.curvature_condition.to_dict().items()})
        payload.update({f"effective_{key}": value for key, value in self.effective_condition.to_dict().items()})
        return payload


def measure_frozen_geometry(
    curvature: SymmetricLinearOperator,
    preconditioner: FrozenPreconditioner,
    *,
    exact_max_dim: int = 512,
    lanczos_steps: int = 64,
    lanczos_starts: int = 2,
    seed: int = 0,
    positive_threshold: float = 1e-10,
    residual_tolerance: float = 1e-5,
    subspace_policy: str = "strict_spd",
) -> GeometryEstimate:
    if curvature.dimension != preconditioner.dimension:
        raise ValueError(
            f"Curvature dimension {curvature.dimension} and preconditioner dimension "
            f"{preconditioner.dimension} differ"
        )
    curvature_condition = estimate_condition(
        curvature,
        exact_max_dim=exact_max_dim,
        lanczos_steps=lanczos_steps,
        starts=lanczos_starts,
        seed=seed,
        positive_threshold=positive_threshold,
        residual_tolerance=residual_tolerance,
        subspace_policy=subspace_policy,
    )
    effective = CongruenceOperator(
        curvature,
        preconditioner.apply_sqrt,
        name=f"effective[{preconditioner.name}]",
    )
    try:
        effective_condition = estimate_condition(
            effective,
            exact_max_dim=exact_max_dim,
            lanczos_steps=lanczos_steps,
            starts=lanczos_starts,
            seed=seed + 1_000_003,
            positive_threshold=positive_threshold,
            residual_tolerance=residual_tolerance,
            subspace_policy=subspace_policy,
        )
    except ValueError as error:
        effective_condition = ConditionEstimate(
            min_eigenvalue=None,
            max_eigenvalue=None,
            condition_number=None,
            min_residual=None,
            max_residual=None,
            censored=True,
            censor_reason=f"invalid_preconditioner:{error}",
            method="preconditioner_validation",
            positive_threshold=positive_threshold,
        )
    gain: float | None = None
    if (
        curvature_condition.condition_number is not None
        and effective_condition.condition_number is not None
        and curvature_condition.condition_number > 0
        and effective_condition.condition_number > 0
    ):
        gain = math.log(curvature_condition.condition_number) - math.log(
            effective_condition.condition_number
        )
    return GeometryEstimate(
        curvature_condition=curvature_condition,
        effective_condition=effective_condition,
        gain=gain,
        preconditioner_name=preconditioner.name,
    )
