from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any, Mapping

import torch


@dataclass(frozen=True)
class DynamicsTrace:
    name: str
    objective_values: torch.Tensor
    errors: torch.Tensor


def _matvec(operator: Any, vector: torch.Tensor) -> torch.Tensor:
    return operator.matvec(vector) if hasattr(operator, 'matvec') else operator @ vector


def original_error_fingerprint(error: torch.Tensor) -> str:
    """Return a stable SHA-256 fingerprint for an original-coordinate error."""

    array = error.detach().reshape(-1).to(device="cpu").contiguous().numpy()
    return hashlib.sha256(array.tobytes()).hexdigest()


def transformed_initial_for_same_error(
    preconditioner: Any,
    original_error: torch.Tensor,
) -> torch.Tensor:
    """Return z0=P^{-1/2}e0 for the symmetric effective operator.

    The caller must establish that the frozen preconditioner is SPD. An
    explicit active-subspace or singular operator does not define a unique
    full-space inverse square root and is therefore rejected by its strict
    ``apply_power`` implementation.
    """

    if not hasattr(preconditioner, "apply_power"):
        raise TypeError("preconditioner must implement apply_power")
    return preconditioner.apply_power(original_error.detach().reshape(-1), -0.5)


def compare_same_original_error(
    curvature: Any,
    preconditioners: Mapping[str, Any],
    original_error: torch.Tensor,
    *,
    steps: int,
    step_sizes: Mapping[str, float],
) -> dict[str, DynamicsTrace]:
    """Run e_{t+1}=(I-eta P H)e_t with one shared original-coordinate e0."""
    e0 = original_error.detach().reshape(-1)
    results: dict[str, DynamicsTrace] = {}
    for name, preconditioner in preconditioners.items():
        if name not in step_sizes:
            raise KeyError(f'missing step size for {name}')
        e = e0.clone()
        history = [e.clone()]
        objectives = [0.5 * torch.dot(e, _matvec(curvature, e))]
        for _ in range(steps):
            gradient = _matvec(curvature, e)
            direction = preconditioner.apply(gradient) if hasattr(preconditioner, 'apply') else _matvec(preconditioner, gradient)
            e = e - float(step_sizes[name]) * direction
            history.append(e.clone())
            objectives.append(0.5 * torch.dot(e, _matvec(curvature, e)))
        results[name] = DynamicsTrace(name, torch.stack(objectives), torch.stack(history))
    return results
