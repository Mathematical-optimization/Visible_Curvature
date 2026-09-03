from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from .functional import FunctionalBlockModel
from .operators import SymmetricLinearOperator
from .preconditioners import FrozenPreconditioner
from .spectral import estimate_condition
from .tasks import Batch


@dataclass
class QuadraticTrajectory:
    method: str
    relative_objective: torch.Tensor
    gradient_norms: torch.Tensor
    step_sizes: torch.Tensor
    final_vector: torch.Tensor

    def to_dict(self) -> dict[str, object]:
        return {
            "method": self.method,
            "relative_objective": self.relative_objective.detach().cpu().tolist(),
            "gradient_norms": self.gradient_norms.detach().cpu().tolist(),
            "step_sizes": self.step_sizes.detach().cpu().tolist(),
        }


@dataclass
class BlockContinuationResult:
    losses: list[float]
    gradient_norms: list[float]
    final_parameter: torch.Tensor
    step_size: float


def quadratic_energy(operator: SymmetricLinearOperator, vector: torch.Tensor) -> torch.Tensor:
    return 0.5 * torch.dot(vector, operator.matvec(vector))


def _resolve_bounds(
    operator: SymmetricLinearOperator,
    eigen_min: float | None,
    eigen_max: float | None,
) -> tuple[float, float]:
    if eigen_min is not None and eigen_max is not None:
        if eigen_min <= 0 or eigen_max < eigen_min:
            raise ValueError("Invalid positive eigenvalue bounds")
        return float(eigen_min), float(eigen_max)
    estimate = estimate_condition(
        operator,
        exact_max_dim=max(512, operator.dimension),
        lanczos_steps=min(128, operator.dimension),
        starts=2,
    )
    if estimate.min_eigenvalue is None or estimate.max_eigenvalue is None:
        raise ValueError("Unable to resolve positive eigenvalue bounds")
    return estimate.min_eigenvalue, estimate.max_eigenvalue


def _relative_curve(
    operator: SymmetricLinearOperator,
    vectors: list[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    energies = torch.stack([quadratic_energy(operator, vector) for vector in vectors])
    gradients = torch.stack(
        [torch.linalg.vector_norm(operator.matvec(vector)) for vector in vectors]
    )
    initial = energies[0]
    if initial <= 0:
        raise ValueError("Initial quadratic energy must be positive")
    return energies / initial, gradients


def run_gradient_descent(
    operator: SymmetricLinearOperator,
    initial: torch.Tensor,
    *,
    steps: int,
    step_size: float | None = None,
    eigen_min: float | None = None,
    eigen_max: float | None = None,
) -> QuadraticTrajectory:
    if steps < 0:
        raise ValueError("steps must be nonnegative")
    operator._validate_vector(initial)
    minimum, maximum = _resolve_bounds(operator, eigen_min, eigen_max)
    eta = 2.0 / (minimum + maximum) if step_size is None else float(step_size)
    vector = initial.clone()
    vectors = [vector.clone()]
    for _ in range(steps):
        vector = vector - eta * operator.matvec(vector)
        vectors.append(vector.clone())
    relative, gradient_norms = _relative_curve(operator, vectors)
    return QuadraticTrajectory(
        method="gradient_descent",
        relative_objective=relative,
        gradient_norms=gradient_norms,
        step_sizes=torch.full((steps,), eta, dtype=operator.dtype, device=operator.device),
        final_vector=vector,
    )


def run_chebyshev(
    operator: SymmetricLinearOperator,
    initial: torch.Tensor,
    *,
    steps: int,
    eigen_min: float | None = None,
    eigen_max: float | None = None,
) -> QuadraticTrajectory:
    if steps < 0:
        raise ValueError("steps must be nonnegative")
    operator._validate_vector(initial)
    minimum, maximum = _resolve_bounds(operator, eigen_min, eigen_max)
    if math.isclose(minimum, maximum, rel_tol=1e-15, abs_tol=1e-15):
        vectors = [initial.clone()]
        zero = torch.zeros_like(initial)
        vectors.extend(zero.clone() for _ in range(steps))
        relative, gradient_norms = _relative_curve(operator, vectors)
        return QuadraticTrajectory(
            method="chebyshev",
            relative_objective=relative,
            gradient_norms=gradient_norms,
            step_sizes=torch.empty(0, dtype=operator.dtype, device=operator.device),
            final_vector=vectors[-1],
        )

    center = 0.5 * (maximum + minimum)
    radius = 0.5 * (maximum - minimum)

    def transformed(vector: torch.Tensor) -> torch.Tensor:
        return (center * vector - operator.matvec(vector)) / radius

    numerator_previous = initial.clone()  # T_0(Z) e_0
    denominator_previous = 1.0
    vectors = [initial.clone()]
    if steps >= 1:
        numerator_current = transformed(initial)
        denominator_current = center / radius
        vectors.append(numerator_current / denominator_current)
    else:
        numerator_current = numerator_previous
        denominator_current = denominator_previous

    for _degree in range(2, steps + 1):
        numerator_next = 2.0 * transformed(numerator_current) - numerator_previous
        denominator_next = 2.0 * (center / radius) * denominator_current - denominator_previous
        vectors.append(numerator_next / denominator_next)
        numerator_previous, numerator_current = numerator_current, numerator_next
        denominator_previous, denominator_current = denominator_current, denominator_next

    relative, gradient_norms = _relative_curve(operator, vectors)
    return QuadraticTrajectory(
        method="chebyshev",
        relative_objective=relative,
        gradient_norms=gradient_norms,
        step_sizes=torch.empty(0, dtype=operator.dtype, device=operator.device),
        final_vector=vectors[-1],
    )


def run_conjugate_gradient(
    operator: SymmetricLinearOperator,
    initial: torch.Tensor,
    *,
    steps: int,
    tolerance: float = 1e-12,
) -> QuadraticTrajectory:
    if steps < 0:
        raise ValueError("steps must be nonnegative")
    operator._validate_vector(initial)
    vector = initial.clone()
    residual = -operator.matvec(vector)
    direction = residual.clone()
    residual_square = torch.dot(residual, residual)
    vectors = [vector.clone()]
    step_sizes: list[torch.Tensor] = []

    for _ in range(steps):
        if torch.sqrt(residual_square) <= tolerance:
            break
        image = operator.matvec(direction)
        denominator = torch.dot(direction, image)
        if denominator <= 0:
            raise ValueError("Conjugate gradient requires a positive-definite operator")
        alpha = residual_square / denominator
        vector = vector + alpha * direction
        residual = residual - alpha * image
        new_residual_square = torch.dot(residual, residual)
        vectors.append(vector.clone())
        step_sizes.append(alpha)
        if torch.sqrt(new_residual_square) <= tolerance:
            residual_square = new_residual_square
            break
        beta = new_residual_square / residual_square
        direction = residual + beta * direction
        residual_square = new_residual_square

    # Keep trajectory lengths comparable by repeating the converged point.
    while len(vectors) < steps + 1:
        vectors.append(vector.clone())
        step_sizes.append(torch.zeros((), dtype=operator.dtype, device=operator.device))

    relative, gradient_norms = _relative_curve(operator, vectors)
    return QuadraticTrajectory(
        method="conjugate_gradient",
        relative_objective=relative,
        gradient_norms=gradient_norms,
        step_sizes=(
            torch.stack(step_sizes)
            if step_sizes
            else torch.empty(0, dtype=operator.dtype, device=operator.device)
        ),
        final_vector=vector,
    )


def run_frozen_block_continuation(
    functional: FunctionalBlockModel,
    batch: Batch,
    preconditioner: FrozenPreconditioner,
    *,
    steps: int,
    step_size: float,
) -> BlockContinuationResult:
    if steps < 0:
        raise ValueError("steps must be nonnegative")
    if step_size <= 0:
        raise ValueError("step_size must be positive")
    if preconditioner.dimension != functional.block.numel:
        raise ValueError("Preconditioner and block dimensions differ")

    parameter = functional.block_parameter.clone()
    gradient_function = torch.func.grad(lambda value: functional.mean_loss(value, batch))
    losses: list[float] = []
    gradient_norms: list[float] = []

    for iteration in range(steps + 1):
        loss = functional.mean_loss(parameter, batch)
        gradient = gradient_function(parameter)
        losses.append(float(loss.detach().item()))
        gradient_norms.append(float(torch.linalg.vector_norm(gradient).detach().item()))
        if iteration == steps:
            break
        direction = preconditioner.apply(gradient.reshape(-1)).reshape(functional.block.shape)
        parameter = (parameter - step_size * direction).detach()

    return BlockContinuationResult(
        losses=losses,
        gradient_norms=gradient_norms,
        final_parameter=parameter,
        step_size=float(step_size),
    )
