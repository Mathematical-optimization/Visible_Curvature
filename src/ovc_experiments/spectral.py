from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from .operators import SymmetricLinearOperator, materialize


@dataclass
class LanczosResult:
    alphas: torch.Tensor
    betas: torch.Tensor
    basis: torch.Tensor
    tridiagonal: torch.Tensor
    ritz_values: torch.Tensor
    ritz_vectors: torch.Tensor
    residual_norms: torch.Tensor
    tail_beta: float
    iterations: int
    breakdown: bool


@dataclass
class ConditionEstimate:
    min_eigenvalue: float | None
    max_eigenvalue: float | None
    condition_number: float | None
    min_residual: float | None
    max_residual: float | None
    censored: bool
    censor_reason: str | None
    method: str
    positive_threshold: float
    negative_eigenvalues: int = 0
    null_eigenvalues: int = 0
    starts: int = 1

    def to_dict(self) -> dict[str, object]:
        return {
            "min_eigenvalue": self.min_eigenvalue,
            "max_eigenvalue": self.max_eigenvalue,
            "condition_number": self.condition_number,
            "min_residual": self.min_residual,
            "max_residual": self.max_residual,
            "censored": self.censored,
            "censor_reason": self.censor_reason,
            "method": self.method,
            "positive_threshold": self.positive_threshold,
            "negative_eigenvalues": self.negative_eigenvalues,
            "null_eigenvalues": self.null_eigenvalues,
            "starts": self.starts,
        }


@dataclass
class SLQResult:
    nodes: torch.Tensor
    weights: torch.Tensor
    probes: int
    steps: int

    def quantile(self, probability: float) -> float:
        if not 0.0 <= probability <= 1.0:
            raise ValueError("probability must be in [0, 1]")
        order = torch.argsort(self.nodes)
        nodes = self.nodes[order]
        weights = self.weights[order]
        cumulative = torch.cumsum(weights, dim=0)
        target = torch.tensor(probability, dtype=cumulative.dtype, device=cumulative.device)
        index = int(torch.searchsorted(cumulative, target, right=False).clamp(max=len(nodes) - 1))
        return float(nodes[index].item())


def _random_unit_vector(
    dimension: int,
    *,
    dtype: torch.dtype,
    device: torch.device,
    seed: int,
    rademacher: bool = False,
) -> torch.Tensor:
    generator_device = device.type if device.type in {"cpu", "cuda"} else "cpu"
    generator = torch.Generator(device=generator_device)
    generator.manual_seed(int(seed))
    if rademacher:
        raw = torch.randint(
            0,
            2,
            (dimension,),
            generator=generator,
            device=device,
            dtype=torch.int64,
        )
        vector = raw.to(dtype=dtype).mul_(2).sub_(1)
    else:
        vector = torch.randn(dimension, generator=generator, dtype=dtype, device=device)
    norm = torch.linalg.vector_norm(vector)
    if norm == 0:
        raise RuntimeError("Random start produced a zero vector")
    return vector / norm


def lanczos(
    operator: SymmetricLinearOperator,
    *,
    steps: int,
    initial_vector: torch.Tensor | None = None,
    seed: int = 0,
    reorthogonalize: bool = True,
    breakdown_tolerance: float = 1e-14,
) -> LanczosResult:
    if steps < 1:
        raise ValueError("steps must be positive")
    maximum_steps = min(int(steps), operator.dimension)
    q = (
        _random_unit_vector(
            operator.dimension,
            dtype=operator.dtype,
            device=operator.device,
            seed=seed,
        )
        if initial_vector is None
        else initial_vector.to(dtype=operator.dtype, device=operator.device)
    )
    operator._validate_vector(q)
    q_norm = torch.linalg.vector_norm(q)
    if q_norm <= breakdown_tolerance:
        raise ValueError("initial_vector must be nonzero")
    q = q / q_norm

    basis: list[torch.Tensor] = []
    alphas: list[torch.Tensor] = []
    computed_betas: list[torch.Tensor] = []
    q_previous = torch.zeros_like(q)
    beta_previous = torch.zeros((), dtype=operator.dtype, device=operator.device)
    breakdown = False

    for _ in range(maximum_steps):
        basis.append(q)
        z = operator.matvec(q) - beta_previous * q_previous
        alpha = torch.dot(q, z)
        z = z - alpha * q

        if reorthogonalize:
            q_matrix = torch.stack(basis, dim=1)
            # Two passes suppress loss of orthogonality for clustered spectra.
            for _pass in range(2):
                z = z - q_matrix @ (q_matrix.T @ z)

        beta = torch.linalg.vector_norm(z)
        alphas.append(alpha)
        computed_betas.append(beta)

        if beta <= breakdown_tolerance:
            breakdown = True
            break
        if len(alphas) >= maximum_steps:
            break
        q_previous, q = q, z / beta
        beta_previous = beta

    m = len(alphas)
    alpha_tensor = torch.stack(alphas)
    off_diagonal = (
        torch.stack(computed_betas[: m - 1])
        if m > 1
        else torch.empty(0, dtype=operator.dtype, device=operator.device)
    )
    tail_beta_tensor = computed_betas[m - 1]
    tridiagonal = torch.diag(alpha_tensor)
    if m > 1:
        tridiagonal = tridiagonal + torch.diag(off_diagonal, diagonal=1)
        tridiagonal = tridiagonal + torch.diag(off_diagonal, diagonal=-1)

    ritz_values, tridiagonal_vectors = torch.linalg.eigh(tridiagonal)
    basis_matrix = torch.stack(basis, dim=1)
    ritz_vectors = basis_matrix @ tridiagonal_vectors
    residuals = torch.abs(tail_beta_tensor * tridiagonal_vectors[-1, :])

    return LanczosResult(
        alphas=alpha_tensor,
        betas=off_diagonal,
        basis=basis_matrix,
        tridiagonal=tridiagonal,
        ritz_values=ritz_values,
        ritz_vectors=ritz_vectors,
        residual_norms=residuals,
        tail_beta=float(tail_beta_tensor.item()),
        iterations=m,
        breakdown=breakdown,
    )


def estimate_condition(
    operator: SymmetricLinearOperator,
    *,
    exact_max_dim: int = 512,
    lanczos_steps: int = 64,
    starts: int = 2,
    seed: int = 0,
    positive_threshold: float = 1e-10,
    residual_tolerance: float = 1e-5,
) -> ConditionEstimate:
    if operator.dimension <= exact_max_dim:
        matrix = materialize(operator)
        eigenvalues = torch.linalg.eigvalsh(matrix)
        positive = eigenvalues[eigenvalues > positive_threshold]
        negative_count = int((eigenvalues < -positive_threshold).sum().item())
        null_count = int((torch.abs(eigenvalues) <= positive_threshold).sum().item())
        if positive.numel() == 0:
            return ConditionEstimate(
                min_eigenvalue=None,
                max_eigenvalue=None,
                condition_number=None,
                min_residual=None,
                max_residual=None,
                censored=True,
                censor_reason="no_positive_eigenvalue",
                method="exact",
                positive_threshold=positive_threshold,
                negative_eigenvalues=negative_count,
                null_eigenvalues=null_count,
            )
        minimum = float(positive.min().item())
        maximum = float(positive.max().item())
        return ConditionEstimate(
            min_eigenvalue=minimum,
            max_eigenvalue=maximum,
            condition_number=maximum / minimum,
            min_residual=0.0,
            max_residual=0.0,
            censored=False,
            censor_reason=None,
            method="exact",
            positive_threshold=positive_threshold,
            negative_eigenvalues=negative_count,
            null_eigenvalues=null_count,
        )

    candidate_minima: list[tuple[float, float]] = []
    candidate_maxima: list[tuple[float, float]] = []
    negative_count = 0
    null_count = 0
    used_steps = min(lanczos_steps, operator.dimension)
    for start in range(max(1, starts)):
        result = lanczos(operator, steps=used_steps, seed=seed + 104729 * start)
        for value_tensor, residual_tensor in zip(result.ritz_values, result.residual_norms):
            value = float(value_tensor.item())
            residual = float(residual_tensor.item())
            converged = residual <= residual_tolerance * max(1.0, abs(value))
            if value > positive_threshold and converged:
                candidate_minima.append((value, residual))
                candidate_maxima.append((value, residual))
            elif value < -positive_threshold:
                negative_count += 1
            else:
                null_count += 1

    if not candidate_minima:
        return ConditionEstimate(
            min_eigenvalue=None,
            max_eigenvalue=None,
            condition_number=None,
            min_residual=None,
            max_residual=None,
            censored=True,
            censor_reason="smallest_positive_ritz_unresolved",
            method="lanczos",
            positive_threshold=positive_threshold,
            negative_eigenvalues=negative_count,
            null_eigenvalues=null_count,
            starts=max(1, starts),
        )

    minimum_pair = min(candidate_minima, key=lambda item: item[0])
    maximum_pair = max(candidate_maxima, key=lambda item: item[0])
    minimum, minimum_residual = minimum_pair
    maximum, maximum_residual = maximum_pair
    if not math.isfinite(minimum) or minimum <= positive_threshold:
        return ConditionEstimate(
            min_eigenvalue=None,
            max_eigenvalue=maximum,
            condition_number=None,
            min_residual=minimum_residual,
            max_residual=maximum_residual,
            censored=True,
            censor_reason="smallest_positive_ritz_unresolved",
            method="lanczos",
            positive_threshold=positive_threshold,
            negative_eigenvalues=negative_count,
            null_eigenvalues=null_count,
            starts=max(1, starts),
        )
    return ConditionEstimate(
        min_eigenvalue=minimum,
        max_eigenvalue=maximum,
        condition_number=maximum / minimum,
        min_residual=minimum_residual,
        max_residual=maximum_residual,
        censored=False,
        censor_reason=None,
        method="lanczos",
        positive_threshold=positive_threshold,
        negative_eigenvalues=negative_count,
        null_eigenvalues=null_count,
        starts=max(1, starts),
    )


def slq_spectrum(
    operator: SymmetricLinearOperator,
    *,
    probes: int = 8,
    steps: int = 32,
    seed: int = 0,
) -> SLQResult:
    if probes < 1:
        raise ValueError("probes must be positive")
    all_nodes: list[torch.Tensor] = []
    all_weights: list[torch.Tensor] = []
    for probe in range(probes):
        initial = _random_unit_vector(
            operator.dimension,
            dtype=operator.dtype,
            device=operator.device,
            seed=seed + 65537 * probe,
            rademacher=True,
        )
        result = lanczos(
            operator,
            steps=min(steps, operator.dimension),
            initial_vector=initial,
            seed=seed + 65537 * probe,
        )
        quadrature_weights = result.tridiagonal.new_tensor(1.0 / probes) * (
            torch.linalg.eigh(result.tridiagonal).eigenvectors[0, :] ** 2
        )
        all_nodes.append(result.ritz_values)
        all_weights.append(quadrature_weights)
    nodes = torch.cat(all_nodes)
    weights = torch.cat(all_weights)
    weights = weights / weights.sum()
    return SLQResult(nodes=nodes, weights=weights, probes=probes, steps=min(steps, operator.dimension))

# The hardened estimator is now the single implementation for both small and
# large operators.  Keeping one path prevents exact small-dimensional runs
# from silently using positive-active semantics while large runs use a
# different policy.
from .hardened_spectral import estimate_condition as _hardened_estimate_condition
from .hardened_spectral import set_condition_diagnostics_path


def estimate_condition(operator, *args, **kwargs):
    if args:
        raise TypeError("estimate_condition accepts keyword arguments after operator")
    kwargs.setdefault("subspace_policy", "strict_spd")
    return _hardened_estimate_condition(
        operator,
        result_type=ConditionEstimate,
        **kwargs,
    )
