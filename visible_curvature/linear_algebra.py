from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable, Sequence

import torch

Tensor = torch.Tensor
MatVec = Callable[[Tensor], Tensor]


@dataclass
class LanczosResult:
    eigvals: Tensor
    alpha: Tensor
    beta: Tensor
    ritz_residuals: Tensor
    steps: int
    min_ritz: float
    max_ritz: float
    min_residual: float
    max_residual: float
    terminal_beta: float


def _normalize(v: Tensor, eps: float = 1e-30) -> Tensor:
    n = torch.linalg.vector_norm(v)
    if float(n) <= eps:
        raise RuntimeError("Lanczos start vector has near-zero norm")
    return v / n


def make_lanczos_starts(
    dim: int,
    starts: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    seed: int = 0,
) -> list[Tensor]:
    """Create deterministic normalized start vectors for paired operator comparisons."""
    if int(starts) <= 0:
        raise ValueError("starts must be positive")
    vectors: list[Tensor] = []
    for i in range(int(starts)):
        gen = torch.Generator(device=device).manual_seed(int(seed) + 104729 * i)
        vectors.append(_normalize(torch.randn(int(dim), generator=gen, device=device, dtype=dtype)))
    return vectors


def lanczos(
    matvec: MatVec,
    dim: int,
    steps: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    seed: int = 0,
    reorthogonalize: bool = True,
    start: Tensor | None = None,
) -> LanczosResult:
    """Symmetric Lanczos with optional full reorthogonalization.

    The returned Ritz residual for an eigenpair ``(theta, y)`` of the
    tridiagonal projection is ``beta_terminal * |y[-1]|``.  This diagnostic
    makes estimator convergence auditable and is exact in exact arithmetic.
    ``matvec`` accepts and returns flattened tensors of length ``dim``.
    """
    m = min(int(steps), int(dim))
    if m <= 0:
        raise ValueError("steps and dim must be positive")
    if start is None:
        gen = torch.Generator(device=device).manual_seed(int(seed))
        q = torch.randn(dim, generator=gen, device=device, dtype=dtype)
    else:
        q = start.to(device=device, dtype=dtype).flatten()
        if q.numel() != dim:
            raise ValueError(f"Lanczos start has {q.numel()} elements, expected {dim}")
    q = _normalize(q)
    q_prev = torch.zeros_like(q)
    beta_prev = torch.zeros((), device=device, dtype=dtype)
    basis: list[Tensor] = []
    alphas: list[Tensor] = []
    off_diagonal: list[Tensor] = []
    terminal_beta = torch.zeros((), device=device, dtype=dtype)

    for j in range(m):
        z = matvec(q).flatten().to(device=device, dtype=dtype)
        if z.numel() != dim:
            raise ValueError(f"matvec returned {z.numel()} elements, expected {dim}")
        if j > 0:
            z = z - beta_prev * q_prev
        alpha = torch.dot(q, z)
        z = z - alpha * q
        if reorthogonalize:
            # Two-pass modified Gram-Schmidt materially improves the lower tail
            # for ill-conditioned operators while keeping the implementation
            # transparent for audit.
            for _ in range(2):
                for qi in basis:
                    z = z - torch.dot(qi, z) * qi
                z = z - torch.dot(q, z) * q
        beta = torch.linalg.vector_norm(z)
        basis.append(q)
        alphas.append(alpha)
        terminal_beta = beta
        if j == m - 1 or float(beta) < 1e-12:
            break
        off_diagonal.append(beta)
        q_prev = q
        q = z / beta
        beta_prev = beta

    alpha_t = torch.stack(alphas)
    beta_t = torch.stack(off_diagonal) if off_diagonal else torch.empty(0, device=device, dtype=dtype)
    k = int(alpha_t.numel())
    T = torch.diag(alpha_t)
    if k > 1:
        b = beta_t[: k - 1]
        T = T + torch.diag(b, 1) + torch.diag(b, -1)
    eigvals, eigvecs = torch.linalg.eigh(T)
    # If the Krylov space spans the complete invariant subspace, terminal_beta
    # is numerically zero and all residuals vanish.
    residuals = terminal_beta.abs() * eigvecs[-1, :].abs()
    return LanczosResult(
        eigvals=eigvals,
        alpha=alpha_t,
        beta=beta_t,
        ritz_residuals=residuals,
        steps=k,
        min_ritz=float(eigvals[0].detach().cpu()),
        max_ritz=float(eigvals[-1].detach().cpu()),
        min_residual=float(residuals[0].detach().cpu()),
        max_residual=float(residuals[-1].detach().cpu()),
        terminal_beta=float(terminal_beta.detach().cpu()),
    )


def multi_start_lanczos(
    matvec: MatVec,
    dim: int,
    steps: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    starts: int | Sequence[Tensor] = 2,
    seed: int = 0,
) -> dict:
    """Estimate spectral endpoints from deterministic or supplied starts.

    Passing the same sequence of start vectors to several operators implements
    a common-random-number comparison and removes a preventable source of noise
    from optimizer-order differences.
    """
    if isinstance(starts, int):
        start_vectors = make_lanczos_starts(dim, starts, device, dtype, seed)
    else:
        start_vectors = [v.to(device=device, dtype=dtype).flatten() for v in starts]
        if not start_vectors:
            raise ValueError("At least one Lanczos start vector is required")
    results = [
        lanczos(matvec, dim, steps, device, dtype, start=v)
        for v in start_vectors
    ]
    min_index = min(range(len(results)), key=lambda i: results[i].min_ritz)
    max_index = max(range(len(results)), key=lambda i: results[i].max_ritz)
    min_run = results[min_index]
    max_run = results[max_index]
    all_ritz = torch.cat([r.eigvals.detach().cpu() for r in results])
    runs = [
        {
            "run": i,
            "steps": r.steps,
            "min_ritz": r.min_ritz,
            "max_ritz": r.max_ritz,
            "min_residual": r.min_residual,
            "max_residual": r.max_residual,
            "terminal_beta": r.terminal_beta,
        }
        for i, r in enumerate(results)
    ]
    return {
        "min_ritz": min_run.min_ritz,
        "max_ritz": max_run.max_ritz,
        "min_ritz_residual": min_run.min_residual,
        "max_ritz_residual": max_run.max_residual,
        "ritz_values": all_ritz,
        "steps": max(r.steps for r in results),
        "starts": len(results),
        "runs": runs,
        "start_vectors": start_vectors,
    }


def condition_from_spectrum(
    min_eig: float,
    max_eig: float,
    rel_floor: float = 1e-8,
    abs_floor: float = 0.0,
    truncation_tau: float | None = None,
) -> float:
    """Return a scale-invariant condition number or a truncated condition.

    ``rel_floor`` is only a reliability test for the ordinary condition number.
    With ``truncation_tau`` the reported quantity is

    ``max_eig / max(min_eig, truncation_tau * max_eig)``.

    The latter remains finite for unresolved or slightly negative lower Ritz
    values and must be labelled as a truncated diagnostic, not an exact
    condition number.
    """
    if not math.isfinite(min_eig) or not math.isfinite(max_eig) or max_eig <= 0:
        return float("inf")
    scale = abs(float(max_eig))
    if truncation_tau is not None:
        tau = float(truncation_tau)
        if not 0.0 < tau <= 1.0:
            raise ValueError("truncation_tau must lie in (0, 1]")
        denominator = max(float(min_eig), tau * scale)
        return float(max_eig / denominator)
    floor = max(float(abs_floor), float(rel_floor) * scale)
    if min_eig <= floor:
        return float("inf")
    return float(max_eig / min_eig)


def truncated_condition_sweep(min_eig: float, max_eig: float, taus: Sequence[float]) -> dict[float, float]:
    return {
        float(tau): condition_from_spectrum(min_eig, max_eig, truncation_tau=float(tau))
        for tau in taus
    }


def chebyshev_envelope(K: float, T: int) -> float:
    if K <= 1.0:
        return 0.0 if T >= 1 else 1.0
    gamma = math.log((math.sqrt(K) + 1.0) / (math.sqrt(K) - 1.0))
    x = T * gamma
    if x > 40:
        return 4.0 * math.exp(-2.0 * x)
    c = math.cosh(x)
    return 1.0 / (c * c)


def chebyshev_hitting_time(K: float, eps: float) -> int:
    if eps <= 0 or eps >= 1:
        raise ValueError("eps must lie in (0,1)")
    if K <= 1.0:
        return 1
    gamma = math.log((math.sqrt(K) + 1.0) / (math.sqrt(K) - 1.0))
    return int(math.ceil(math.acosh(eps ** -0.5) / gamma))


def constant_step_hitting_time(K: float, eps: float) -> int:
    """Endpoint hitting time for the minimax constant step (paper Lemma 21)."""
    if eps <= 0 or eps >= 1:
        raise ValueError("eps must lie in (0,1)")
    if K <= 1.0:
        return 1
    if not math.isfinite(K):
        return math.inf  # type: ignore[return-value]
    q = (K - 1.0) / (K + 1.0)
    return 1 + int(math.ceil(math.log(eps) / (2.0 * math.log(q))))


def conjugate_gradient(
    matvec: MatVec,
    b: Tensor,
    x0: Tensor | None = None,
    tol: float = 1e-6,
    max_iter: int = 100,
    objective_callback: Callable[[Tensor], float] | None = None,
) -> tuple[Tensor, dict]:
    x = torch.zeros_like(b) if x0 is None else x0.clone()
    r = b - matvec(x)
    p = r.clone()
    rs_old = torch.dot(r.flatten(), r.flatten())
    bnorm = max(float(torch.linalg.vector_norm(b)), 1e-30)
    hist = [math.sqrt(max(float(rs_old), 0.0)) / bnorm]
    objective_history = [objective_callback(x) if objective_callback else float("nan")]
    converged = hist[-1] <= tol
    if converged:
        return x, {"iterations": 0, "relative_residual": hist[-1], "history": hist, "objective_history": objective_history, "converged": True}
    for i in range(int(max_iter)):
        Ap = matvec(p)
        denom = torch.dot(p.flatten(), Ap.flatten())
        if not math.isfinite(float(denom)) or float(denom) <= 1e-30:
            break
        alpha = rs_old / denom
        x = x + alpha * p
        r = r - alpha * Ap
        rs_new = torch.dot(r.flatten(), r.flatten())
        rel = math.sqrt(max(float(rs_new), 0.0)) / bnorm
        hist.append(rel)
        objective_history.append(objective_callback(x) if objective_callback else float("nan"))
        if rel <= tol:
            return x, {"iterations": i + 1, "relative_residual": rel, "history": hist, "objective_history": objective_history, "converged": True}
        beta = rs_new / rs_old
        p = r + beta * p
        rs_old = rs_new
    return x, {"iterations": len(hist) - 1, "relative_residual": hist[-1], "history": hist, "objective_history": objective_history, "converged": False}


def symmetric_eigendecomp_with_metadata(
    A: Tensor,
    max_full_dim: int = 4096,
    approx_k: int = 64,
) -> tuple[Tensor, Tensor, dict[str, float | int | str]]:
    """Eigenpairs plus scope and residual metadata for full or tail decompositions.

    The tail path deliberately uses the Frobenius norm as a cheap scale proxy.
    Computing ``ord=2`` here would invoke another expensive spectral routine and
    defeat the purpose of using LOBPCG for large factors.
    """
    A = 0.5 * (A + A.T)
    n = int(A.shape[0])
    full = n <= int(max_full_dim) or int(approx_k) * 2 >= n
    matrix_scale = max(
        float(torch.linalg.vector_norm(A).detach().cpu()),
        torch.finfo(A.dtype).tiny,
    )
    if full:
        vals, vecs = torch.linalg.eigh(A)
        scope = "full"
    else:
        k = min(max(1, int(approx_k) // 2), max(1, n // 4))
        ridge = 1e-8 * matrix_scale
        A2 = A + ridge * torch.eye(n, device=A.device, dtype=A.dtype)
        vals_lo, vecs_lo = torch.lobpcg(A2, k=k, largest=False, niter=100)
        vals_hi, vecs_hi = torch.lobpcg(A2, k=k, largest=True, niter=100)
        vals = torch.cat([vals_lo - ridge, vals_hi - ridge])
        vecs = torch.cat([vecs_lo, vecs_hi], dim=1)
        order = torch.argsort(vals)
        vals, vecs = vals[order], vecs[:, order]
        scope = "spectral_tails"
    if vals.numel():
        residuals = torch.linalg.vector_norm(A @ vecs - vecs * vals.unsqueeze(0), dim=0) / matrix_scale
        max_residual = float(residuals.max().detach().cpu())
    else:
        max_residual = float("nan")
    metadata: dict[str, float | int | str] = {
        "scope": scope,
        "dimension": n,
        "modes_returned": int(vals.numel()),
        "mode_fraction": float(vals.numel() / max(n, 1)),
        "max_relative_residual": max_residual,
        "scale_proxy": "frobenius_norm",
    }
    return vals, vecs, metadata


def symmetric_eigendecomp(
    A: Tensor,
    max_full_dim: int = 4096,
    approx_k: int = 64,
) -> tuple[Tensor, Tensor]:
    """Eigenpairs ordered ascending, using full eigh or LOBPCG spectral tails."""
    vals, vecs, _ = symmetric_eigendecomp_with_metadata(A, max_full_dim=max_full_dim, approx_k=approx_k)
    return vals, vecs


def symmetric_eigen_powers_with_metadata(
    A: Tensor,
    powers: Sequence[float],
    damping: float = 0.0,
    eig_floor: float = 0.0,
    relative_eig_floor: float = 64.0,
) -> tuple[dict[float, Tensor], dict[str, float]]:
    """Compute spectral powers in float64 and report root-floor diagnostics.

    ``relative_eig_floor`` is expressed in multiples of float64 machine
    epsilon times the observed factor scale.  The resulting matrices are cast
    back to the input dtype, while the decomposition itself remains in
    float64 for stable inverse roots.
    """
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be a square matrix")
    if float(damping) < 0.0:
        raise ValueError("damping must be nonnegative")
    if float(eig_floor) < 0.0:
        raise ValueError("eig_floor must be nonnegative")
    if float(relative_eig_floor) < 0.0:
        raise ValueError("relative_eig_floor must be nonnegative")

    original_dtype = A.dtype
    work = (0.5 * (A + A.T)).to(dtype=torch.float64)
    n = int(work.shape[0])
    if damping:
        work = work + float(damping) * torch.eye(n, device=work.device, dtype=work.dtype)
    vals, vecs = torch.linalg.eigh(work)
    raw_min = float(vals.min().detach().cpu()) if vals.numel() else float("nan")
    raw_max = float(vals.max().detach().cpu()) if vals.numel() else float("nan")
    scale = max(abs(raw_min) if math.isfinite(raw_min) else 0.0, abs(raw_max) if math.isfinite(raw_max) else 0.0, torch.finfo(torch.float64).tiny)
    adaptive = float(relative_eig_floor) * torch.finfo(torch.float64).eps * scale
    effective_floor = max(float(eig_floor), adaptive, torch.finfo(torch.float64).tiny)
    floor_mask = vals < effective_floor
    clamped = vals.clamp_min(effective_floor)
    outputs = {
        float(power): ((vecs * clamped.pow(float(power)).unsqueeze(0)) @ vecs.T).to(dtype=original_dtype)
        for power in powers
    }
    metadata = {
        "effective_eig_floor": float(effective_floor),
        "floored_fraction": float(floor_mask.double().mean().detach().cpu()) if vals.numel() else 0.0,
        "raw_min_eigenvalue": raw_min,
        "raw_max_eigenvalue": raw_max,
    }
    return outputs, metadata


def symmetric_eigen_powers(
    A: Tensor,
    powers: Sequence[float],
    damping: float = 0.0,
    eig_floor: float = 0.0,
    relative_eig_floor: float = 64.0,
) -> dict[float, Tensor]:
    """Backward-compatible wrapper around the metadata-producing routine."""
    values, _ = symmetric_eigen_powers_with_metadata(
        A,
        powers,
        damping=damping,
        eig_floor=eig_floor,
        relative_eig_floor=relative_eig_floor,
    )
    return values


def matrix_power_symmetric(
    A: Tensor,
    power: float,
    damping: float = 0.0,
    eig_floor: float = 0.0,
    relative_eig_floor: float = 64.0,
) -> Tensor:
    return symmetric_eigen_powers(
        A,
        [power],
        damping=damping,
        eig_floor=eig_floor,
        relative_eig_floor=relative_eig_floor,
    )[float(power)]
