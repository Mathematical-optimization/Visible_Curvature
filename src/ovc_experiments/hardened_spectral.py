from __future__ import annotations

import inspect
import json
import math
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

import torch

from .io import append_jsonl_strict

Tensor = torch.Tensor
_DIAGNOSTICS_WRITER: Optional["ConditionDiagnosticsWriter"] = None


class ConditionDiagnosticsWriter:
    """Append-only strict-JSON writer for condition-estimation diagnostics."""

    def __init__(self, path: str | Path, *, reset: bool = False) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if reset:
            self.path.write_text("", encoding="utf-8")

    def write(self, record: dict[str, Any]) -> None:
        append_jsonl_strict(record, self.path)


def set_condition_diagnostics_path(
    path: str | Path | None,
    *,
    truncate: bool = False,
) -> None:
    """Compatibility setter; new callers should pass a writer explicitly."""

    global _DIAGNOSTICS_WRITER
    _DIAGNOSTICS_WRITER = (
        None if path is None else ConditionDiagnosticsWriter(path, reset=truncate)
    )


def _dimension(operator: Any) -> int:
    for name in ('dimension', 'dim'):
        value = getattr(operator, name, None)
        if value is not None:
            return int(value() if callable(value) else value)
    shape = getattr(operator, 'shape', None)
    if shape is not None:
        return int(shape[0])
    matrix = getattr(operator, 'matrix', None)
    if matrix is not None:
        return int(matrix.shape[0])
    raise TypeError('operator must expose dimension, dim, shape, or matrix')


def _dtype(operator: Any) -> torch.dtype:
    value = getattr(operator, 'dtype', None)
    if isinstance(value, torch.dtype):
        return value
    matrix = getattr(operator, 'matrix', None)
    if isinstance(matrix, Tensor):
        return matrix.dtype
    return torch.float64


def _device(operator: Any) -> torch.device:
    value = getattr(operator, 'device', None)
    if value is not None:
        return torch.device(value)
    matrix = getattr(operator, 'matrix', None)
    if isinstance(matrix, Tensor):
        return matrix.device
    return torch.device('cpu')


def _matvec(operator: Any, vector: Tensor) -> Tensor:
    if hasattr(operator, 'matvec'):
        out = operator.matvec(vector)
    elif callable(operator):
        out = operator(vector)
    else:
        out = operator @ vector
    return out.reshape(-1)


def _explicit_diagonal(operator: Any) -> Optional[Tensor]:
    diagonal = getattr(operator, 'diagonal', None)
    if isinstance(diagonal, Tensor):
        return diagonal.reshape(-1)
    matrix = getattr(operator, 'matrix', None)
    if isinstance(matrix, Tensor) and matrix.ndim == 2:
        diag = torch.diagonal(matrix)
        # This check is intentionally limited to moderate matrices.
        if matrix.numel() <= 16_000_000:
            off = matrix - torch.diag_embed(diag)
            scale = max(float(matrix.abs().max().item()), 1.0)
            if float(off.abs().max().item()) <= 100 * torch.finfo(matrix.dtype).eps * scale:
                return diag
    return None


def _materialize(operator: Any, n: int, dtype: torch.dtype, device: torch.device) -> Tensor:
    matrix = getattr(operator, 'matrix', None)
    if isinstance(matrix, Tensor):
        return ((matrix + matrix.mT) * 0.5).to(dtype=dtype, device=device)
    eye = torch.eye(n, dtype=dtype, device=device)
    columns = [_matvec(operator, eye[:, j]) for j in range(n)]
    dense = torch.stack(columns, dim=1)
    return (dense + dense.mT) * 0.5


def _active_threshold(
    maximum: float,
    n: int,
    dtype: torch.dtype,
    absolute: float,
    relative: float,
) -> float:
    eps = torch.finfo(dtype).eps if dtype.is_floating_point else torch.finfo(torch.float64).eps
    return max(float(absolute), float(relative) * max(maximum, 0.0), 64.0 * eps * n * max(maximum, torch.finfo(dtype).tiny))


def _lanczos(operator: Any, steps: int, start: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    n = start.numel()
    q = start / torch.linalg.vector_norm(start)
    q_prev = torch.zeros_like(q)
    beta_prev = torch.zeros((), dtype=q.dtype, device=q.device)
    basis: list[Tensor] = []
    alphas: list[Tensor] = []
    betas: list[Tensor] = []
    for j in range(min(steps, n)):
        z = _matvec(operator, q)
        if j:
            z = z - beta_prev * q_prev
        alpha = torch.dot(q, z)
        z = z - alpha * q
        # Full reorthogonalization is deliberate: endpoint certification is
        # more important than a few extra dot products in diagnostics.
        for old in basis:
            z = z - torch.dot(old, z) * old
        beta = torch.linalg.vector_norm(z)
        basis.append(q)
        alphas.append(alpha)
        if j == min(steps, n) - 1 or float(beta.item()) <= torch.finfo(q.dtype).eps:
            break
        betas.append(beta)
        q_prev, q = q, z / beta
        beta_prev = beta
    k = len(alphas)
    T = torch.diag(torch.stack(alphas))
    if k > 1:
        off = torch.stack(betas[: k - 1])
        T = T + torch.diag(off, diagonal=1) + torch.diag(off, diagonal=-1)
    theta, y = torch.linalg.eigh(T)
    Q = torch.stack(basis, dim=1)
    ritz = Q @ y
    residuals = torch.empty_like(theta)
    for j in range(theta.numel()):
        v = ritz[:, j]
        residuals[j] = torch.linalg.vector_norm(_matvec(operator, v) - theta[j] * v)
    return theta, residuals, ritz


def _certified(value: float, residual: float, scale: float, tolerance: float, dtype: torch.dtype, n: int) -> bool:
    eps_floor = 64.0 * torch.finfo(dtype).eps * n * max(abs(scale), torch.finfo(dtype).tiny)
    return math.isfinite(value) and math.isfinite(residual) and residual <= tolerance * max(abs(value), eps_floor)


def _cg_solve(
    operator: Any,
    b: Tensor,
    *,
    tolerance: float,
    max_iterations: int,
    diagonal: Optional[Tensor] = None,
) -> tuple[Tensor, float, int, bool]:
    x = torch.zeros_like(b)
    r = b.clone()
    if diagonal is not None:
        floor = max(float(diagonal.abs().max().item()) * 1e-15, torch.finfo(b.dtype).tiny)
        inv = diagonal.clamp_min(floor).reciprocal()
        z = inv * r
    else:
        z = r.clone()
    p = z.clone()
    rz = torch.dot(r, z)
    bnorm = max(float(torch.linalg.vector_norm(b).item()), torch.finfo(b.dtype).tiny)
    for iteration in range(1, max_iterations + 1):
        Ap = _matvec(operator, p)
        denom = torch.dot(p, Ap)
        if not torch.isfinite(denom) or float(denom.item()) <= 0:
            return x, float(torch.linalg.vector_norm(r).item()) / bnorm, iteration, False
        alpha = rz / denom
        x = x + alpha * p
        r = r - alpha * Ap
        rel = float(torch.linalg.vector_norm(r).item()) / bnorm
        if rel <= tolerance:
            return x, rel, iteration, True
        if diagonal is not None:
            z = inv * r
        else:
            z = r.clone()
        rz_new = torch.dot(r, z)
        beta = rz_new / rz
        p = z + beta * p
        rz = rz_new
    rel = float(torch.linalg.vector_norm(r).item()) / bnorm
    return x, rel, max_iterations, rel <= tolerance


def _inverse_iteration_minimum(
    operator: Any,
    *,
    n: int,
    dtype: torch.dtype,
    device: torch.device,
    starts: int,
    seed: int,
    outer_iterations: int,
    cg_tolerance: float,
    cg_max_iterations: int,
    endpoint_tolerance: float,
    scale: float,
) -> tuple[Optional[float], Optional[float], dict[str, Any]]:
    diagonal = _explicit_diagonal(operator)
    estimates: list[tuple[float, float]] = []
    cg_records: list[dict[str, Any]] = []
    generator = torch.Generator(device=device)
    generator.manual_seed(seed + 104729)
    for start_index in range(starts):
        x = torch.randn(n, dtype=dtype, device=device, generator=generator)
        x = x / torch.linalg.vector_norm(x)
        successful = True
        total_cg = 0
        last_rel = math.inf
        for _ in range(outer_iterations):
            y, rel, iters, ok = _cg_solve(
                operator,
                x,
                tolerance=cg_tolerance,
                max_iterations=cg_max_iterations,
                diagonal=diagonal,
            )
            total_cg += iters
            last_rel = rel
            norm_y = torch.linalg.vector_norm(y)
            if not ok or not torch.isfinite(norm_y) or float(norm_y.item()) == 0.0:
                successful = False
                break
            x = y / norm_y
        if successful:
            Ax = _matvec(operator, x)
            theta = float(torch.dot(x, Ax).item())
            residual = float(torch.linalg.vector_norm(Ax - theta * x).item())
            if theta > 0 and _certified(theta, residual, scale, endpoint_tolerance, dtype, n):
                estimates.append((theta, residual))
        cg_records.append({'start': start_index, 'success': successful, 'cg_iterations': total_cg, 'last_relative_residual': last_rel})
    if not estimates:
        return None, None, {'records': cg_records}
    estimates.sort(key=lambda item: item[0])
    best = estimates[0]
    return best[0], best[1], {'records': cg_records}


def _construct(result_type: Any, data: dict[str, Any]) -> Any:
    if result_type is None:
        from types import SimpleNamespace
        public = dict(data)
        # Historical direct callers cast the value to float even when the
        # estimate is censored. Preserve that API with +inf while diagnostics
        # and typed ConditionEstimate objects retain None and cannot create a
        # finite conditioning gain.
        if public.get('condition_number') is None:
            public['condition_number'] = math.inf
        return SimpleNamespace(**public)
    signature = inspect.signature(result_type)
    aliases = {
        'condition': data['condition_number'],
        'condition_number': data['condition_number'],
        'minimum_eigenvalue': data['minimum_eigenvalue'],
        'min_eigenvalue': data['minimum_eigenvalue'],
        'lambda_min': data['minimum_eigenvalue'],
        'maximum_eigenvalue': data['maximum_eigenvalue'],
        'max_eigenvalue': data['maximum_eigenvalue'],
        'lambda_max': data['maximum_eigenvalue'],
        'minimum_residual': data['minimum_residual'],
        'min_residual': data['minimum_residual'],
        'residual_min': data['minimum_residual'],
        'maximum_residual': data['maximum_residual'],
        'max_residual': data['maximum_residual'],
        'residual_max': data['maximum_residual'],
        'censored': data['censored'],
        'is_censored': data['censored'],
        'method': data['method'],
        'censor_reason': data['censor_reason'],
        'reason': data['censor_reason'],
        'positive_threshold': data['positive_threshold'],
        'threshold': data['positive_threshold'],
        'null_eigenvalues': data['null_eigenvalues'],
        'negative_eigenvalues': data['negative_eigenvalues'],
        'starts': data['starts'],
        'lanczos_steps': data['lanczos_steps'],
    }
    kwargs: dict[str, Any] = {}
    for name, parameter in signature.parameters.items():
        if name in aliases:
            kwargs[name] = aliases[name]
        elif parameter.default is inspect.Parameter.empty:
            # Required legacy fields are populated by type-shaped safe values.
            annotation = parameter.annotation
            if annotation is bool:
                kwargs[name] = False
            elif annotation is int:
                kwargs[name] = 0
            elif annotation is str:
                kwargs[name] = ''
            else:
                kwargs[name] = None
    try:
        return result_type(**kwargs)
    except Exception:
        # Dataclasses without a conventional constructor are rare, but keeping
        # a namespace is preferable to fabricating a finite condition number.
        from types import SimpleNamespace
        return SimpleNamespace(**data)


def _write_diagnostic(
    record: dict[str, Any],
    writer: ConditionDiagnosticsWriter | None = None,
) -> None:
    selected = writer or _DIAGNOSTICS_WRITER
    if selected is not None:
        selected.write(record)



def condition_diagnostics_jsonl_to_csv(source: str | Path, destination: str | Path) -> Path:
    import csv
    source_path = Path(source)
    destination_path = Path(destination)
    records = [json.loads(line) for line in source_path.read_text(encoding='utf-8').splitlines() if line.strip()]
    fields = sorted({key for record in records for key in record})
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    with destination_path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)
    return destination_path

def estimate_condition(operator: Any, *args: Any, result_type: Any = None, **kwargs: Any) -> Any:
    """Estimate a positive-active-spectrum condition number without false minima.

    A large-dimensional minimum is reported only when an endpoint-specific
    Ritz pair or inverse-iteration pair passes a scale-aware residual test and
    a budget-stability check. Otherwise the result is censored with an infinite
    condition number rather than a misleading finite value.
    """
    n = _dimension(operator)
    dtype = _dtype(operator)
    if dtype not in (torch.float32, torch.float64):
        dtype = torch.float64
    device = _device(operator)
    exact_max = int(kwargs.pop('exact_condition_max_dim', kwargs.pop('exact_max_dim', 512)))
    steps = int(kwargs.pop('lanczos_steps', kwargs.pop('steps', min(128, n))))
    starts = int(kwargs.pop('starts', kwargs.pop('num_starts', 3)))
    tolerance = float(kwargs.pop('residual_tolerance', kwargs.pop('tolerance', 1e-7)))
    absolute_threshold = float(kwargs.pop('positive_threshold', kwargs.pop('absolute_threshold', 0.0)))
    relative_threshold = float(kwargs.pop('relative_positive_threshold', kwargs.pop('relative_threshold', 1e-10)))
    seed = int(kwargs.pop('seed', 0))
    subspace_policy = str(kwargs.pop('subspace_policy', 'strict_spd'))
    if subspace_policy not in {'strict_spd', 'positive_active', 'pseudoinverse'}:
        raise ValueError(f'unknown subspace policy: {subspace_policy}')
    minimum_method = str(kwargs.pop('minimum_method', 'auto'))
    inverse_outer = int(kwargs.pop('inverse_outer_iterations', 10))
    cg_tolerance = float(kwargs.pop('cg_tolerance', min(1e-10, tolerance * 0.1)))
    cg_max = int(kwargs.pop('cg_max_iterations', min(max(8 * n, 500), 20000)))
    q01 = kwargs.pop('slq_q01', kwargs.pop('spectral_quantile_01', None))
    diagnostic_context = kwargs.pop('diagnostic_context', {}) or {}
    diagnostics_writer = kwargs.pop('diagnostics_writer', None)
    if diagnostics_writer is not None and not isinstance(
        diagnostics_writer, ConditionDiagnosticsWriter
    ):
        raise TypeError('diagnostics_writer must be a ConditionDiagnosticsWriter')

    minimum = math.nan
    maximum = math.nan
    min_residual = math.nan
    max_residual = math.nan
    negative_count = 0
    null_count = 0
    censored = False
    reason = ''
    method = 'exact'

    diagonal = _explicit_diagonal(operator)
    if diagonal is not None:
        eig = diagonal.to(dtype=dtype, device=device)
        method = 'diagonal_exact'
    elif n <= exact_max:
        eig = torch.linalg.eigvalsh(_materialize(operator, n, dtype, device))
        method = 'dense_exact'
    else:
        eig = None

    if eig is not None:
        maximum = float(eig.max().item())
        threshold = _active_threshold(maximum, n, eig.dtype, absolute_threshold, relative_threshold)
        negative_count = int((eig < -threshold).sum().item())
        active = eig[eig > threshold]
        null_count = int((eig.abs() <= threshold).sum().item())
        if active.numel() == 0 or maximum <= 0:
            censored = True
            reason = 'no_positive_eigenvalue'
        elif negative_count and subspace_policy == 'strict_spd':
            censored = True
            reason = 'negative_eigenvalues_in_strict_spd_mode'
        elif null_count and subspace_policy == 'strict_spd':
            censored = True
            reason = 'null_eigenvalues_in_strict_spd_mode'
        else:
            minimum = float(active.min().item())
            maximum = float(active.max().item())
            min_residual = 0.0
            max_residual = 0.0
    else:
        method = 'lanczos_endpoints'
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)
        max_pairs: list[tuple[float, float]] = []
        min_pairs_full: list[tuple[float, float]] = []
        min_pairs_half: list[tuple[float, float]] = []
        for start_index in range(starts):
            start = torch.randn(n, dtype=dtype, device=device, generator=generator)
            theta, residuals, _ = _lanczos(operator, steps, start)
            max_pairs.append((float(theta[-1].item()), float(residuals[-1].item())))
            min_pairs_full.append((float(theta[0].item()), float(residuals[0].item())))
            half_steps = max(8, steps // 2)
            theta_half, residuals_half, _ = _lanczos(operator, half_steps, start)
            min_pairs_half.append((float(theta_half[0].item()), float(residuals_half[0].item())))
        max_pairs.sort(key=lambda item: item[0], reverse=True)
        maximum, max_residual = max_pairs[0]
        threshold = _active_threshold(maximum, n, dtype, absolute_threshold, relative_threshold)
        max_ok = _certified(maximum, max_residual, maximum, tolerance, dtype, n)

        # A diagonal was already handled exactly. For a generic operator, a
        # smallest Ritz value is accepted only if it is both residual-certified
        # and stable under doubling the Krylov budget.
        valid_min: list[tuple[float, float]] = []
        for full, half in zip(min_pairs_full, min_pairs_half):
            value, residual = full
            stable = abs(value - half[0]) <= max(tolerance * max(abs(value), threshold), threshold)
            if value > threshold and stable and _certified(value, residual, maximum, tolerance, dtype, n):
                valid_min.append(full)
        inverse_meta: dict[str, Any] = {}
        if valid_min:
            valid_min.sort(key=lambda item: item[0])
            minimum, min_residual = valid_min[0]
        elif minimum_method in {'auto', 'inverse_iteration', 'shift_invert'}:
            minimum, min_residual, inverse_meta = _inverse_iteration_minimum(
                operator,
                n=n,
                dtype=dtype,
                device=device,
                starts=starts,
                seed=seed,
                outer_iterations=inverse_outer,
                cg_tolerance=cg_tolerance,
                cg_max_iterations=cg_max,
                endpoint_tolerance=tolerance,
                scale=maximum,
            )
            if minimum is not None:
                method = 'lanczos_max+inverse_iteration_min'
        min_ok = minimum is not None and math.isfinite(minimum) and minimum > threshold
        if not max_ok:
            censored = True
            reason = 'unresolved_maximum_eigenvalue'
        elif not min_ok:
            censored = True
            reason = 'unresolved_smallest_positive_eigenvalue'
            minimum = math.nan
            min_residual = math.nan

    if not censored and q01 is not None and math.isfinite(minimum):
        q01_value = float(q01)
        if q01_value > 0 and minimum > q01_value * (1.0 + 10.0 * tolerance):
            censored = True
            reason = 'minimum_inconsistent_with_slq_q01'

    condition = None if censored or not math.isfinite(minimum) or minimum <= 0 else maximum / minimum
    minimum_output = None if not math.isfinite(minimum) else float(minimum)
    maximum_output = None if not math.isfinite(maximum) else float(maximum)
    minimum_residual_output = None if not math.isfinite(min_residual) else float(min_residual)
    maximum_residual_output = None if not math.isfinite(max_residual) else float(max_residual)
    data = {
        'condition_number': condition,
        'minimum_eigenvalue': minimum_output,
        'maximum_eigenvalue': maximum_output,
        'minimum_residual': minimum_residual_output,
        'maximum_residual': maximum_residual_output,
        'censored': bool(censored),
        'censor_reason': reason,
        'method': method,
        'positive_threshold': float(threshold if 'threshold' in locals() else absolute_threshold),
        'null_eigenvalues': int(null_count),
        'negative_eigenvalues': int(negative_count),
        'starts': int(starts),
        'lanczos_steps': int(steps),
        **diagnostic_context,
    }
    _write_diagnostic(data, diagnostics_writer)
    return _construct(result_type, data)
