from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Iterable, Mapping, Optional

import torch

from .curvature_policy import validate_curvature_policy
from .hardened_spectral import (
    ConditionDiagnosticsWriter,
    condition_diagnostics_jsonl_to_csv,
    estimate_condition,
)
from .io import append_dataframe_row, append_jsonl_strict
from .safe_operators import FunctionOperator
from .spectral_power import symmetric_matrix_power
from .streaming_moments import MatrixMomentStatistics, accumulate_matrix_moments
from .paths import resolve_output_dir
from .spectrum_utils import FactorwiseDamping, factorwise_damping, positive_spectrum_summary

Tensor = torch.Tensor


@dataclass(frozen=True)
class HardenedBlockConfig:
    curvature_kind: str = 'ggn'
    primary_analysis: bool = True
    centered_moments: bool = True
    population_moments: bool = True
    adam_damping: float = 0.0
    shampoo_damping: float = 0.0
    shampoo_damping_ratio: float | None = None
    shampoo_exponent: float = 0.25
    subspace_policy: str = 'strict_spd'
    exact_condition_max_dim: int = 512
    lanczos_steps: int = 128
    lanczos_starts: int = 3
    residual_tolerance: float = 1e-7
    relative_positive_threshold: float = 1e-10
    retain_raw_gradients: bool = False

    def validate(self) -> 'HardenedBlockConfig':
        validate_curvature_policy(self.curvature_kind, primary_analysis=self.primary_analysis)
        if self.retain_raw_gradients:
            raise ValueError('hardened full-scale path never retains raw per-example gradients')
        if self.shampoo_exponent < 0:
            raise ValueError('shampoo_exponent must be non-negative')
        if self.shampoo_damping < 0:
            raise ValueError('shampoo_damping must be non-negative')
        if self.shampoo_damping_ratio is not None and self.shampoo_damping_ratio < 0:
            raise ValueError('shampoo_damping_ratio must be non-negative')
        return self


@dataclass(frozen=True)
class HardenedBlockResult:
    row: Mapping[str, Any]
    moments: MatrixMomentStatistics


def _operator_dim(operator: Any) -> int:
    for name in ('dimension', 'dim'):
        value = getattr(operator, name, None)
        if value is not None:
            return int(value() if callable(value) else value)
    return int(operator.shape[0])


def _matvec(operator: Any, vector: Tensor) -> Tensor:
    return operator.matvec(vector) if hasattr(operator, 'matvec') else operator @ vector


def _two_sided_effective_operator(
    curvature: Any,
    preconditioner: Any,
    *,
    dimension: int,
    dtype: torch.dtype,
    device: torch.device,
) -> FunctionOperator:
    def matvec(vector: Tensor) -> Tensor:
        x = preconditioner.apply_sqrt(vector) if hasattr(preconditioner, 'apply_sqrt') else preconditioner.sqrt_matvec(vector)
        y = _matvec(curvature, x)
        return preconditioner.apply_sqrt(y) if hasattr(preconditioner, 'apply_sqrt') else preconditioner.sqrt_matvec(y)
    return FunctionOperator(dimension=dimension, function=matvec, dtype=dtype, device=device, name='effective_curvature')


def _invalid_condition(
    *,
    reason: str,
    context: Mapping[str, Any],
    writer: ConditionDiagnosticsWriter | None,
    config: HardenedBlockConfig,
) -> SimpleNamespace:
    record = {
        'condition_number': None,
        'minimum_eigenvalue': None,
        'maximum_eigenvalue': None,
        'minimum_residual': None,
        'maximum_residual': None,
        'censored': True,
        'censor_reason': reason,
        'method': 'preconditioner_validation',
        'positive_threshold': config.relative_positive_threshold,
        'null_eigenvalues': 0,
        'negative_eigenvalues': 0,
        'starts': config.lanczos_starts,
        'lanczos_steps': config.lanczos_steps,
        **context,
    }
    if writer is not None:
        writer.write(record)
    return SimpleNamespace(**record)


class _AdamPreconditioner:
    def __init__(self, diagonal: Tensor, damping: float) -> None:
        shifted = diagonal.reshape(-1) + damping
        if (shifted <= 0).any():
            raise ValueError('Adam statistic plus damping must be positive in strict full-space mode')
        self.inverse_sqrt = shifted.rsqrt()
        self.inverse_quarter = shifted.pow(-0.25)
    def apply(self, vector: Tensor) -> Tensor:
        return self.inverse_sqrt * vector
    def apply_sqrt(self, vector: Tensor) -> Tensor:
        return self.inverse_quarter * vector


class _ShampooPreconditioner:
    def __init__(
        self,
        left: Tensor,
        right: Tensor,
        damping_left: float,
        damping_right: float,
        exponent: float,
        policy: str,
        relative_eigenvalue_floor: float,
    ) -> None:
        self.rows = left.shape[0]
        self.cols = right.shape[0]
        # P = R^{-alpha} \otimes L^{-alpha}; P^{1/2} uses exponent -alpha/2.
        self.left_half = symmetric_matrix_power(
            left,
            -0.5 * exponent,
            damping=damping_left,
            relative_eigenvalue_floor=relative_eigenvalue_floor,
            subspace_policy=policy,
        )
        self.right_half = symmetric_matrix_power(
            right,
            -0.5 * exponent,
            damping=damping_right,
            relative_eigenvalue_floor=relative_eigenvalue_floor,
            subspace_policy=policy,
        )
    def apply_sqrt(self, vector: Tensor) -> Tensor:
        matrix = vector.reshape(self.rows, self.cols)
        # vec_r(L X R) = (L \otimes R^T) under row-major flattening.
        return (self.left_half @ matrix @ self.right_half).reshape(-1)


def analyze_block_streaming(
    *,
    curvature_operator: Any,
    gradient_factory: Callable[[], Iterable[Tensor]],
    rows: int,
    cols: int,
    example_count: int,
    config: HardenedBlockConfig = HardenedBlockConfig(),
    output_dir: str | Path | None = None,
    config_path: str | Path | None = None,
    metadata: Optional[Mapping[str, Any]] = None,
) -> HardenedBlockResult:
    """Primary full-scale analysis path with O(d+r^2+c^2) gradient memory."""
    config.validate()
    if _operator_dim(curvature_operator) != rows * cols:
        raise ValueError('curvature dimension does not match block shape')
    moments = accumulate_matrix_moments(
        gradient_factory(), rows, cols, population=config.population_moments, dtype=torch.float64, device='cpu'
    )
    if moments.count != example_count:
        raise ValueError(f'expected {example_count} examples, observed {moments.count}')
    adam_diag = moments.adam_diagonal_centered if config.centered_moments else moments.adam_diagonal_uncentered
    left = moments.left_centered if config.centered_moments else moments.left_uncentered
    right = moments.right_centered if config.centered_moments else moments.right_uncentered
    factor_damping_error: str | None = None
    if config.shampoo_damping_ratio is None:
        damping_left = float(config.shampoo_damping)
        damping_right = float(config.shampoo_damping)
        normalized_damping: FactorwiseDamping | None = None
    else:
        try:
            normalized_damping = factorwise_damping(
                left,
                right,
                normalized_ratio=config.shampoo_damping_ratio,
                relative_threshold=config.relative_positive_threshold,
            )
            damping_left = normalized_damping.left
            damping_right = normalized_damping.right
        except ValueError as error:
            # The normalized rule has no meaningful reference scale. Preserve
            # the block result and censor Shampoo rather than inventing an
            # absolute fallback such as 1.0.
            normalized_damping = None
            damping_left = float('nan')
            damping_right = float('nan')
            factor_damping_error = f'invalid_factor_damping:{error}'
    dtype = getattr(curvature_operator, 'dtype', torch.float64)
    device = torch.device(getattr(curvature_operator, 'device', 'cpu'))
    dimension = rows * cols
    diagnostics_path = None
    diagnostics_writer = None
    if output_dir is not None:
        output = resolve_output_dir(output_dir, config_path=config_path, create=True)
        diagnostics_path = output / 'solver_diagnostics.jsonl'
        diagnostics_writer = ConditionDiagnosticsWriter(diagnostics_path)
    common = dict(
        exact_condition_max_dim=config.exact_condition_max_dim,
        lanczos_steps=config.lanczos_steps,
        starts=config.lanczos_starts,
        residual_tolerance=config.residual_tolerance,
        relative_positive_threshold=config.relative_positive_threshold,
        subspace_policy=config.subspace_policy,
        diagnostics_writer=diagnostics_writer,
    )
    base_context = dict(metadata or {})
    condition_h = estimate_condition(
        curvature_operator,
        diagnostic_context={**base_context, 'operator': 'curvature'},
        **common,
    )
    try:
        adam = _AdamPreconditioner(
            adam_diag.to(device=device, dtype=dtype), config.adam_damping
        )
        condition_adam = estimate_condition(
            _two_sided_effective_operator(
                curvature_operator,
                adam,
                dimension=dimension,
                dtype=dtype,
                device=device,
            ),
            diagnostic_context={**base_context, 'operator': 'adam_effective'},
            **common,
        )
    except ValueError as error:
        condition_adam = _invalid_condition(
            reason=f'invalid_preconditioner:{error}',
            context={**base_context, 'operator': 'adam_effective'},
            writer=diagnostics_writer,
            config=config,
        )
    if factor_damping_error is not None:
        condition_shampoo = _invalid_condition(
            reason=factor_damping_error,
            context={**base_context, 'operator': 'shampoo_effective'},
            writer=diagnostics_writer,
            config=config,
        )
    else:
        try:
            shampoo = _ShampooPreconditioner(
                left.to(device=device, dtype=dtype),
                right.to(device=device, dtype=dtype),
                damping_left,
                damping_right,
                config.shampoo_exponent,
                config.subspace_policy,
                config.relative_positive_threshold,
            )
            condition_shampoo = estimate_condition(
                _two_sided_effective_operator(
                    curvature_operator,
                    shampoo,
                    dimension=dimension,
                    dtype=dtype,
                    device=device,
                ),
                diagnostic_context={**base_context, 'operator': 'shampoo_effective'},
                **common,
            )
        except ValueError as error:
            condition_shampoo = _invalid_condition(
                reason=f'invalid_preconditioner:{error}',
                context={**base_context, 'operator': 'shampoo_effective'},
                writer=diagnostics_writer,
                config=config,
            )
    def value(obj: Any, names: tuple[str, ...], default: Any = None) -> Any:
        for name in names:
            if hasattr(obj, name):
                return getattr(obj, name)
        return default
    def finite_or_nan(raw: Any) -> float:
        if raw is None:
            return float('nan')
        result = float(raw)
        return result if math.isfinite(result) else float('nan')

    kh = finite_or_nan(value(condition_h, ('condition_number', 'condition')))
    ka = finite_or_nan(value(condition_adam, ('condition_number', 'condition')))
    ks = finite_or_nan(value(condition_shampoo, ('condition_number', 'condition')))
    def factor_diagnostics(matrix: Tensor, damping: float, prefix: str) -> dict[str, float | int]:
        summary = positive_spectrum_summary(
            matrix,
            relative_threshold=config.relative_positive_threshold,
        )
        minimum = (
            float(summary.minimum_active)
            if summary.minimum_active is not None
            else float('nan')
        )
        maximum = (
            float(summary.maximum_active)
            if summary.maximum_active is not None
            else float('nan')
        )
        damping_is_finite = math.isfinite(damping)
        return {
            f'{prefix}_active_rank': summary.active_rank,
            f'{prefix}_rank_fraction': float(summary.active_rank / summary.dimension),
            f'{prefix}_numerical_null_rank': summary.numerical_null_rank,
            f'{prefix}_negative_rank': summary.negative_rank,
            f'{prefix}_spectral_threshold': summary.threshold,
            f'{prefix}_lambda_min_active': minimum,
            f'{prefix}_lambda_max_active': maximum,
            f'{prefix}_rho_over_m': (
                float(damping / minimum)
                if damping_is_finite and minimum > 0
                else float('nan')
            ),
            f'{prefix}_rho_over_M': (
                float(damping / maximum)
                if damping_is_finite and maximum > 0
                else float('nan')
            ),
        }
    factor_stats = {
        **factor_diagnostics(left, damping_left, 'left'),
        **factor_diagnostics(right, damping_right, 'right'),
    }
    row = {
        **(metadata or {}),
        **factor_stats,
        'curvature_kind': config.curvature_kind,
        'control_only': config.curvature_kind in {'fisher', 'empirical_fisher'},
        'example_count': moments.count,
        'dimension': dimension,
        'shampoo_damping_left': damping_left,
        'shampoo_damping_right': damping_right,
        'shampoo_damping_ratio': config.shampoo_damping_ratio,
        'K_curvature': kh,
        'K_adam': ka,
        'K_shampoo': ks,
        'G_adam': math.log(kh) - math.log(ka) if math.isfinite(kh) and math.isfinite(ka) else float('nan'),
        'G_shampoo': math.log(kh) - math.log(ks) if math.isfinite(kh) and math.isfinite(ks) else float('nan'),
        'delta_G': math.log(ka) - math.log(ks) if math.isfinite(ka) and math.isfinite(ks) else float('nan'),
        'curvature_censored': bool(value(condition_h, ('censored', 'is_censored'), True)),
        'adam_censored': bool(value(condition_adam, ('censored', 'is_censored'), True)),
        'shampoo_censored': bool(value(condition_shampoo, ('censored', 'is_censored'), True)),
        'curvature_censor_reason': value(condition_h, ('censor_reason', 'reason'), None),
        'adam_censor_reason': value(condition_adam, ('censor_reason', 'reason'), None),
        'shampoo_censor_reason': value(condition_shampoo, ('censor_reason', 'reason'), None),
        'left_effective_rank': moments.effective_rank(left),
        'right_effective_rank': moments.effective_rank(right),
        'adam_active_fraction': float((adam_diag > config.relative_positive_threshold * adam_diag.max().clamp_min(1e-300)).double().mean().item()),
    }
    if output_dir is not None:
        geometry_path = output / 'geometry.csv'
        append_dataframe_row(dict(row), geometry_path)
        if diagnostics_path is not None:
            condition_diagnostics_jsonl_to_csv(diagnostics_path, output / 'solver_diagnostics.csv')
        append_jsonl_strict({
            'count': moments.count, 'rows': moments.rows, 'cols': moments.cols,
            **(metadata or {}),
            'left_effective_rank': row['left_effective_rank'],
            'right_effective_rank': row['right_effective_rank'],
            'raw_gradients_retained': False,
        }, output / 'moments_summary.jsonl')
    return HardenedBlockResult(row=row, moments=moments)

