from __future__ import annotations

import hashlib
import json
import math
import traceback
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import torch

from .adapters import load_model_bundle
from .block_registry import MatrixBlock, block_metadata, discover_matrix_blocks
from .config import ensure_output_dir, save_resolved_config, validate_config
from .covariance import CovarianceEstimate, CovarianceState, bootstrap_state, merge_states
from .curvature import (
    BlockGGNOperator,
    BlockHessianOperator,
    LinearMatrixOperator,
    effective_operator,
    estimate_partial_traces_and_diagonal,
    stabilize_curvature,
)
from .data import build_dataloader_factory
from .diagnostics import (
    adam_coordinate_elasticity,
    combine_factor_elasticities,
    normalized_commutator,
    predicted_delta_g_components,
    safe_log_condition,
    visible_elasticity,
)
from .interventions import build_factor_intervention
from .linear_algebra import (
    condition_from_spectrum,
    make_lanczos_starts,
    multi_start_lanczos,
    symmetric_eigendecomp_with_metadata,
    truncated_condition_sweep,
)
from .mechanism import mean_gradient_contamination
from .partial_trace_stability import save_partial_trace_artifact
from .provenance import runtime_provenance
from .preconditioners import (
    AdamFormPreconditioner,
    ShampooFormPreconditioner,
    SymmetricSpectrum,
    resolve_factor_damping,
    resolve_relative_damping,
)
from .reliability import (
    calibrated_bootstrap_interval,
    classify_reliable_ordering,
    ritz_residual_check,
    tau_sign_stability,
)
from .utils import (
    canonical_config_hash,
    get_device,
    get_dtype,
    json_dump,
    move_batch_to_device,
    only_parameter_requires_grad,
    protocol_config_hash,
    seed_everything,
    eval_mode,
)

SCHEMA_VERSION = "1.0"
PRIMARY_ALPHA = 0.25


def _controls_enabled_for_block(analysis: Mapping[str, Any], block_name: str) -> bool:
    """Return whether preregistered expensive controls run on ``block_name``.

    An empty list preserves the historical behavior of running controls on every
    selected block. Primary block metrics are unaffected by this selector.
    """
    controls = analysis.get("controls", {})
    names = [str(value) for value in controls.get("block_names", [])]
    return not names or str(block_name) in set(names)


def _name_list_digest(names: Sequence[str]) -> str:
    encoded = json.dumps(list(names), sort_keys=False, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


BLOCK_FAILURE_COLUMNS = [
    "schema_version",
    "config_hash",
    "protocol_hash",
    "seed",
    "block_name",
    "block_type",
    "error_type",
    "error_message",
    "traceback",
]
BLOCK_METRIC_COLUMNS = [
    "schema_version",
    "config_hash",
    "protocol_hash",
    "seed",
    "block_name",
    "block_type",
    "covariance_moment",
    "assignment",
    "alpha",
    "K_adam",
    "K_shampoo",
    "delta_g",
    "endpoint_numerically_reliable",
    "ordering_inferentially_reliable",
    "reliable_ordering",
    "reliability_reasons",
]
BOOTSTRAP_COLUMNS = [
    "schema_version",
    "config_hash",
    "protocol_hash",
    "seed",
    "block_name",
    "block_type",
    "replicate",
    "covariance_moment",
    "assignment",
    "alpha",
    "K_H",
    "K_adam",
    "K_shampoo",
    "G_adam",
    "G_shampoo",
    "delta_g",
    "delta_g_from_gains",
    "fallback_tau",
    "condition_metric",
    "adam_condition_saturated",
    "shampoo_condition_saturated",
]
CONTROL_COLUMNS = [
    "schema_version",
    "config_hash",
    "protocol_hash",
    "seed",
    "block_name",
    "block_type",
    "covariance_moment",
    "assignment",
    "alpha",
    "damping_coefficient",
    "sweep_mode",
    "adam_damping_coefficient",
    "shampoo_damping_coefficient",
    "adam_damping",
    "shampoo_left_damping",
    "shampoo_right_damping",
    "K_H",
    "K_adam",
    "K_shampoo",
    "G_adam",
    "G_shampoo",
    "delta_g",
    "delta_g_from_gains",
    "fallback_tau",
    "control_estimand",
    "control_value",
    "delta_g_scalar_limit",
    "delta_g_distance_to_scalar_limit",
    "alpha_reference",
    "alpha_reference_delta_g",
    "alpha_delta_from_practical",
    "alpha_abs_delta_g_change",
    "condition_metric",
    "H_condition_metric",
    "H_condition_saturated",
    "adam_condition_saturated",
    "shampoo_condition_saturated",
    "endpoint_numerically_reliable",
]


RIDGE_SENSITIVITY_COLUMNS = [
    "schema_version",
    "config_hash",
    "protocol_hash",
    "seed",
    "block_name",
    "block_type",
    "covariance_moment",
    "assignment",
    "alpha",
    "ridge_coefficient",
    "ridge_mode",
    "target_ridge",
    "nominal_shift",
    "curvature_shift",
    "primary_curvature_shift",
    "curvature_raw_min_ritz",
    "curvature_raw_max_ritz",
    "K_H",
    "K_adam",
    "K_shampoo",
    "G_adam",
    "G_shampoo",
    "delta_g",
    "delta_g_from_gains",
    "condition_metric",
    "fallback_tau",
    "adam_condition_saturated",
    "shampoo_condition_saturated",
    "endpoint_numerically_reliable",
]


def ridge_sensitivity_plan(
    *,
    raw_min_ritz: float,
    raw_max_ritz: float,
    coefficients: Sequence[float],
) -> list[dict[str, float]]:
    """Return relative-ridge targets and their one-pass nominal shifts."""
    minimum = float(raw_min_ritz)
    maximum = float(raw_max_ritz)
    if not math.isfinite(minimum) or not math.isfinite(maximum):
        raise ValueError("raw Ritz endpoints must be finite")
    scale = max(abs(maximum), float(torch.finfo(torch.float64).tiny))
    rows: list[dict[str, float]] = []
    for raw_value in coefficients:
        coefficient = float(raw_value)
        if not math.isfinite(coefficient) or coefficient < 0.0:
            raise ValueError(
                "ridge sensitivity coefficients must be finite and nonnegative"
            )
        target = coefficient * scale
        rows.append(
            {
                "ridge_coefficient": coefficient,
                "target_ridge": target,
                "nominal_shift": max(0.0, -minimum + target),
            }
        )
    return rows



def load_curvature_shift_overrides(
    curvature_cfg: Mapping[str, Any],
) -> tuple[dict[str, float], str]:
    """Load a versioned block-to-shift mapping and return its SHA-256 digest."""
    raw_path = curvature_cfg.get("shift_overrides_path")
    if raw_path in (None, ""):
        return {}, ""
    path = Path(str(raw_path)).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"curvature shift override file does not exist: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("curvature shift override file must contain a JSON object")
    values = payload.get("blocks", payload)
    if not isinstance(values, Mapping):
        raise ValueError("curvature shift override document must contain a 'blocks' mapping")
    result: dict[str, float] = {}
    for name, value in values.items():
        shift = float(value)
        if not math.isfinite(shift) or shift < 0.0:
            raise ValueError(f"invalid curvature shift override for {name!r}: {value!r}")
        result[str(name)] = shift
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return result, digest


def iter_block_analyses(blocks: Sequence[Any], analyze_one: Callable[[Any], Any]):
    """Yield per-block results while preserving failures for audit."""
    for block in blocks:
        try:
            yield block, analyze_one(block), None
        except Exception as error:  # noqa: BLE001 - block-level isolation is intentional
            yield block, None, {
                "block_name": getattr(block, "name", "unknown"),
                "block_type": getattr(block, "block_type", "unknown"),
                "error_type": type(error).__name__,
                "error_message": str(error),
                "traceback": traceback.format_exc(),
            }


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], columns: Sequence[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(list(rows))
    if frame.empty and columns is not None:
        frame = pd.DataFrame(columns=list(columns))
    frame.to_csv(path, index=False)


def _take_n(factory: Callable[[], Iterable], n: int, skip: int = 0) -> list[Any]:
    output: list[Any] = []
    for index, batch in enumerate(factory()):
        if index < int(skip):
            continue
        output.append(batch)
        if len(output) >= int(n):
            break
    if len(output) < int(n):
        raise RuntimeError(
            f"Requested {n} batches after skip={skip}, but the dataloader produced {len(output)}"
        )
    return output


def collect_block_covariance(
    model: torch.nn.Module,
    block: MatrixBlock,
    factory: Callable[[], Iterable],
    loss_fn: Callable,
    device: torch.device,
    *,
    num_batches: int,
    group_size: int,
    skip_batches: int,
) -> tuple[CovarianceState, list[CovarianceState], list[float]]:
    """Collect one centered covariance state and mergeable bootstrap groups."""
    if int(group_size) <= 0:
        raise ValueError("group_size must be positive")
    groups: list[CovarianceState] = []
    current = CovarianceState.zeros(block.shape, device="cpu", dtype=torch.float64)
    losses: list[float] = []
    seen = 0
    with eval_mode(model), only_parameter_requires_grad(model, block.param):
        for index, batch in enumerate(factory()):
            if index < int(skip_batches):
                continue
            if seen >= int(num_batches):
                break
            moved = move_batch_to_device(batch, device)
            loss = loss_fn(model, moved)
            gradient_full = torch.autograd.grad(
                loss, block.param, create_graph=False, retain_graph=False
            )[0]
            gradient = block.extract(gradient_full).detach().float().cpu()
            current.update(gradient)
            losses.append(float(loss.detach().cpu()))
            seen += 1
            if current.count >= int(group_size):
                groups.append(current)
                current = CovarianceState.zeros(block.shape, device="cpu", dtype=torch.float64)
    if current.count:
        groups.append(current)
    if sum(group.count for group in groups) < 2:
        raise RuntimeError("At least two gradient batches are required for centered covariance")
    return merge_states(groups), groups, losses


def _average_operator(
    operators: Sequence[LinearMatrixOperator],
    shape: tuple[int, int],
    device: torch.device,
    dtype: torch.dtype,
) -> LinearMatrixOperator:
    if not operators:
        raise ValueError("No curvature operators were supplied")
    if len(operators) == 1:
        return operators[0]

    def matvec(matrix: torch.Tensor) -> torch.Tensor:
        output = torch.zeros(shape, device=device, dtype=dtype)
        for operator in operators:
            output.add_(operator.matvec_matrix(matrix))
        return output / len(operators)

    return LinearMatrixOperator(shape, matvec, device=device, dtype=dtype)


def _build_raw_curvature(
    *,
    kind: str,
    model: torch.nn.Module,
    block: MatrixBlock,
    batches: Sequence[Any],
    loss_fn: Callable,
    ggn_spec_fn: Callable | None,
    device: torch.device,
    dtype: torch.dtype,
) -> LinearMatrixOperator:
    if kind == "ggn":
        if ggn_spec_fn is None:
            raise RuntimeError("curvature.kind=ggn requires a causal-LM GGN adapter")
        operators = [
            BlockGGNOperator(model, block, batch, ggn_spec_fn, device=device, dtype=dtype)
            for batch in batches
        ]
    elif kind == "hessian":
        operators = [
            BlockHessianOperator(model, block, batch, loss_fn, device=device, dtype=dtype)
            for batch in batches
        ]
    else:
        raise ValueError("curvature.kind must be ggn or hessian")
    return _average_operator(operators, block.shape, device, dtype)


def _condition_record(
    min_eig: float,
    max_eig: float,
    *,
    relative_floor: float,
    fallback_tau: float,
    force_truncated: bool,
) -> tuple[float, str, bool]:
    ordinary = condition_from_spectrum(min_eig, max_eig, rel_floor=relative_floor)
    if math.isfinite(ordinary) and not force_truncated:
        return ordinary, "ordinary", False
    truncated = condition_from_spectrum(
        min_eig, max_eig, truncation_tau=float(fallback_tau)
    )
    saturated = bool(
        math.isfinite(max_eig)
        and max_eig > 0
        and min_eig <= float(fallback_tau) * max_eig
    )
    return truncated, "truncated", saturated


def _pair_conditions(
    adam_spec: Mapping[str, Any],
    shampoo_spec: Mapping[str, Any],
    *,
    relative_floor: float,
    fallback_tau: float,
    reference_spec: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate Adam, Shampoo, and optional reference curvature compatibly.

    Congruence by a positive-definite preconditioner preserves nullity.  If
    the unpreconditioned reference curvature has no accepted ordinary lower
    endpoint, finite ordinary Ritz minima for the two effective operators are
    not treated as evidence of finite ordinary condition numbers.  The whole
    comparison is then evaluated with the same relative-tau truncation.
    """
    ordinary_adam = condition_from_spectrum(
        float(adam_spec["min_ritz"]),
        float(adam_spec["max_ritz"]),
        rel_floor=relative_floor,
    )
    ordinary_shampoo = condition_from_spectrum(
        float(shampoo_spec["min_ritz"]),
        float(shampoo_spec["max_ritz"]),
        rel_floor=relative_floor,
    )
    ordinary_reference = (
        condition_from_spectrum(
            float(reference_spec["min_ritz"]),
            float(reference_spec["max_ritz"]),
            rel_floor=relative_floor,
        )
        if reference_spec is not None
        else 1.0
    )
    force_truncated = not all(
        math.isfinite(value)
        for value in (ordinary_reference, ordinary_adam, ordinary_shampoo)
    )
    K_adam, metric_a, saturated_a = _condition_record(
        float(adam_spec["min_ritz"]),
        float(adam_spec["max_ritz"]),
        relative_floor=relative_floor,
        fallback_tau=fallback_tau,
        force_truncated=force_truncated,
    )
    K_shampoo, metric_s, saturated_s = _condition_record(
        float(shampoo_spec["min_ritz"]),
        float(shampoo_spec["max_ritz"]),
        relative_floor=relative_floor,
        fallback_tau=fallback_tau,
        force_truncated=force_truncated,
    )
    if metric_a != metric_s:
        raise RuntimeError("Adam and Shampoo condition metrics must match")
    delta = safe_log_condition(K_adam) - safe_log_condition(K_shampoo)
    record: dict[str, Any] = {
        "K_adam": K_adam,
        "K_shampoo": K_shampoo,
        "delta_g": delta,
        "condition_metric": metric_a,
        "fallback_tau": float(fallback_tau),
        "adam_condition_saturated": saturated_a,
        "shampoo_condition_saturated": saturated_s,
    }
    if reference_spec is not None:
        K_H, H_saturated = _condition_for_metric(
            reference_spec,
            metric=metric_a,
            relative_floor=relative_floor,
            fallback_tau=fallback_tau,
        )
        record.update(
            {
                "K_H": K_H,
                "H_condition_metric": metric_a,
                "H_condition_saturated": H_saturated,
                **_gain_record(
                    K_H=K_H, K_adam=K_adam, K_shampoo=K_shampoo
                ),
            }
        )
    return record



def _condition_for_metric(
    spectrum: Mapping[str, Any],
    *,
    metric: str,
    relative_floor: float,
    fallback_tau: float,
) -> tuple[float, bool]:
    """Evaluate one Ritz interval with an explicitly declared condition metric."""
    minimum = float(spectrum["min_ritz"])
    maximum = float(spectrum["max_ritz"])
    if metric == "ordinary":
        value = condition_from_spectrum(minimum, maximum, rel_floor=relative_floor)
        return value, False
    if metric in {"truncated", "truncated_condition"}:
        value = condition_from_spectrum(
            minimum, maximum, truncation_tau=float(fallback_tau)
        )
        saturated = bool(
            math.isfinite(maximum)
            and maximum > 0.0
            and minimum <= float(fallback_tau) * maximum
        )
        return value, saturated
    raise ValueError(f"unsupported condition metric: {metric!r}")


def _gain_record(*, K_H: float, K_adam: float, K_shampoo: float) -> dict[str, float]:
    """Return scale-invariant gains computed with one compatible metric."""
    log_h = safe_log_condition(float(K_H))
    log_adam = safe_log_condition(float(K_adam))
    log_shampoo = safe_log_condition(float(K_shampoo))
    g_adam = log_h - log_adam
    g_shampoo = log_h - log_shampoo
    return {
        "G_adam": g_adam,
        "G_shampoo": g_shampoo,
        "delta_g_from_gains": g_shampoo - g_adam,
    }


def _control_estimand_record(
    *,
    sweep_mode: str,
    delta_g: float,
    G_adam: float,
    G_shampoo: float,
) -> dict[str, float | str]:
    """Declare the theory-aligned scalar target for a frozen control row."""
    delta = float(delta_g)
    g_adam = float(G_adam)
    g_shampoo = float(G_shampoo)
    scalar_limit = -g_adam
    distance = abs(delta - scalar_limit) if all(
        math.isfinite(value) for value in (delta, scalar_limit)
    ) else float("nan")
    if sweep_mode == "joint":
        estimand = "abs_delta_g"
        value = abs(delta) if math.isfinite(delta) else float("nan")
    elif sweep_mode == "shampoo_only":
        estimand = "abs_g_shampoo"
        value = abs(g_shampoo) if math.isfinite(g_shampoo) else float("nan")
    else:
        estimand = "signed_delta_g"
        value = delta
    return {
        "control_estimand": estimand,
        "control_value": value,
        "delta_g_scalar_limit": scalar_limit,
        "delta_g_distance_to_scalar_limit": distance,
    }


def _annotate_alpha_control_rows(
    rows: Sequence[Mapping[str, Any]], *, practical_alpha: float = PRIMARY_ALPHA
) -> list[dict[str, Any]]:
    """Attach within-block signed alpha responses relative to practical Shampoo."""
    copied = [dict(row) for row in rows]
    group_keys = (
        "protocol_hash",
        "seed",
        "block_name",
        "covariance_moment",
        "assignment",
        "damping_coefficient",
    )
    baselines: dict[tuple[Any, ...], float] = {}
    for row in copied:
        try:
            alpha = float(row.get("alpha", float("nan")))
            delta = float(row.get("delta_g", float("nan")))
        except (TypeError, ValueError):
            continue
        if math.isclose(alpha, float(practical_alpha), rel_tol=0.0, abs_tol=1.0e-12):
            key = tuple(row.get(name) for name in group_keys)
            baselines[key] = delta
    for row in copied:
        key = tuple(row.get(name) for name in group_keys)
        baseline = baselines.get(key, float("nan"))
        try:
            delta = float(row.get("delta_g", float("nan")))
        except (TypeError, ValueError):
            delta = float("nan")
        change = delta - baseline if math.isfinite(delta) and math.isfinite(baseline) else float("nan")
        row["alpha_reference"] = float(practical_alpha)
        row["alpha_reference_delta_g"] = baseline
        row["alpha_delta_from_practical"] = change
        row["alpha_abs_delta_g_change"] = (
            abs(delta) - abs(baseline)
            if math.isfinite(delta) and math.isfinite(baseline)
            else float("nan")
        )
        row["control_estimand"] = (
            f"signed_delta_g_change_from_alpha_{float(practical_alpha):g}"
        )
        row["control_value"] = change
    return copied


def _tau_deltas(
    adam_spec: Mapping[str, Any],
    shampoo_spec: Mapping[str, Any],
    taus: Sequence[float],
) -> dict[float, float]:
    adam = truncated_condition_sweep(
        float(adam_spec["min_ritz"]), float(adam_spec["max_ritz"]), taus
    )
    shampoo = truncated_condition_sweep(
        float(shampoo_spec["min_ritz"]), float(shampoo_spec["max_ritz"]), taus
    )
    return {
        float(tau): safe_log_condition(adam[float(tau)])
        - safe_log_condition(shampoo[float(tau)])
        for tau in taus
    }


def build_spectral_gain_rows(
    metadata: Mapping[str, Any],
    adam_spec: Mapping[str, Any],
    shampoo_spec: Mapping[str, Any],
    *,
    taus: Sequence[float],
    covariance_moment: str,
    assignment: str,
    alpha: float,
    sweep_mode: str,
    damping_coefficient: float = float("nan"),
    adam_damping_coefficient: float = float("nan"),
    shampoo_damping_coefficient: float = float("nan"),
) -> list[dict[str, Any]]:
    """Build auditable truncated-condition rows for one operator comparison."""
    adam_min = float(adam_spec["min_ritz"])
    adam_max = float(adam_spec["max_ritz"])
    shampoo_min = float(shampoo_spec["min_ritz"])
    shampoo_max = float(shampoo_spec["max_ritz"])
    adam_conditions = truncated_condition_sweep(adam_min, adam_max, taus)
    shampoo_conditions = truncated_condition_sweep(
        shampoo_min, shampoo_max, taus
    )
    rows: list[dict[str, Any]] = []
    for raw_tau in taus:
        tau = float(raw_tau)
        k_adam = float(adam_conditions[tau])
        k_shampoo = float(shampoo_conditions[tau])
        rows.append(
            {
                **dict(metadata),
                "covariance_moment": str(covariance_moment),
                "assignment": str(assignment),
                "alpha": float(alpha),
                "sweep_mode": str(sweep_mode),
                "damping_coefficient": float(damping_coefficient),
                "adam_damping_coefficient": float(adam_damping_coefficient),
                "shampoo_damping_coefficient": float(
                    shampoo_damping_coefficient
                ),
                "tau": tau,
                "K_adam": k_adam,
                "K_shampoo": k_shampoo,
                "delta_g": safe_log_condition(k_adam)
                - safe_log_condition(k_shampoo),
                "condition_metric": "truncated_condition",
                "adam_condition_saturated": bool(
                    not math.isfinite(adam_min)
                    or not math.isfinite(adam_max)
                    or adam_max <= 0.0
                    or adam_min <= tau * adam_max
                ),
                "shampoo_condition_saturated": bool(
                    not math.isfinite(shampoo_min)
                    or not math.isfinite(shampoo_max)
                    or shampoo_max <= 0.0
                    or shampoo_min <= tau * shampoo_max
                ),
            }
        )
    return rows


def _endpoint_numerical_reliability(
    specs: Sequence[Mapping[str, Any]], reliability_cfg: Mapping[str, Any]
) -> dict[str, Any]:
    relative_min = float(
        reliability_cfg.get("max_min_ritz_residual_over_min", 0.25)
    )
    relative_max = float(
        reliability_cfg.get("max_max_ritz_residual_over_max", 1.0e-3)
    )
    passed = all(
        ritz_residual_check(
            float(spec.get("min_ritz", float("nan"))),
            float(spec.get("max_ritz", float("nan"))),
            float(spec.get("min_ritz_residual", float("nan"))),
            float(spec.get("max_ritz_residual", float("nan"))),
            relative_to_min=relative_min,
            relative_to_max=relative_max,
        )
        for spec in specs
    )
    return {
        "numerically_reliable": bool(passed),
        "numerical_reliability_reasons": "" if passed else "ritz_residual",
    }


def _pair_numerical_reliability(
    adam_spec: Mapping[str, Any],
    shampoo_spec: Mapping[str, Any],
    reliability_cfg: Mapping[str, Any],
    *,
    tau_sweep: Sequence[float],
    rel_floor: float,
    curvature_shift_ok: bool,
) -> dict[str, Any]:
    del rel_floor
    endpoint = _endpoint_numerical_reliability(
        [adam_spec, shampoo_spec], reliability_cfg
    )
    stability = tau_sign_stability(
        _tau_deltas(adam_spec, shampoo_spec, tau_sweep),
        zero_tol=float(reliability_cfg.get("sign_zero_tol", 1.0e-12)),
    )
    reasons: list[str] = []
    if not endpoint["numerically_reliable"]:
        reasons.append("ritz_residual")
    if not stability["stable"]:
        reasons.append("tau_sign")
    if not curvature_shift_ok:
        reasons.append("curvature_shift")
    return {
        "numerically_reliable": not reasons,
        "numerical_reliability_reasons": ",".join(reasons),
        "tau_sign_stable": bool(stability["stable"]),
        "tau_sign": int(stability["sign"]),
        "tau_sign_reason": str(stability["reason"]),
    }


def _spec_columns(prefix: str, spec: Mapping[str, Any]) -> dict[str, Any]:
    return {
        f"lambda_min_{prefix}": float(spec["min_ritz"]),
        f"lambda_max_{prefix}": float(spec["max_ritz"]),
        f"min_ritz_residual_{prefix}": float(spec["min_ritz_residual"]),
        f"max_ritz_residual_{prefix}": float(spec["max_ritz_residual"]),
        f"lanczos_steps_{prefix}": int(spec["steps"]),
        f"lanczos_starts_{prefix}": int(spec["starts"]),
    }


def _resolve_dampings(
    covariance: CovarianceEstimate,
    preconditioner_cfg: Mapping[str, Any],
    *,
    coefficient_override: float | None = None,
    adam_coefficient_override: float | None = None,
    shampoo_coefficient_override: float | None = None,
) -> dict[str, float]:
    adam_cfg = preconditioner_cfg.get("adam", {})
    shampoo_cfg = preconditioner_cfg.get("shampoo", {})
    if coefficient_override is not None and (
        adam_coefficient_override is not None
        or shampoo_coefficient_override is not None
    ):
        raise ValueError(
            "coefficient_override cannot be combined with separate damping overrides"
        )
    adam_coefficient = float(
        coefficient_override
        if coefficient_override is not None
        else (
            adam_coefficient_override
            if adam_coefficient_override is not None
            else adam_cfg.get("damping_coefficient", 1.0e-3)
        )
    )
    shampoo_coefficient = float(
        coefficient_override
        if coefficient_override is not None
        else (
            shampoo_coefficient_override
            if shampoo_coefficient_override is not None
            else shampoo_cfg.get("damping_coefficient", 1.0e-3)
        )
    )
    return {
        "adam_coefficient": adam_coefficient,
        "shampoo_coefficient": shampoo_coefficient,
        "adam": resolve_relative_damping(
            covariance.diag,
            coefficient=adam_coefficient,
            statistic=str(adam_cfg.get("damping_statistic", "median")),
            minimum=float(adam_cfg.get("damping_min", 0.0)),
        ),
        "left": resolve_factor_damping(
            covariance.left,
            coefficient=shampoo_coefficient,
            statistic=str(shampoo_cfg.get("damping_statistic", "lambda_max")),
            minimum=float(shampoo_cfg.get("damping_min", 0.0)),
        ),
        "right": resolve_factor_damping(
            covariance.right,
            coefficient=shampoo_coefficient,
            statistic=str(shampoo_cfg.get("damping_statistic", "lambda_max")),
            minimum=float(shampoo_cfg.get("damping_min", 0.0)),
        ),
    }


def damping_sweep_plan(
    preconditioner_cfg: Mapping[str, Any],
    damping_cfg: Mapping[str, Any],
) -> list[dict[str, float | str]]:
    """Resolve joint and Shampoo-only damping controls deterministically."""
    modes = [str(value) for value in damping_cfg.get("modes", ["joint"])]
    coefficients = [
        float(value)
        for value in damping_cfg.get("coefficients", [0.0, 0.01, 1.0])
    ]
    base_adam = float(
        preconditioner_cfg.get("adam", {}).get("damping_coefficient", 1.0e-3)
    )
    rows: list[dict[str, float | str]] = []
    for mode in modes:
        if mode not in {"joint", "shampoo_only"}:
            raise ValueError(
                "analysis.damping_sweep.modes may contain only joint and shampoo_only"
            )
        for coefficient in coefficients:
            rows.append(
                {
                    "sweep_mode": mode,
                    "damping_coefficient": coefficient,
                    "adam_damping_coefficient": (
                        coefficient if mode == "joint" else base_adam
                    ),
                    "shampoo_damping_coefficient": coefficient,
                }
            )
    return rows


def _make_adam(
    covariance: CovarianceEstimate,
    damping: float,
    cfg: Mapping[str, Any],
) -> AdamFormPreconditioner:
    adam_cfg = cfg.get("adam", {})
    return AdamFormPreconditioner(
        covariance.diag,
        damping=damping,
        relative_floor=float(adam_cfg.get("relative_floor", 64.0)),
        absolute_floor=float(adam_cfg.get("absolute_floor", 0.0)),
    )


def _make_shampoo(
    left: torch.Tensor,
    right: torch.Tensor,
    dampings: tuple[float, float],
    alpha: float,
    cfg: Mapping[str, Any],
    spectra: tuple[SymmetricSpectrum, SymmetricSpectrum] | None = None,
) -> ShampooFormPreconditioner:
    shampoo_cfg = cfg.get("shampoo", {})
    left_spectrum, right_spectrum = spectra or (None, None)
    return ShampooFormPreconditioner(
        left,
        right,
        damping=dampings,
        factor_exponent=alpha,
        eig_floor=float(shampoo_cfg.get("eig_floor", 0.0)),
        relative_eig_floor=float(shampoo_cfg.get("relative_eig_floor", 64.0)),
        left_spectrum=left_spectrum,
        right_spectrum=right_spectrum,
    )


class _LanczosSpectrumCache:
    """Block-local cache for deterministic fixed-operator Lanczos spectra."""

    def __init__(self) -> None:
        self._values: dict[tuple[Any, ...], dict[str, Any]] = {}
        self._hits = 0
        self._misses = 0

    def get_or_compute(
        self,
        key: Sequence[Any],
        compute: Callable[[], dict[str, Any]],
    ) -> dict[str, Any]:
        normalized = tuple(key)
        if normalized in self._values:
            self._hits += 1
            return self._values[normalized]
        value = compute()
        self._values[normalized] = value
        self._misses += 1
        return value

    def diagnostics(self) -> dict[str, int]:
        return {
            "entries": len(self._values),
            "hits": self._hits,
            "misses": self._misses,
        }


def _float_cache_token(value: float) -> str:
    return float(value).hex()


def _evaluate_preconditioner_pair(
    H: LinearMatrixOperator,
    adam: AdamFormPreconditioner,
    shampoo: ShampooFormPreconditioner,
    *,
    starts: Sequence[torch.Tensor],
    steps: int,
    condition_cfg: Mapping[str, Any],
    spectrum_cache: _LanczosSpectrumCache | None = None,
    adam_cache_key: Sequence[Any] | None = None,
    shampoo_cache_key: Sequence[Any] | None = None,
    reference_spec: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    adam_operator = effective_operator(H, adam.apply_half)
    shampoo_operator = effective_operator(H, shampoo.apply_half)

    def compute_adam() -> dict[str, Any]:
        return multi_start_lanczos(
            adam_operator.matvec,
            H.dim,
            steps=steps,
            device=H.device,
            dtype=H.dtype,
            starts=starts,
        )

    def compute_shampoo() -> dict[str, Any]:
        return multi_start_lanczos(
            shampoo_operator.matvec,
            H.dim,
            steps=steps,
            device=H.device,
            dtype=H.dtype,
            starts=starts,
        )

    adam_spec = (
        spectrum_cache.get_or_compute(adam_cache_key, compute_adam)
        if spectrum_cache is not None and adam_cache_key is not None
        else compute_adam()
    )
    shampoo_spec = (
        spectrum_cache.get_or_compute(shampoo_cache_key, compute_shampoo)
        if spectrum_cache is not None and shampoo_cache_key is not None
        else compute_shampoo()
    )
    pair = _pair_conditions(
        adam_spec,
        shampoo_spec,
        relative_floor=float(condition_cfg.get("relative_floor", 1.0e-8)),
        fallback_tau=float(condition_cfg.get("fallback_tau", 1.0e-4)),
        reference_spec=reference_spec,
    )
    return pair, adam_spec, shampoo_spec


def _factor_cache_tuple(spectrum: SymmetricSpectrum) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    dimension = int(spectrum.values.numel())
    return spectrum.values, spectrum.vectors, {
        "scope": "full",
        "dimension": dimension,
        "modes_returned": dimension,
        "mode_fraction": 1.0,
        "max_relative_residual": 0.0,
    }


def _factor_metrics(
    H_left: torch.Tensor,
    H_right: torch.Tensor,
    H_diag: torch.Tensor,
    covariance: CovarianceEstimate,
    dampings: Mapping[str, float],
    factor_cfg: Mapping[str, Any],
    factor_spectra: tuple[SymmetricSpectrum, SymmetricSpectrum],
    curvature_caches: tuple[tuple, tuple],
    *,
    alpha: float,
) -> dict[str, Any]:
    common = {
        "max_full_eigh_dim": int(factor_cfg.get("max_full_eigh_dim", 4096)),
        "approx_k": int(factor_cfg.get("approx_eig_k", 64)),
        "curvature_rel_floor": float(factor_cfg.get("curvature_rel_floor", 1.0e-6)),
    }
    fit_left, aux_left = visible_elasticity(
        H_left,
        covariance.left,
        damping=float(dampings["left"]),
        curvature_eigendecomp=curvature_caches[0],
        covariance_eigendecomp=_factor_cache_tuple(factor_spectra[0]),
        **common,
    )
    fit_right, aux_right = visible_elasticity(
        H_right,
        covariance.right,
        damping=float(dampings["right"]),
        curvature_eigendecomp=curvature_caches[1],
        covariance_eigendecomp=_factor_cache_tuple(factor_spectra[1]),
        **common,
    )

    def width(aux: Mapping[str, Any]) -> float:
        low = float(aux.get("h_min_used", float("nan")))
        high = float(aux.get("h_max_used", float("nan")))
        return math.log(high / low) if high > low > 0 else 0.0

    width_left = width(aux_left)
    width_right = width(aux_right)
    adam_fit, adam_aux = adam_coordinate_elasticity(
        H_diag,
        covariance.diag,
        damping=float(dampings["adam"]),
        curvature_rel_floor=float(factor_cfg.get("curvature_rel_floor", 1.0e-6)),
    )
    r_sh = combine_factor_elasticities(
        fit_left, fit_right, width_left, width_right
    )
    prediction = predicted_delta_g_components(
        r_left=fit_left.slope,
        width_left=width_left,
        r_right=fit_right.slope,
        width_right=width_right,
        r_adam=adam_fit.slope,
        width_adam=float(adam_aux["curvature_log_width"]),
        factor_exponent=alpha,
    )
    return {
        "r_left": fit_left.slope,
        "r_right": fit_right.slope,
        "r_shampoo": r_sh,
        "r_adam": adam_fit.slope,
        "r2_left": fit_left.r2,
        "r2_right": fit_right.r2,
        "r2_adam": adam_fit.r2,
        "curvature_log_width_left": width_left,
        "curvature_log_width_right": width_right,
        "curvature_log_width_adam": float(adam_aux["curvature_log_width"]),
        "commutator_left": normalized_commutator(H_left, covariance.left),
        "commutator_right": normalized_commutator(H_right, covariance.right),
        "principal_angle_left_deg": aux_left["mean_top_principal_angle_deg"],
        "principal_angle_right_deg": aux_right["mean_top_principal_angle_deg"],
        "overlap_left": aux_left["top_subspace_overlap_fro2"],
        "overlap_right": aux_right["top_subspace_overlap_fro2"],
        "elasticity_scope_left": aux_left["eigenspectrum_scope"],
        "elasticity_scope_right": aux_right["eigenspectrum_scope"],
        "elasticity_eigen_max_residual_left": aux_left["eigen_max_relative_residual"],
        "elasticity_eigen_max_residual_right": aux_right["eigen_max_relative_residual"],
        "partial_trace_negative_spectral_mass_left": aux_left["negative_spectral_mass"],
        "partial_trace_negative_spectral_mass_right": aux_right["negative_spectral_mass"],
        "num_factor_modes_left": int(aux_left["num_factor_modes"]),
        "num_factor_modes_right": int(aux_right["num_factor_modes"]),
        "num_coordinate_modes_adam": int(adam_aux["num_coordinate_modes"]),
        "rank_corr_left": aux_left["rank_corr_h_q"],
        "rank_corr_right": aux_right["rank_corr_h_q"],
        "rank_corr_adam": adam_aux["rank_corr_coordinate"],
        **prediction,
        # Backward-compatible alias.  This is the full commuting--Kronecker
        # proxy, not the response-consumption term alone.
        "delta_g_predicted": prediction["delta_g_predicted_full_proxy"],
    }


def _factor_reliability(metrics: Mapping[str, Any], cfg: Mapping[str, Any]) -> tuple[bool, str]:
    min_r2 = float(cfg.get("min_r2", 0.5))
    min_modes = int(cfg.get("min_elasticity_modes", 4))
    min_width = float(cfg.get("min_curvature_log_width", 1.0e-3))
    max_floor = float(cfg.get("max_preconditioner_floored_fraction", 0.25))

    checks: list[tuple[str, bool]] = [
        (
            "r2_left",
            math.isfinite(float(metrics.get("r2_left", float("nan"))))
            and float(metrics["r2_left"]) >= min_r2,
        ),
        (
            "r2_right",
            math.isfinite(float(metrics.get("r2_right", float("nan"))))
            and float(metrics["r2_right"]) >= min_r2,
        ),
        (
            "r2_adam",
            math.isfinite(float(metrics.get("r2_adam", float("nan"))))
            and float(metrics["r2_adam"]) >= min_r2,
        ),
        (
            "commutator",
            all(
                math.isfinite(float(metrics.get(key, float("nan"))))
                and float(metrics[key]) <= float(cfg.get("max_commutator", 0.5))
                for key in ("commutator_left", "commutator_right")
            ),
        ),
        (
            "eigen_residual",
            all(
                math.isfinite(float(metrics.get(key, float("nan"))))
                and float(metrics[key])
                <= float(cfg.get("max_factor_eigen_residual", 1.0e-3))
                for key in (
                    "elasticity_eigen_max_residual_left",
                    "elasticity_eigen_max_residual_right",
                )
            ),
        ),
        (
            "negative_mass",
            all(
                math.isfinite(float(metrics.get(key, float("nan"))))
                and float(metrics[key])
                <= float(cfg.get("max_factor_negative_mass", 0.05))
                for key in (
                    "partial_trace_negative_spectral_mass_left",
                    "partial_trace_negative_spectral_mass_right",
                )
            ),
        ),
        (
            "insufficient_left_modes",
            int(metrics.get("num_factor_modes_left", 0)) >= min_modes,
        ),
        (
            "insufficient_right_modes",
            int(metrics.get("num_factor_modes_right", 0)) >= min_modes,
        ),
        (
            "insufficient_adam_modes",
            int(metrics.get("num_coordinate_modes_adam", 0)) >= min_modes,
        ),
        (
            "degenerate_left_curvature_width",
            math.isfinite(float(metrics.get("curvature_log_width_left", float("nan"))))
            and float(metrics["curvature_log_width_left"]) >= min_width,
        ),
        (
            "degenerate_right_curvature_width",
            math.isfinite(float(metrics.get("curvature_log_width_right", float("nan"))))
            and float(metrics["curvature_log_width_right"]) >= min_width,
        ),
        (
            "degenerate_adam_curvature_width",
            math.isfinite(float(metrics.get("curvature_log_width_adam", float("nan"))))
            and float(metrics["curvature_log_width_adam"]) >= min_width,
        ),
        (
            "predictor_nonfinite",
            math.isfinite(
                float(metrics.get("delta_g_predicted_full_proxy", float("nan")))
            ),
        ),
        (
            "floor_dominated_preconditioner",
            all(
                math.isfinite(float(metrics.get(key, float("nan"))))
                and float(metrics[key]) <= max_floor
                for key in (
                    "adam_floored_fraction",
                    "shampoo_left_floored_fraction",
                    "shampoo_right_floored_fraction",
                )
            ),
        ),
    ]
    failed = [name for name, passed in checks if not passed]
    return not failed, ",".join(failed)


def _base_metadata(
    cfg: Mapping[str, Any],
    block: MatrixBlock,
    model_metadata: Mapping[str, Any],
    data_metadata: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "config_hash": canonical_config_hash(cfg),
        "protocol_hash": protocol_config_hash(cfg),
        "seed": int(cfg.get("seed", 0)),
        "checkpoint_label": cfg.get(
            "checkpoint_label", cfg.get("model", {}).get("revision", "checkpoint")
        ),
        "experiment_label": cfg.get("experiment_label", "default"),
        **model_metadata,
        **data_metadata,
        **block_metadata(block),
    }


def _analyze_block(
    *,
    cfg: Mapping[str, Any],
    model: torch.nn.Module,
    loss_fn: Callable,
    ggn_spec_fn: Callable | None,
    factory: Callable[[], Iterable],
    curvature_batches: Sequence[Any],
    block: MatrixBlock,
    device: torch.device,
    model_metadata: Mapping[str, Any],
    data_metadata: Mapping[str, Any],
    curvature_shift_overrides: Mapping[str, float],
    curvature_shift_override_sha256: str,
) -> tuple[
    list[dict],
    list[dict],
    list[dict],
    list[dict],
    list[dict],
    list[dict],
    list[dict],
]:
    analysis = cfg["analysis"]
    covariance_cfg = analysis["covariance"]
    curvature_cfg = analysis["curvature"]
    condition_cfg = analysis.get("condition", {})
    preconditioner_cfg = analysis.get("preconditioners", {})
    factor_cfg = analysis.get("factor_diagnostics", {})
    reliability_cfg = analysis.get("reliability", {})
    seed = int(cfg.get("seed", 0))
    dtype = get_dtype(str(curvature_cfg.get("dtype", "float32")))

    covariance_state, groups, losses = collect_block_covariance(
        model,
        block,
        factory,
        loss_fn,
        device,
        num_batches=int(covariance_cfg.get("num_batches", 16)),
        group_size=int(covariance_cfg.get("group_size", 4)),
        skip_batches=int(covariance_cfg.get("skip_batches", 0)),
    )
    centered_cpu = covariance_state.finalize(
        ddof=int(covariance_cfg.get("ddof", 0)), dtype=torch.float64
    )
    moments_cpu = {
        "centered": centered_cpu,
        "uncentered": centered_cpu.uncentered(),
    }

    raw = _build_raw_curvature(
        kind=str(curvature_cfg.get("kind", "ggn")),
        model=model,
        block=block,
        batches=curvature_batches,
        loss_fn=loss_fn,
        ggn_spec_fn=ggn_spec_fn,
        device=device,
        dtype=dtype,
    )
    shift_override: float | None = None
    if curvature_cfg.get("shift_overrides_path"):
        if block.name not in curvature_shift_overrides:
            raise KeyError(
                f"curvature shift override is missing selected block {block.name!r}"
            )
        shift_override = float(curvature_shift_overrides[block.name])
    build = stabilize_curvature(
        raw,
        psd_mode=str(curvature_cfg.get("psd_mode", "shift")),
        ridge=float(curvature_cfg.get("ridge", 1.0e-6)),
        ridge_mode=str(curvature_cfg.get("ridge_mode", "relative_max")),
        lanczos_steps=int(curvature_cfg.get("stabilize_lanczos_steps", 12)),
        lanczos_starts=int(curvature_cfg.get("stabilize_lanczos_starts", 1)),
        seed=seed + 1009,
        max_rounds=int(curvature_cfg.get("stabilize_rounds", 2)),
        shift_override=shift_override,
    )
    H = build.operator
    lanczos_steps = int(curvature_cfg.get("lanczos_steps", 32))
    lanczos_starts = int(curvature_cfg.get("lanczos_starts", 2))
    starts = make_lanczos_starts(
        H.dim,
        lanczos_starts,
        H.device,
        H.dtype,
        seed=seed + 2017,
    )
    H_spec = multi_start_lanczos(
        H.matvec,
        H.dim,
        steps=lanczos_steps,
        device=H.device,
        dtype=H.dtype,
        starts=starts,
    )
    K_H_native, H_metric_native, H_saturated_native = _condition_record(
        float(H_spec["min_ritz"]),
        float(H_spec["max_ritz"]),
        relative_floor=float(condition_cfg.get("relative_floor", 1.0e-8)),
        fallback_tau=float(condition_cfg.get("fallback_tau", 1.0e-4)),
        force_truncated=False,
    )
    H_left, H_right, H_diag = estimate_partial_traces_and_diagonal(
        H,
        num_probes=int(curvature_cfg.get("partial_trace_probes", 16)),
        seed=seed + 3011,
    )
    partial_trace_artifact = save_partial_trace_artifact(
        Path(str(cfg.get("output_dir", "outputs/run")))
        / "partial_trace_artifacts",
        block_name=block.name,
        left=H_left,
        right=H_right,
        covariance_left=centered_cpu.left,
        covariance_right=centered_cpu.right,
    )
    curvature_cache_left = symmetric_eigendecomp_with_metadata(
        H_left,
        max_full_dim=int(factor_cfg.get("max_full_eigh_dim", 4096)),
        approx_k=int(factor_cfg.get("approx_eig_k", 64)),
    )
    curvature_cache_right = symmetric_eigendecomp_with_metadata(
        H_right,
        max_full_dim=int(factor_cfg.get("max_full_eigh_dim", 4096)),
        approx_k=int(factor_cfg.get("approx_eig_k", 64)),
    )

    shift_ratio = float(build.shift) / max(abs(float(build.raw_max_ritz)), 1.0e-30)
    curvature_shift_ok = shift_ratio <= float(reliability_cfg.get("max_shift_ratio", 0.25))
    tau_sweep = [
        float(value)
        for value in condition_cfg.get(
            "tau_sweep", [float(condition_cfg.get("fallback_tau", 1.0e-4))]
        )
    ]
    metadata = _base_metadata(cfg, block, model_metadata, data_metadata)
    contamination = mean_gradient_contamination(centered_cpu)

    block_rows: list[dict] = []
    bootstrap_rows: list[dict] = []
    intervention_rows: list[dict] = []
    alpha_rows: list[dict] = []
    damping_rows: list[dict] = []
    tau_rows: list[dict] = []
    moment_cache: dict[str, dict[str, Any]] = {}
    spectrum_cache = _LanczosSpectrumCache()
    budget_token = (lanczos_steps, lanczos_starts, seed + 2017)

    def adam_cache_key(moment: str, damping: float) -> tuple[Any, ...]:
        return (
            "adam",
            moment,
            _float_cache_token(damping),
            *budget_token,
        )

    def shampoo_cache_key(
        moment: str,
        assignment: str,
        alpha: float,
        left_damping: float,
        right_damping: float,
    ) -> tuple[Any, ...]:
        return (
            "shampoo",
            moment,
            assignment,
            _float_cache_token(alpha),
            _float_cache_token(left_damping),
            _float_cache_token(right_damping),
            *budget_token,
        )

    for moment in ("centered", "uncentered"):
        covariance = moments_cpu[moment].to(device=device, dtype=dtype)
        dampings = _resolve_dampings(covariance, preconditioner_cfg)
        factor_spectra = (
            SymmetricSpectrum.from_matrix(covariance.left),
            SymmetricSpectrum.from_matrix(covariance.right),
        )
        adam = _make_adam(covariance, dampings["adam"], preconditioner_cfg)
        shampoo = _make_shampoo(
            covariance.left,
            covariance.right,
            (dampings["left"], dampings["right"]),
            PRIMARY_ALPHA,
            preconditioner_cfg,
            spectra=factor_spectra,
        )
        pair, adam_spec, shampoo_spec = _evaluate_preconditioner_pair(
            H,
            adam,
            shampoo,
            starts=starts,
            steps=lanczos_steps,
            condition_cfg=condition_cfg,
            reference_spec=H_spec,
            spectrum_cache=spectrum_cache,
            adam_cache_key=adam_cache_key(moment, dampings["adam"]),
            shampoo_cache_key=shampoo_cache_key(
                moment,
                "observed",
                PRIMARY_ALPHA,
                dampings["left"],
                dampings["right"],
            ),
        )
        pair_reliability = _pair_numerical_reliability(
            adam_spec,
            shampoo_spec,
            reliability_cfg,
            tau_sweep=tau_sweep,
            rel_floor=float(condition_cfg.get("relative_floor", 1.0e-8)),
            curvature_shift_ok=curvature_shift_ok,
        )
        factors = _factor_metrics(
            H_left,
            H_right,
            H_diag,
            covariance,
            dampings,
            factor_cfg,
            factor_spectra,
            (curvature_cache_left, curvature_cache_right),
            alpha=PRIMARY_ALPHA,
        )
        K_H, H_saturated = _condition_for_metric(
            H_spec,
            metric=str(pair["condition_metric"]),
            relative_floor=float(condition_cfg.get("relative_floor", 1.0e-8)),
            fallback_tau=float(condition_cfg.get("fallback_tau", 1.0e-4)),
        )
        gains = _gain_record(
            K_H=K_H,
            K_adam=float(pair["K_adam"]),
            K_shampoo=float(pair["K_shampoo"]),
        )
        factor_gate_metrics = {
            **factors,
            "adam_floored_fraction": adam.floored_fraction,
            "shampoo_left_floored_fraction": shampoo.left_floored_fraction,
            "shampoo_right_floored_fraction": shampoo.right_floored_fraction,
        }
        factor_reliable, factor_reasons = _factor_reliability(
            factor_gate_metrics, reliability_cfg
        )
        row = {
            **metadata,
            "covariance_moment": moment,
            "assignment": "observed",
            "alpha": PRIMARY_ALPHA,
            "covariance_batches": int(covariance.count),
            "mean_covariance_loss": float(np.mean(losses)),
            "curvature_kind": str(curvature_cfg.get("kind", "ggn")),
            "curvature_raw_min_ritz": float(build.raw_min_ritz),
            "curvature_raw_max_ritz": float(build.raw_max_ritz),
            "curvature_shift": float(build.shift),
            "curvature_shift_source": str(build.shift_source),
            "curvature_shift_override_sha256": curvature_shift_override_sha256,
            "curvature_target_ridge": float(build.target_ridge),
            "curvature_shift_over_raw_max": shift_ratio,
            "partial_trace_artifact": str(partial_trace_artifact.resolve()),
            "K_H": K_H,
            "H_condition_metric": str(pair["condition_metric"]),
            "H_condition_saturated": H_saturated,
            "K_H_native": K_H_native,
            "H_condition_metric_native": H_metric_native,
            "H_condition_saturated_native": H_saturated_native,
            **_spec_columns("H", H_spec),
            **pair,
            **gains,
            **_spec_columns("Adam", adam_spec),
            **_spec_columns("Shampoo", shampoo_spec),
            "adam_damping": dampings["adam"],
            "shampoo_left_damping": dampings["left"],
            "shampoo_right_damping": dampings["right"],
            "adam_effective_floor": adam.effective_floor,
            "adam_floored_fraction": adam.floored_fraction,
            "shampoo_left_effective_eig_floor": shampoo.left_effective_eig_floor,
            "shampoo_right_effective_eig_floor": shampoo.right_effective_eig_floor,
            "shampoo_left_floored_fraction": shampoo.left_floored_fraction,
            "shampoo_right_floored_fraction": shampoo.right_floored_fraction,
            **factors,
            **contamination,
            "factor_elasticity_reliable": factor_reliable,
            "factor_elasticity_reliability_reasons": factor_reasons,
            "endpoint_numerically_reliable": bool(pair_reliability["numerically_reliable"]),
            "numerical_reliability_reasons": pair_reliability["numerical_reliability_reasons"],
            "tau_sign_stable": pair_reliability["tau_sign_stable"],
            "bootstrap_ci_low": float("nan"),
            "bootstrap_ci_high": float("nan"),
            "bootstrap_reps_finite": 0,
            "ordering_inferentially_reliable": False,
            "reliable_ordering": "inconclusive",
            "reliability_reasons": "bootstrap_interval_inconclusive",
        }
        block_rows.append(row)
        tau_rows.extend(
            build_spectral_gain_rows(
                metadata,
                adam_spec,
                shampoo_spec,
                taus=tau_sweep,
                covariance_moment=moment,
                assignment="observed",
                alpha=PRIMARY_ALPHA,
                sweep_mode="primary",
                damping_coefficient=float(
                    preconditioner_cfg.get("shampoo", {}).get(
                        "damping_coefficient", float("nan")
                    )
                ),
                adam_damping_coefficient=float(
                    preconditioner_cfg.get("adam", {}).get(
                        "damping_coefficient", float("nan")
                    )
                ),
                shampoo_damping_coefficient=float(
                    preconditioner_cfg.get("shampoo", {}).get(
                        "damping_coefficient", float("nan")
                    )
                ),
            )
        )
        moment_cache[moment] = {
            "covariance": covariance,
            "dampings": dampings,
            "factor_spectra": factor_spectra,
            "adam": adam,
            "adam_spec": adam_spec,
            "pair": pair,
        }

    # Delta-only grouped covariance bootstrap for the primary endpoint.
    bootstrap_cfg = analysis.get("bootstrap", {})
    bootstrap_reps = int(bootstrap_cfg.get("reps", 0))
    if bootstrap_reps > 0:
        primary = moment_cache["centered"]
        low_steps = int(bootstrap_cfg.get("lanczos_steps", max(3, lanczos_steps // 2)))
        low_starts = make_lanczos_starts(
            H.dim,
            int(bootstrap_cfg.get("lanczos_starts", 1)),
            H.device,
            H.dtype,
            seed=seed + 4001,
        )
        low_pair, _, _ = _evaluate_preconditioner_pair(
            H,
            primary["adam"],
            _make_shampoo(
                primary["covariance"].left,
                primary["covariance"].right,
                (primary["dampings"]["left"], primary["dampings"]["right"]),
                PRIMARY_ALPHA,
                preconditioner_cfg,
                spectra=primary["factor_spectra"],
            ),
            starts=low_starts,
            steps=low_steps,
            condition_cfg=condition_cfg,
            reference_spec=H_spec,
        )
        rng = np.random.default_rng(seed + 5003)
        bootstrap_values: list[float] = []
        for replicate in range(bootstrap_reps):
            state = bootstrap_state(groups, rng)
            covariance = state.finalize(
                ddof=int(covariance_cfg.get("ddof", 0)), dtype=dtype
            ).to(device=device, dtype=dtype)
            dampings = _resolve_dampings(covariance, preconditioner_cfg)
            spectra = (
                SymmetricSpectrum.from_matrix(covariance.left),
                SymmetricSpectrum.from_matrix(covariance.right),
            )
            adam = _make_adam(covariance, dampings["adam"], preconditioner_cfg)
            shampoo = _make_shampoo(
                covariance.left,
                covariance.right,
                (dampings["left"], dampings["right"]),
                PRIMARY_ALPHA,
                preconditioner_cfg,
                spectra=spectra,
            )
            pair, _, _ = _evaluate_preconditioner_pair(
                H,
                adam,
                shampoo,
                starts=low_starts,
                steps=low_steps,
                condition_cfg=condition_cfg,
                reference_spec=H_spec,
            )
            bootstrap_values.append(float(pair["delta_g"]))
            bootstrap_rows.append(
                {
                    **{key: metadata[key] for key in metadata if key in {"schema_version", "config_hash", "protocol_hash", "seed", "block_name", "block_type"}},
                    "replicate": replicate,
                    "covariance_moment": "centered",
                    "assignment": "observed",
                    "alpha": PRIMARY_ALPHA,
                    **pair,
                }
            )
        ci_low, ci_high = calibrated_bootstrap_interval(
            point=float(primary["pair"]["delta_g"]),
            bootstrap_values=bootstrap_values,
            reference=float(low_pair["delta_g"]),
            alpha=float(bootstrap_cfg.get("alpha", 0.05)),
            minimum_reps=int(bootstrap_cfg.get("minimum_reps", 2)),
        )
        centered_row = next(
            row for row in block_rows if row["covariance_moment"] == "centered"
        )
        classification = classify_reliable_ordering(
            delta=float(centered_row["delta_g"]),
            ci_low=ci_low,
            ci_high=ci_high,
            checks={
                "endpoint_numerically_reliable": bool(centered_row["endpoint_numerically_reliable"]),
                "tau_sign_stable": bool(centered_row["tau_sign_stable"]),
                "curvature_shift_ok": curvature_shift_ok,
            },
        )
        centered_row["bootstrap_ci_low"] = ci_low
        centered_row["bootstrap_ci_high"] = ci_high
        centered_row["bootstrap_reps_finite"] = int(
            np.isfinite(np.asarray(bootstrap_values, dtype=float)).sum()
        )
        centered_row["ordering_inferentially_reliable"] = bool(classification["reliable"])
        centered_row["reliable_ordering"] = str(classification["reliable_label"])
        centered_row["reliability_reasons"] = str(classification["reliability_reasons"])

    # Centered factor-assignment, alpha, and damping controls.
    centered = moment_cache["centered"]
    assignments = [str(value) for value in analysis.get("assignments", ["observed", "aligned", "reversed"])]
    block_controls_enabled = _controls_enabled_for_block(analysis, block.name)
    controls_enabled = block_controls_enabled and any(
        bool(analysis.get(name, {}).get("enabled", False))
        for name in ("interventions", "alpha_sweep", "damping_sweep")
    )
    assignment_cache: dict[str, dict[str, Any]] = {}
    if controls_enabled:
        # Exact assignment controls require complete curvature-factor bases.
        hleft_values, hleft_vectors = torch.linalg.eigh(H_left.double())
        hright_values, hright_vectors = torch.linalg.eigh(H_right.double())
        for assignment in assignments:
            left, right = build_factor_intervention(
                centered["covariance"].left,
                centered["covariance"].right,
                H_left,
                H_right,
                mode=assignment,  # type: ignore[arg-type]
                seed=seed + 6007,
                left_curvature_eigendecomp=(hleft_values, hleft_vectors),
                right_curvature_eigendecomp=(hright_values, hright_vectors),
            )
            assignment_cache[assignment] = {
                "left": left,
                "right": right,
                "spectra": (
                    SymmetricSpectrum.from_matrix(left),
                    SymmetricSpectrum.from_matrix(right),
                ),
            }

    def control_row(
        *,
        assignment: str,
        alpha: float,
        damping_coefficient: float,
        dampings: Mapping[str, float],
        adam_spec: Mapping[str, Any],
        pair: Mapping[str, Any],
        shampoo_spec: Mapping[str, Any],
        sweep_mode: str = "control",
        adam_damping_coefficient: float = float("nan"),
        shampoo_damping_coefficient: float = float("nan"),
    ) -> dict[str, Any]:
        numerical = _endpoint_numerical_reliability(
            [adam_spec, shampoo_spec], reliability_cfg
        )
        compatible_K_H, compatible_H_saturated = _condition_for_metric(
            H_spec,
            metric=str(pair["condition_metric"]),
            relative_floor=float(condition_cfg.get("relative_floor", 1.0e-8)),
            fallback_tau=float(condition_cfg.get("fallback_tau", 1.0e-4)),
        )
        gains = _gain_record(
            K_H=compatible_K_H,
            K_adam=float(pair["K_adam"]),
            K_shampoo=float(pair["K_shampoo"]),
        )
        estimand = _control_estimand_record(
            sweep_mode=sweep_mode,
            delta_g=float(pair["delta_g"]),
            G_adam=float(gains["G_adam"]),
            G_shampoo=float(gains["G_shampoo"]),
        )
        return {
            **{key: metadata[key] for key in metadata if key in {"schema_version", "config_hash", "protocol_hash", "seed", "block_name", "block_type"}},
            "covariance_moment": "centered",
            "assignment": assignment,
            "alpha": alpha,
            "damping_coefficient": damping_coefficient,
            "sweep_mode": sweep_mode,
            "adam_damping_coefficient": adam_damping_coefficient,
            "shampoo_damping_coefficient": shampoo_damping_coefficient,
            "adam_damping": dampings["adam"],
            "shampoo_left_damping": dampings["left"],
            "shampoo_right_damping": dampings["right"],
            "K_H": compatible_K_H,
            "H_condition_metric": str(pair["condition_metric"]),
            "H_condition_saturated": compatible_H_saturated,
            **pair,
            **gains,
            **estimand,
            "endpoint_numerically_reliable": bool(numerical["numerically_reliable"]),
        }

    if block_controls_enabled and bool(
        analysis.get("interventions", {}).get("enabled", False)
    ):
        for assignment in assignments:
            cached = assignment_cache[assignment]
            shampoo = _make_shampoo(
                cached["left"],
                cached["right"],
                (centered["dampings"]["left"], centered["dampings"]["right"]),
                PRIMARY_ALPHA,
                preconditioner_cfg,
                spectra=cached["spectra"],
            )
            pair, _, shampoo_spec = _evaluate_preconditioner_pair(
                H,
                centered["adam"],
                shampoo,
                starts=starts,
                steps=lanczos_steps,
                condition_cfg=condition_cfg,
                reference_spec=H_spec,
                spectrum_cache=spectrum_cache,
                adam_cache_key=adam_cache_key(
                    "centered", centered["dampings"]["adam"]
                ),
                shampoo_cache_key=shampoo_cache_key(
                    "centered",
                    assignment,
                    PRIMARY_ALPHA,
                    centered["dampings"]["left"],
                    centered["dampings"]["right"],
                ),
            )
            intervention_rows.append(
                control_row(
                    assignment=assignment,
                    alpha=PRIMARY_ALPHA,
                    damping_coefficient=float(
                        preconditioner_cfg.get("shampoo", {}).get("damping_coefficient", 1.0e-3)
                    ),
                    dampings=centered["dampings"],
                    adam_spec=centered["adam_spec"],
                    pair=pair,
                    shampoo_spec=shampoo_spec,
                    sweep_mode="intervention",
                    adam_damping_coefficient=float(
                        centered["dampings"]["adam_coefficient"]
                    ),
                    shampoo_damping_coefficient=float(
                        centered["dampings"]["shampoo_coefficient"]
                    ),
                )
            )
            tau_rows.extend(
                build_spectral_gain_rows(
                    metadata,
                    centered["adam_spec"],
                    shampoo_spec,
                    taus=tau_sweep,
                    covariance_moment="centered",
                    assignment=assignment,
                    alpha=PRIMARY_ALPHA,
                    sweep_mode="intervention",
                    damping_coefficient=float(
                        preconditioner_cfg.get("shampoo", {}).get(
                            "damping_coefficient", float("nan")
                        )
                    ),
                    adam_damping_coefficient=float(
                        preconditioner_cfg.get("adam", {}).get(
                            "damping_coefficient", float("nan")
                        )
                    ),
                    shampoo_damping_coefficient=float(
                        preconditioner_cfg.get("shampoo", {}).get(
                            "damping_coefficient", float("nan")
                        )
                    ),
                )
            )

    alpha_cfg = analysis.get("alpha_sweep", {})
    if block_controls_enabled and bool(alpha_cfg.get("enabled", False)):
        for alpha in [float(value) for value in alpha_cfg.get("values", [0.25, 0.5])]:
            for assignment in assignments:
                cached = assignment_cache[assignment]
                shampoo = _make_shampoo(
                    cached["left"],
                    cached["right"],
                    (centered["dampings"]["left"], centered["dampings"]["right"]),
                    alpha,
                    preconditioner_cfg,
                    spectra=cached["spectra"],
                )
                pair, _, shampoo_spec = _evaluate_preconditioner_pair(
                    H,
                    centered["adam"],
                    shampoo,
                    starts=starts,
                    steps=lanczos_steps,
                    condition_cfg=condition_cfg,
                    reference_spec=H_spec,
                    spectrum_cache=spectrum_cache,
                    adam_cache_key=adam_cache_key(
                        "centered", centered["dampings"]["adam"]
                    ),
                    shampoo_cache_key=shampoo_cache_key(
                        "centered",
                        assignment,
                        alpha,
                        centered["dampings"]["left"],
                        centered["dampings"]["right"],
                    ),
                )
                alpha_rows.append(
                    control_row(
                        assignment=assignment,
                        alpha=alpha,
                        damping_coefficient=float(
                            preconditioner_cfg.get("shampoo", {}).get("damping_coefficient", 1.0e-3)
                        ),
                        dampings=centered["dampings"],
                        adam_spec=centered["adam_spec"],
                        pair=pair,
                        shampoo_spec=shampoo_spec,
                        sweep_mode="alpha",
                        adam_damping_coefficient=float(
                            centered["dampings"]["adam_coefficient"]
                        ),
                        shampoo_damping_coefficient=float(
                            centered["dampings"]["shampoo_coefficient"]
                        ),
                    )
                )
                tau_rows.extend(
                    build_spectral_gain_rows(
                        metadata,
                        centered["adam_spec"],
                        shampoo_spec,
                        taus=tau_sweep,
                        covariance_moment="centered",
                        assignment=assignment,
                        alpha=alpha,
                        sweep_mode="alpha",
                        damping_coefficient=float(
                            preconditioner_cfg.get("shampoo", {}).get(
                                "damping_coefficient", float("nan")
                            )
                        ),
                        adam_damping_coefficient=float(
                            preconditioner_cfg.get("adam", {}).get(
                                "damping_coefficient", float("nan")
                            )
                        ),
                        shampoo_damping_coefficient=float(
                            preconditioner_cfg.get("shampoo", {}).get(
                                "damping_coefficient", float("nan")
                            )
                        ),
                    )
                )

    damping_cfg = analysis.get("damping_sweep", {})
    if block_controls_enabled and bool(damping_cfg.get("enabled", False)):
        for sweep in damping_sweep_plan(preconditioner_cfg, damping_cfg):
            sweep_mode = str(sweep["sweep_mode"])
            coefficient = float(sweep["damping_coefficient"])
            adam_coefficient = float(sweep["adam_damping_coefficient"])
            shampoo_coefficient = float(sweep["shampoo_damping_coefficient"])
            dampings = _resolve_dampings(
                centered["covariance"],
                preconditioner_cfg,
                adam_coefficient_override=adam_coefficient,
                shampoo_coefficient_override=shampoo_coefficient,
            )
            adam = (
                centered["adam"]
                if sweep_mode == "shampoo_only"
                else _make_adam(
                    centered["covariance"], dampings["adam"], preconditioner_cfg
                )
            )
            for assignment in assignments:
                cached = assignment_cache[assignment]
                shampoo = _make_shampoo(
                    cached["left"],
                    cached["right"],
                    (dampings["left"], dampings["right"]),
                    PRIMARY_ALPHA,
                    preconditioner_cfg,
                    spectra=cached["spectra"],
                )
                pair, adam_spec, shampoo_spec = _evaluate_preconditioner_pair(
                    H,
                    adam,
                    shampoo,
                    starts=starts,
                    steps=lanczos_steps,
                    condition_cfg=condition_cfg,
                    reference_spec=H_spec,
                    spectrum_cache=spectrum_cache,
                    adam_cache_key=adam_cache_key(
                        "centered", dampings["adam"]
                    ),
                    shampoo_cache_key=shampoo_cache_key(
                        "centered",
                        assignment,
                        PRIMARY_ALPHA,
                        dampings["left"],
                        dampings["right"],
                    ),
                )
                damping_rows.append(
                    control_row(
                        assignment=assignment,
                        alpha=PRIMARY_ALPHA,
                        damping_coefficient=coefficient,
                        dampings=dampings,
                        adam_spec=adam_spec,
                        pair=pair,
                        shampoo_spec=shampoo_spec,
                        sweep_mode=sweep_mode,
                        adam_damping_coefficient=adam_coefficient,
                        shampoo_damping_coefficient=shampoo_coefficient,
                    )
                )
                tau_rows.extend(
                    build_spectral_gain_rows(
                        metadata,
                        adam_spec,
                        shampoo_spec,
                        taus=tau_sweep,
                        covariance_moment="centered",
                        assignment=assignment,
                        alpha=PRIMARY_ALPHA,
                        sweep_mode=f"damping_{sweep_mode}",
                        damping_coefficient=coefficient,
                        adam_damping_coefficient=adam_coefficient,
                        shampoo_damping_coefficient=shampoo_coefficient,
                    )
                )

    ridge_rows: list[dict[str, Any]] = []
    ridge_cfg = analysis.get("ridge_sensitivity", {})
    if block_controls_enabled and bool(ridge_cfg.get("enabled", False)):
        sensitivity_plan = ridge_sensitivity_plan(
            raw_min_ritz=build.raw_min_ritz,
            raw_max_ritz=build.raw_max_ritz,
            coefficients=[
                float(value) for value in ridge_cfg.get("coefficients", [])
            ],
        )
        observed_shampoo = _make_shampoo(
            centered["covariance"].left,
            centered["covariance"].right,
            (centered["dampings"]["left"], centered["dampings"]["right"]),
            PRIMARY_ALPHA,
            preconditioner_cfg,
            spectra=centered["factor_spectra"],
        )
        for sensitivity in sensitivity_plan:
            coefficient = float(sensitivity["ridge_coefficient"])
            ridge_build = stabilize_curvature(
                raw,
                psd_mode=str(curvature_cfg.get("psd_mode", "shift")),
                ridge=coefficient,
                ridge_mode="relative_max",
                lanczos_steps=int(
                    curvature_cfg.get("stabilize_lanczos_steps", 12)
                ),
                lanczos_starts=int(
                    curvature_cfg.get("stabilize_lanczos_starts", 1)
                ),
                seed=seed + 1009,
                max_rounds=int(curvature_cfg.get("stabilize_rounds", 2)),
                shift_override=None,
            )
            H_ridge = ridge_build.operator
            H_ridge_spec = multi_start_lanczos(
                H_ridge.matvec,
                H_ridge.dim,
                steps=lanczos_steps,
                device=H_ridge.device,
                dtype=H_ridge.dtype,
                starts=starts,
            )
            ridge_pair, ridge_adam_spec, ridge_shampoo_spec = (
                _evaluate_preconditioner_pair(
                    H_ridge,
                    centered["adam"],
                    observed_shampoo,
                    starts=starts,
                    steps=lanczos_steps,
                    condition_cfg=condition_cfg,
                    reference_spec=H_ridge_spec,
                )
            )
            ridge_K_H, _ = _condition_for_metric(
                H_ridge_spec,
                metric=str(ridge_pair["condition_metric"]),
                relative_floor=float(condition_cfg.get("relative_floor", 1.0e-8)),
                fallback_tau=float(condition_cfg.get("fallback_tau", 1.0e-4)),
            )
            gains = _gain_record(
                K_H=ridge_K_H,
                K_adam=float(ridge_pair["K_adam"]),
                K_shampoo=float(ridge_pair["K_shampoo"]),
            )
            numerical = _endpoint_numerical_reliability(
                [ridge_adam_spec, ridge_shampoo_spec], reliability_cfg
            )
            ridge_rows.append(
                {
                    **{
                        key: metadata[key]
                        for key in metadata
                        if key
                        in {
                            "schema_version",
                            "config_hash",
                            "protocol_hash",
                            "seed",
                            "block_name",
                            "block_type",
                        }
                    },
                    "covariance_moment": "centered",
                    "assignment": "observed",
                    "alpha": PRIMARY_ALPHA,
                    "ridge_coefficient": coefficient,
                    "ridge_mode": "relative_max",
                    "target_ridge": float(ridge_build.target_ridge),
                    "nominal_shift": float(sensitivity["nominal_shift"]),
                    "curvature_shift": float(ridge_build.shift),
                    "primary_curvature_shift": float(build.shift),
                    "curvature_raw_min_ritz": float(ridge_build.raw_min_ritz),
                    "curvature_raw_max_ritz": float(ridge_build.raw_max_ritz),
                    "K_H": ridge_K_H,
                    **ridge_pair,
                    **gains,
                    "endpoint_numerically_reliable": bool(
                        numerical["numerically_reliable"]
                    ),
                }
            )

    alpha_rows = _annotate_alpha_control_rows(
        alpha_rows, practical_alpha=PRIMARY_ALPHA
    )
    cache_diagnostics = spectrum_cache.diagnostics()
    cache_columns = {
        "lanczos_spectrum_cache_entries": cache_diagnostics["entries"],
        "lanczos_spectrum_cache_hits": cache_diagnostics["hits"],
        "lanczos_spectrum_cache_misses": cache_diagnostics["misses"],
    }
    for rows in (block_rows, intervention_rows, alpha_rows, damping_rows):
        for row in rows:
            row.update(cache_columns)
    return (
        block_rows,
        bootstrap_rows,
        intervention_rows,
        alpha_rows,
        damping_rows,
        tau_rows,
        ridge_rows,
    )


def _runtime_identity(model: Mapping[str, Any], data: Mapping[str, Any]) -> str:
    payload = {
        "model": {
            key: model.get(key)
            for key in (
                "backend",
                "model_name",
                "model_revision",
                "resolved_model_commit",
            )
        },
        "data": {
            key: data.get(key)
            for key in (
                "dataset_name",
                "dataset_revision",
                "dataset_fingerprint",
                "tokenizer_name",
                "tokenizer_revision",
                "resolved_tokenizer_commit",
                "source_order_sha256",
                "packed_token_stream_sha256",
                "selected_chunk_content_sha256",
            )
        },
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def run_frozen_analysis(cfg: Mapping[str, Any]) -> Path:
    """Run the focused frozen checkpoint experiment."""
    cfg = validate_config(cfg, mode="frozen")
    output_dir = ensure_output_dir(cfg)
    save_resolved_config(cfg, output_dir)
    seed = int(cfg.get("seed", 0))
    seed_everything(seed, deterministic=bool(cfg.get("deterministic", True)))
    provenance = runtime_provenance(Path(__file__).resolve().parents[1])
    json_dump(provenance, output_dir / "runtime_provenance.json")
    device = get_device(str(cfg.get("device", "auto")))

    bundle = load_model_bundle(cfg, device)
    factory, raw_data_metadata = build_dataloader_factory(
        cfg,
        bundle.metadata["backend"],
        seed=seed,
        model_metadata=bundle.metadata,
    )
    data_metadata = {
        key: value
        for key, value in raw_data_metadata.items()
        if not str(key).startswith("_")
    }
    analysis = cfg["analysis"]
    curvature_cfg = analysis["curvature"]
    curvature_shift_overrides, curvature_shift_override_sha256 = (
        load_curvature_shift_overrides(curvature_cfg)
    )
    curvature_batches = _take_n(
        factory,
        int(curvature_cfg.get("num_batches", 1)),
        skip=int(curvature_cfg.get("skip_batches", 0)),
    )

    block_cfg = cfg.get("blocks", {})
    blocks = discover_matrix_blocks(
        bundle.model,
        include=block_cfg.get("include"),
        exclude=block_cfg.get("exclude"),
        block_types=block_cfg.get("types"),
        max_blocks=block_cfg.get("max_blocks"),
        split_fused_qkv=bool(block_cfg.get("split_fused_qkv", True)),
        min_numel=int(block_cfg.get("min_numel", 1)),
        max_numel=(
            int(block_cfg["max_numel"])
            if block_cfg.get("max_numel") is not None
            else None
        ),
        selection_strategy=str(block_cfg.get("selection_strategy", "first")),
    )
    if not blocks:
        raise RuntimeError("No matrix blocks matched the selection rules")

    selected_block_names = [block.name for block in blocks]
    expected_exact_names = [str(value) for value in block_cfg.get("exact_names", [])]
    is_scientific_confirmatory = bool(cfg.get("scientific_run", False)) and str(
        analysis.get("compute_tier", "debug")
    ) == "confirmatory"
    if is_scientific_confirmatory and selected_block_names != expected_exact_names:
        missing = [name for name in expected_exact_names if name not in selected_block_names]
        unexpected = [name for name in selected_block_names if name not in expected_exact_names]
        raise RuntimeError(
            "Exact confirmatory block preregistration mismatch: "
            f"missing={missing}, unexpected={unexpected}, "
            f"selected_order={selected_block_names}"
        )

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "package_version": provenance["software"]["package_source_version"],
        "status": "running",
        "config_hash": canonical_config_hash(cfg),
        "protocol_hash": protocol_config_hash(cfg),
        "runtime_identity_sha256": _runtime_identity(bundle.metadata, data_metadata),
        "runtime_environment_sha256": provenance["runtime_environment_sha256"],
        "runtime_provenance": provenance,
        "selected_block_names": selected_block_names,
        "selected_block_names_sha256": _name_list_digest(selected_block_names),
        "expected_exact_block_names": expected_exact_names,
        "expected_exact_block_names_sha256": _name_list_digest(expected_exact_names),
        "control_block_names": [
            str(value)
            for value in analysis.get("controls", {}).get("block_names", [])
        ],
        "scientific_run": bool(cfg.get("scientific_run", False)),
        "synthetic_backend": bundle.metadata.get("backend") == "tiny_causal_lm",
        "experiment_tier": str(analysis.get("compute_tier", "debug")),
        "model": bundle.metadata,
        "data": data_metadata,
        "blocks": [block_metadata(block) for block in blocks],
        "streams": {
            "covariance": {
                "start_batch": int(analysis["covariance"].get("skip_batches", 0)),
                "num_batches": int(analysis["covariance"].get("num_batches", 16)),
            },
            "curvature": {
                "start_batch": int(curvature_cfg.get("skip_batches", 0)),
                "num_batches": int(curvature_cfg.get("num_batches", 1)),
            },
        },
    }
    json_dump(manifest, output_dir / "run_manifest.json")

    block_rows: list[dict] = []
    bootstrap_rows: list[dict] = []
    intervention_rows: list[dict] = []
    alpha_rows: list[dict] = []
    damping_rows: list[dict] = []
    tau_rows: list[dict] = []
    ridge_rows: list[dict] = []
    shift_rows: list[dict] = []
    failures: list[dict] = []

    def analyze_one(block: MatrixBlock):
        return _analyze_block(
            cfg=cfg,
            model=bundle.model,
            loss_fn=bundle.loss_fn,
            ggn_spec_fn=bundle.ggn_spec_fn,
            factory=factory,
            curvature_batches=curvature_batches,
            block=block,
            device=device,
            model_metadata=bundle.metadata,
            data_metadata=data_metadata,
            curvature_shift_overrides=curvature_shift_overrides,
            curvature_shift_override_sha256=curvature_shift_override_sha256,
        )

    for index, (block, result, failure) in enumerate(
        iter_block_analyses(blocks, analyze_one), start=1
    ):
        print(f"[{index}/{len(blocks)}] {block.name} shape={block.shape}", flush=True)
        if failure is not None:
            failures.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "config_hash": canonical_config_hash(cfg),
                    "protocol_hash": protocol_config_hash(cfg),
                    "seed": seed,
                    **failure,
                }
            )
        else:
            rows, boot, intervention, alpha, damping, tau, ridge = result
            block_rows.extend(rows)
            bootstrap_rows.extend(boot)
            intervention_rows.extend(intervention)
            alpha_rows.extend(alpha)
            damping_rows.extend(damping)
            tau_rows.extend(tau)
            ridge_rows.extend(ridge)
            if rows:
                first = rows[0]
                shift_rows.append(
                    {
                        "schema_version": first.get("schema_version", SCHEMA_VERSION),
                        "config_hash": first.get("config_hash", canonical_config_hash(cfg)),
                        "protocol_hash": first.get("protocol_hash", protocol_config_hash(cfg)),
                        "seed": first.get("seed", seed),
                        "block_name": first["block_name"],
                        "block_type": first.get("block_type", "unknown"),
                        "curvature_shift": first.get("curvature_shift"),
                        "curvature_shift_source": first.get("curvature_shift_source"),
                        "curvature_shift_override_sha256": first.get(
                            "curvature_shift_override_sha256", ""
                        ),
                        "curvature_target_ridge": first.get("curvature_target_ridge"),
                        "curvature_raw_min_ritz": first.get("curvature_raw_min_ritz"),
                        "curvature_raw_max_ritz": first.get("curvature_raw_max_ritz"),
                    }
                )

        _write_csv(output_dir / "block_metrics.csv", block_rows, BLOCK_METRIC_COLUMNS)
        _write_csv(
            output_dir / "bootstrap_metrics.csv", bootstrap_rows, BOOTSTRAP_COLUMNS
        )
        _write_csv(
            output_dir / "interventions.csv", intervention_rows, CONTROL_COLUMNS
        )
        _write_csv(output_dir / "alpha_sweep.csv", alpha_rows, CONTROL_COLUMNS)
        _write_csv(output_dir / "damping_sweep.csv", damping_rows, CONTROL_COLUMNS)
        _write_csv(
            output_dir / "ridge_sweep.csv",
            ridge_rows,
            RIDGE_SENSITIVITY_COLUMNS,
        )
        _write_csv(output_dir / "spectral_gain_curve.csv", tau_rows)
        _write_csv(output_dir / "curvature_shift_records.csv", shift_rows)
        _write_csv(
            output_dir / "block_failures.csv", failures, BLOCK_FAILURE_COLUMNS
        )
        if device.type == "cuda":
            torch.cuda.empty_cache()

    centered = [
        row for row in block_rows if row.get("covariance_moment") == "centered"
    ]
    summary = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "config_hash": canonical_config_hash(cfg),
        "protocol_hash": protocol_config_hash(cfg),
        "runtime_identity_sha256": manifest["runtime_identity_sha256"],
        "runtime_environment_sha256": manifest["runtime_environment_sha256"],
        "num_selected_blocks": len(blocks),
        "num_control_blocks_selected": sum(
            _controls_enabled_for_block(analysis, block.name) for block in blocks
        ),
        "num_ridge_sensitivity_rows": len(ridge_rows),
        "num_successful_blocks": len({row["block_name"] for row in block_rows}),
        "num_failed_blocks": len(failures),
        "num_centered_positive": sum(float(row["delta_g"]) > 0 for row in centered),
        "num_centered_negative": sum(float(row["delta_g"]) < 0 for row in centered),
        "num_reliable_orderings": sum(
            bool(row["ordering_inferentially_reliable"]) for row in centered
        ),
        "primary_endpoint": "centered observed alpha=0.25 delta_g",
    }
    json_dump(summary, output_dir / "summary.json")
    manifest.update(
        {
            "status": "complete",
            "num_successful_blocks": summary["num_successful_blocks"],
            "num_failed_blocks": summary["num_failed_blocks"],
            "required_outputs": [
                "block_metrics.csv",
                "bootstrap_metrics.csv",
                "interventions.csv",
                "alpha_sweep.csv",
                "damping_sweep.csv",
                "ridge_sweep.csv",
                "spectral_gain_curve.csv",
                "curvature_shift_records.csv",
                "partial_trace_artifacts/index.json",
                "block_failures.csv",
                "summary.json",
                "resolved_config.yaml",
                "runtime_provenance.json",
            ],
        }
    )
    json_dump(manifest, output_dir / "run_manifest.json")
    return output_dir
