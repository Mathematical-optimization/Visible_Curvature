"""Balanced reliability orchestration for frozen visible-curvature experiments.

This module deliberately leaves the scientific core estimator untouched.  It
runs a nested sequence of diagnostic budgets, certifies endpoint and
partial-trace convergence, and only then executes the expensive bootstrap and
mechanism controls.  The resulting certificates are conservative and remain
separate from the original output tables.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import yaml

from .partial_trace_stability import (
    PartialTraceStabilityThresholds,
    compare_partial_trace_artifacts,
)
from .run_lock import exclusive_output_lock
from .runtime_bootstrap import (
    RuntimeEnvironmentError,
    runtime_env_from_policy,
)


class ReliabilityError(RuntimeError):
    """Raised when a balanced reliability run cannot be certified or executed."""


@dataclass(frozen=True)
class StageSpec:
    index: int
    endpoint_steps: int
    endpoint_starts: int
    partial_trace_probes: int
    label: str
    refinement: bool = False


@dataclass(frozen=True)
class ReliabilityThresholds:
    minimum_stage_count: int = 2
    k_relative_change_tolerance: float = 0.05
    delta_g_absolute_change_tolerance: float = 0.05
    maximum_partial_trace_negative_mass: float = 0.05
    negative_mass_change_tolerance: float = 0.02
    partial_trace_matrix_relative_tolerance: float = 0.10
    partial_trace_subspace_projector_tolerance: float = 0.10
    intervention_factor_relative_tolerance: float = 0.10
    partial_trace_cluster_relative_gap: float = 1.0e-4
    require_partial_trace_artifacts: bool = False
    bootstrap_minimum_finite_reps: int = 100
    zero_tolerance: float = 1e-10


def _normalise_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())


def _find_column(frame: pd.DataFrame, *aliases: str, required: bool = False) -> str | None:
    by_norm = {_normalise_name(c): c for c in frame.columns}
    for alias in aliases:
        if alias in frame.columns:
            return alias
        match = by_norm.get(_normalise_name(alias))
        if match is not None:
            return match
    if required:
        raise ReliabilityError(f"missing required column; tried {aliases}; available={list(frame.columns)}")
    return None


def _as_bool(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y", "pass", "passed"}


def _finite_float(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return result if math.isfinite(result) else float("nan")


def _relative_change(current: float, previous: float, floor: float = 1e-30) -> float:
    if not (math.isfinite(current) and math.isfinite(previous)):
        return float("inf")
    return abs(current - previous) / max(abs(previous), floor)


def _deep_get(mapping: Mapping[str, Any], path: Sequence[str], default: Any = None) -> Any:
    node: Any = mapping
    for key in path:
        if not isinstance(node, Mapping) or key not in node:
            return default
        node = node[key]
    return node


def _deep_set(mapping: dict[str, Any], path: Sequence[str], value: Any, *, require_existing: bool = False) -> bool:
    node: dict[str, Any] = mapping
    for key in path[:-1]:
        child = node.get(key)
        if child is None:
            if require_existing:
                return False
            child = {}
            node[key] = child
        if not isinstance(child, dict):
            if require_existing:
                return False
            raise ReliabilityError(f"cannot set {'.'.join(path)} because {key} is not a mapping")
        node = child
    if require_existing and path[-1] not in node:
        return False
    node[path[-1]] = value
    return True


def _recursive_merge(base: dict[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), dict):
            _recursive_merge(base[key], value)
        else:
            base[key] = copy.deepcopy(value)
    return base


def load_yaml(path: str | Path) -> dict[str, Any]:
    result = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(result, dict):
        raise ReliabilityError(f"expected a YAML mapping in {path}")
    return result


def write_yaml(path: str | Path, value: Mapping[str, Any]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(yaml.safe_dump(dict(value), sort_keys=False, allow_unicode=True), encoding="utf-8")


def build_stage_specs(policy: Mapping[str, Any]) -> list[StageSpec]:
    rel = policy.get("reliability", {})
    endpoint_steps = [int(v) for v in rel.get("endpoint_steps", [64, 96, 128, 192, 256])]
    endpoint_starts = [int(v) for v in rel.get("endpoint_starts", [2, 2, 2, 3, 4])]
    probes = [int(v) for v in rel.get("partial_trace_probes", [32, 64, 128, 256])]
    if not endpoint_steps or not probes:
        raise ReliabilityError("endpoint_steps and partial_trace_probes must be non-empty")
    if sorted(set(endpoint_steps)) != endpoint_steps or any(v <= 0 for v in endpoint_steps):
        raise ReliabilityError("endpoint_steps must be strictly increasing positive integers")
    if sorted(set(probes)) != probes or any(v <= 0 for v in probes):
        raise ReliabilityError("partial_trace_probes must be strictly increasing positive integers")
    if len(endpoint_starts) == 1:
        endpoint_starts = endpoint_starts * len(endpoint_steps)
    if len(endpoint_starts) != len(endpoint_steps):
        raise ReliabilityError("endpoint_starts must have length 1 or match endpoint_steps")
    n = max(len(endpoint_steps), len(probes))
    specs: list[StageSpec] = []
    for idx in range(n):
        step = endpoint_steps[min(idx, len(endpoint_steps) - 1)]
        start = endpoint_starts[min(idx, len(endpoint_starts) - 1)]
        probe = probes[min(idx, len(probes) - 1)]
        specs.append(StageSpec(idx, step, start, probe, f"adaptive_{step:04d}_{probe:04d}"))
    return specs


def thresholds_from_policy(policy: Mapping[str, Any]) -> ReliabilityThresholds:
    rel = policy.get("reliability", {})
    return ReliabilityThresholds(
        minimum_stage_count=int(rel.get("minimum_stage_count", 2)),
        k_relative_change_tolerance=float(rel.get("k_relative_change_tolerance", 0.05)),
        delta_g_absolute_change_tolerance=float(rel.get("delta_g_absolute_change_tolerance", 0.05)),
        maximum_partial_trace_negative_mass=float(rel.get("maximum_partial_trace_negative_mass", 0.05)),
        negative_mass_change_tolerance=float(rel.get("negative_mass_change_tolerance", 0.02)),
        partial_trace_matrix_relative_tolerance=float(
            rel.get("partial_trace_matrix_relative_tolerance", 0.10)
        ),
        partial_trace_subspace_projector_tolerance=float(
            rel.get("partial_trace_subspace_projector_tolerance", 0.10)
        ),
        intervention_factor_relative_tolerance=float(
            rel.get("intervention_factor_relative_tolerance", 0.10)
        ),
        partial_trace_cluster_relative_gap=float(
            rel.get("partial_trace_cluster_relative_gap", 1.0e-4)
        ),
        require_partial_trace_artifacts=bool(
            rel.get("require_partial_trace_artifacts", True)
        ),
        bootstrap_minimum_finite_reps=int(rel.get("bootstrap_minimum_finite_reps", 100)),
        zero_tolerance=float(rel.get("zero_tolerance", 1e-10)),
    )


def _set_if_present(config: dict[str, Any], candidate_paths: Iterable[Sequence[str]], value: Any) -> str | None:
    for path in candidate_paths:
        if _deep_set(config, path, value, require_existing=True):
            return ".".join(path)
    return None


def prepare_core_config(
    base_config: Mapping[str, Any],
    *,
    stage: StageSpec,
    output_dir: Path,
    diagnostic: bool,
    policy: Mapping[str, Any],
    shift_overrides_path: str | Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    cfg = copy.deepcopy(dict(base_config))
    overrides = policy.get("base_overrides", {})
    if overrides:
        if not isinstance(overrides, Mapping):
            raise ReliabilityError("base_overrides must be a mapping")
        _recursive_merge(cfg, overrides)
    cfg["output_dir"] = str(output_dir)
    cfg["experiment_label"] = f"{cfg.get('experiment_label', 'visible-curvature')}-{stage.label}"

    modified: dict[str, Any] = {}
    for value, paths, name in [
        (stage.endpoint_steps, [("analysis", "curvature", "lanczos_steps")], "lanczos_steps"),
        (stage.endpoint_starts, [("analysis", "curvature", "lanczos_starts")], "lanczos_starts"),
        (stage.partial_trace_probes, [("analysis", "curvature", "partial_trace_probes")], "partial_trace_probes"),
    ]:
        path = _set_if_present(cfg, paths, value)
        if path is None:
            raise ReliabilityError(f"base config does not expose required field {name}")
        modified[path] = value

    stabilise_steps = max(8, min(stage.endpoint_steps, stage.endpoint_steps // 2))
    path = _set_if_present(
        cfg,
        [("analysis", "curvature", "stabilize_lanczos_steps"), ("analysis", "curvature", "stabilise_lanczos_steps")],
        stabilise_steps,
    )
    if path is not None:
        modified[path] = stabilise_steps
    path = _set_if_present(
        cfg,
        [("analysis", "curvature", "stabilize_lanczos_starts"), ("analysis", "curvature", "stabilise_lanczos_starts")],
        max(1, min(stage.endpoint_starts, 2)),
    )
    if path is not None:
        modified[path] = max(1, min(stage.endpoint_starts, 2))

    fixed_ridge = policy.get("reliability", {}).get("fixed_relative_ridge")
    if fixed_ridge is not None:
        ridge_path = _set_if_present(
            cfg,
            [("analysis", "curvature", "ridge")],
            float(fixed_ridge),
        )
        if ridge_path is None:
            raise ReliabilityError("base config does not expose analysis.curvature.ridge")
        modified[ridge_path] = float(fixed_ridge)
        ridge_mode_path = _set_if_present(
            cfg,
            [("analysis", "curvature", "ridge_mode")],
            "relative_max",
        )
        if ridge_mode_path is None:
            raise ReliabilityError("base config does not expose analysis.curvature.ridge_mode")
        modified[ridge_mode_path] = "relative_max"

    if shift_overrides_path is not None:
        resolved_overrides = str(Path(shift_overrides_path).resolve())
        _deep_set(
            cfg,
            ("analysis", "curvature", "shift_overrides_path"),
            resolved_overrides,
        )
        modified["analysis.curvature.shift_overrides_path"] = resolved_overrides

    if diagnostic:
        cfg["scientific_run"] = False
        _deep_set(cfg, ("analysis", "compute_tier"), "debug")
        _set_if_present(cfg, [("analysis", "bootstrap", "reps")], 0)
        _set_if_present(cfg, [("analysis", "bootstrap", "minimum_reps")], 0)
        _set_if_present(cfg, [("analysis", "interventions", "enabled")], False)
        _set_if_present(cfg, [("analysis", "alpha_sweep", "enabled")], False)
        _set_if_present(cfg, [("analysis", "damping_sweep", "enabled")], False)
    else:
        final_scientific = bool(policy.get("final_scientific_run", True))
        cfg["scientific_run"] = final_scientific
        _deep_set(cfg, ("analysis", "compute_tier"), "confirmatory" if final_scientific else "debug")
        final_reps = int(policy.get("reliability", {}).get("final_bootstrap_reps", 100))
        _set_if_present(cfg, [("analysis", "bootstrap", "reps")], final_reps)
        _set_if_present(cfg, [("analysis", "bootstrap", "minimum_reps")], final_reps)
        _set_if_present(cfg, [("analysis", "interventions", "enabled")], True)
        _set_if_present(cfg, [("analysis", "alpha_sweep", "enabled")], True)
        _set_if_present(cfg, [("analysis", "damping_sweep", "enabled")], True)
    return cfg, modified


def _primary_rows(frame: pd.DataFrame, policy: Mapping[str, Any]) -> pd.DataFrame:
    result = frame.copy()
    primary = policy.get("primary", {})
    moment = str(primary.get("covariance_moment", "centered"))
    assignment = str(primary.get("assignment", "observed"))
    alpha = float(primary.get("alpha", 0.25))
    sweep_mode = str(primary.get("sweep_mode", "primary"))
    moment_col = _find_column(result, "covariance_moment")
    assignment_col = _find_column(result, "assignment")
    alpha_col = _find_column(result, "alpha")
    sweep_mode_col = _find_column(result, "sweep_mode")
    if moment_col is not None:
        result = result[result[moment_col].astype(str) == moment]
    if assignment_col is not None:
        result = result[result[assignment_col].astype(str) == assignment]
    if alpha_col is not None:
        numeric = pd.to_numeric(result[alpha_col], errors="coerce")
        result = result[np.isclose(numeric, alpha)]
    if sweep_mode_col is not None:
        result = result[result[sweep_mode_col].astype(str) == sweep_mode]
    return result.copy()


def summarise_stage(output_dir: Path, stage: StageSpec, policy: Mapping[str, Any]) -> pd.DataFrame:
    metrics_path = output_dir / "block_metrics.csv"
    if not metrics_path.exists():
        raise ReliabilityError(f"missing {metrics_path}")
    frame = _primary_rows(pd.read_csv(metrics_path), policy)
    block_col = _find_column(frame, "block_name", required=True)
    aliases = {
        "K_adam": ("K_adam", "k_adam"),
        "K_shampoo": ("K_shampoo", "k_shampoo"),
        "delta_g": ("delta_g", "DeltaG"),
        "native_endpoint_reliable": ("endpoint_numerically_reliable",),
        "min_ritz_residual_adam": ("min_ritz_residual_Adam", "adam_min_ritz_residual"),
        "min_ritz_residual_shampoo": ("min_ritz_residual_Shampoo", "shampoo_min_ritz_residual"),
        "max_ritz_residual_adam": ("max_ritz_residual_Adam", "adam_max_ritz_residual"),
        "max_ritz_residual_shampoo": ("max_ritz_residual_Shampoo", "shampoo_max_ritz_residual"),
        "negative_mass_left": ("partial_trace_negative_spectral_mass_left",),
        "negative_mass_right": ("partial_trace_negative_spectral_mass_right",),
        "condition_metric": ("condition_metric",),
        "adam_condition_saturated": ("adam_condition_saturated",),
        "shampoo_condition_saturated": ("shampoo_condition_saturated",),
        "partial_trace_artifact": ("partial_trace_artifact",),
    }
    rows: list[dict[str, Any]] = []
    for _, row in frame.iterrows():
        item: dict[str, Any] = {
            "block_name": str(row[block_col]),
            "stage_index": stage.index,
            "stage_label": stage.label,
            "refinement": stage.refinement,
            "endpoint_steps": stage.endpoint_steps,
            "endpoint_starts": stage.endpoint_starts,
            "partial_trace_probes": stage.partial_trace_probes,
            "output_dir": str(output_dir),
        }
        for output_name, candidates in aliases.items():
            column = _find_column(frame, *candidates)
            item[output_name] = row[column] if column is not None else np.nan
        rows.append(item)
    if not rows:
        raise ReliabilityError(f"no primary rows found in {metrics_path}")
    return pd.DataFrame(rows)


def certify_convergence(stage_rows: pd.DataFrame, thresholds: ReliabilityThresholds) -> pd.DataFrame:
    required = {"block_name", "stage_index", "K_adam", "K_shampoo", "delta_g"}
    missing = sorted(required.difference(stage_rows.columns))
    if missing:
        raise ReliabilityError(f"stage table missing columns: {missing}")
    certificates: list[dict[str, Any]] = []
    for block_name, group in stage_rows.groupby("block_name", sort=True):
        group = group.sort_values("stage_index").reset_index(drop=True)
        current = group.iloc[-1]
        previous = group.iloc[-2] if len(group) >= 2 else None
        k_adam_change = float("inf")
        k_shampoo_change = float("inf")
        delta_change = float("inf")
        neg_left_change = float("inf")
        neg_right_change = float("inf")
        if previous is not None:
            k_adam_change = _relative_change(_finite_float(current["K_adam"]), _finite_float(previous["K_adam"]))
            k_shampoo_change = _relative_change(_finite_float(current["K_shampoo"]), _finite_float(previous["K_shampoo"]))
            delta_change = abs(_finite_float(current["delta_g"]) - _finite_float(previous["delta_g"]))
            neg_left_change = abs(_finite_float(current.get("negative_mass_left")) - _finite_float(previous.get("negative_mass_left")))
            neg_right_change = abs(_finite_float(current.get("negative_mass_right")) - _finite_float(previous.get("negative_mass_right")))
        native_endpoint = _as_bool(current.get("native_endpoint_reliable"))
        enough_stages = len(group) >= thresholds.minimum_stage_count
        endpoint_stable = (
            enough_stages
            and k_adam_change <= thresholds.k_relative_change_tolerance
            and k_shampoo_change <= thresholds.k_relative_change_tolerance
            and delta_change <= thresholds.delta_g_absolute_change_tolerance
        )
        endpoint_certified = native_endpoint and endpoint_stable
        neg_left = _finite_float(current.get("negative_mass_left"))
        neg_right = _finite_float(current.get("negative_mass_right"))
        current_artifact_value = current.get("partial_trace_artifact")
        previous_artifact_value = (
            previous.get("partial_trace_artifact") if previous is not None else None
        )
        current_artifact = (
            Path(str(current_artifact_value))
            if current_artifact_value is not None
            and str(current_artifact_value).strip()
            and str(current_artifact_value).lower() != "nan"
            else None
        )
        previous_artifact = (
            Path(str(previous_artifact_value))
            if previous_artifact_value is not None
            and str(previous_artifact_value).strip()
            and str(previous_artifact_value).lower() != "nan"
            else None
        )
        artifacts_available = bool(
            current_artifact is not None
            and previous_artifact is not None
            and current_artifact.exists()
            and previous_artifact.exists()
        )
        geometry: dict[str, Any] = {}
        if artifacts_available:
            geometry = compare_partial_trace_artifacts(
                previous_artifact,
                current_artifact,
                PartialTraceStabilityThresholds(
                    matrix_relative_tolerance=thresholds.partial_trace_matrix_relative_tolerance,
                    subspace_projector_tolerance=thresholds.partial_trace_subspace_projector_tolerance,
                    intervention_factor_relative_tolerance=thresholds.intervention_factor_relative_tolerance,
                    cluster_relative_gap=thresholds.partial_trace_cluster_relative_gap,
                    maximum_negative_mass=thresholds.maximum_partial_trace_negative_mass,
                    negative_mass_change_tolerance=thresholds.negative_mass_change_tolerance,
                ),
            )
            neg_left = _finite_float(geometry.get("negative_mass_left"))
            neg_right = _finite_float(geometry.get("negative_mass_right"))
            partial_trace_level_ok = _as_bool(
                geometry.get("partial_trace_psd_checks_passed")
            )
            partial_trace_matrix_stable = _as_bool(
                geometry.get("partial_trace_matrix_stable")
            )
            partial_trace_subspace_stable = _as_bool(
                geometry.get("partial_trace_subspace_stable")
            )
            intervention_factors_stable = _as_bool(
                geometry.get("intervention_factors_stable")
            )
            partial_trace_stable = bool(
                enough_stages
                and partial_trace_matrix_stable
                and partial_trace_subspace_stable
                and intervention_factors_stable
            )
            partial_trace_certified = bool(
                enough_stages
                and geometry.get("partial_trace_checks_passed", False)
            )
        else:
            partial_trace_level_ok = (
                math.isfinite(neg_left)
                and math.isfinite(neg_right)
                and neg_left <= thresholds.maximum_partial_trace_negative_mass
                and neg_right <= thresholds.maximum_partial_trace_negative_mass
            )
            partial_trace_matrix_stable = False
            partial_trace_subspace_stable = False
            intervention_factors_stable = False
            legacy_probe_stable = (
                enough_stages
                and neg_left_change <= thresholds.negative_mass_change_tolerance
                and neg_right_change <= thresholds.negative_mass_change_tolerance
            )
            partial_trace_stable = bool(
                legacy_probe_stable and not thresholds.require_partial_trace_artifacts
            )
            partial_trace_certified = bool(
                partial_trace_level_ok and partial_trace_stable
            )
        endpoint_reasons: list[str] = []
        if not native_endpoint:
            endpoint_reasons.append("native_endpoint_check")
        if not enough_stages:
            endpoint_reasons.append("insufficient_stages")
        if k_adam_change > thresholds.k_relative_change_tolerance:
            endpoint_reasons.append("K_adam_not_stable")
        if k_shampoo_change > thresholds.k_relative_change_tolerance:
            endpoint_reasons.append("K_shampoo_not_stable")
        if delta_change > thresholds.delta_g_absolute_change_tolerance:
            endpoint_reasons.append("delta_g_not_stable")
        partial_reasons: list[str] = []
        if thresholds.require_partial_trace_artifacts and not artifacts_available:
            partial_reasons.append("artifacts_unavailable")
        if not partial_trace_level_ok:
            partial_reasons.append("negative_mass")
        if not partial_trace_stable:
            partial_reasons.append("probe_convergence")
        geometry_reason = str(geometry.get("partial_trace_stability_reasons", ""))
        if geometry_reason:
            partial_reasons.extend(
                reason
                for reason in geometry_reason.split(",")
                if reason and reason not in partial_reasons
            )
        certificates.append({
            "block_name": block_name,
            "stage_count": len(group),
            "selected_stage_label": str(current["stage_label"]),
            "selected_endpoint_steps": int(current["endpoint_steps"]),
            "selected_endpoint_starts": int(current["endpoint_starts"]),
            "selected_partial_trace_probes": int(current["partial_trace_probes"]),
            "selected_K_adam": _finite_float(current["K_adam"]),
            "selected_K_shampoo": _finite_float(current["K_shampoo"]),
            "selected_delta_g": _finite_float(current["delta_g"]),
            "K_adam_relative_change": k_adam_change,
            "K_shampoo_relative_change": k_shampoo_change,
            "delta_g_absolute_change": delta_change,
            "negative_mass_left": neg_left,
            "negative_mass_right": neg_right,
            "negative_mass_left_change": neg_left_change,
            "negative_mass_right_change": neg_right_change,
            "native_endpoint_reliable": native_endpoint,
            "endpoint_stable": endpoint_stable,
            "adaptive_endpoint_certified": endpoint_certified,
            "endpoint_certification_reasons": ",".join(endpoint_reasons),
            "partial_trace_level_ok": partial_trace_level_ok,
            "partial_trace_stable": partial_trace_stable,
            "partial_trace_artifacts_available": artifacts_available,
            "partial_trace_matrix_stable": partial_trace_matrix_stable,
            "partial_trace_subspace_stable": partial_trace_subspace_stable,
            "intervention_factors_stable": intervention_factors_stable,
            "partial_trace_checks_passed": partial_trace_certified,
            "max_matrix_relative_change": _finite_float(
                geometry.get("max_matrix_relative_change")
            ),
            "max_subspace_projector_distance": _finite_float(
                geometry.get("max_subspace_projector_distance")
            ),
            "max_intervention_factor_relative_change": _finite_float(
                geometry.get("max_intervention_factor_relative_change")
            ),
            "adaptive_partial_trace_certified": partial_trace_certified,
            "partial_trace_certification_reasons": ",".join(partial_reasons),
        })
    return pd.DataFrame(certificates)


def tau_sign_reason(values: Iterable[float], zero_tolerance: float = 1e-10) -> str:
    finite = [float(v) for v in values if math.isfinite(float(v))]
    if not finite:
        return "missing"
    signs = set()
    zero_seen = False
    for value in finite:
        if abs(value) <= zero_tolerance:
            zero_seen = True
        elif value > 0:
            signs.add(1)
        else:
            signs.add(-1)
    if len(signs) >= 2:
        return "sign_flip"
    if not signs:
        return "all_saturated"
    if zero_seen:
        return "one_sided_with_coarse_saturation"
    return "stable_nonzero"


def find_tau_table(output_dir: Path, policy: Mapping[str, Any]) -> tuple[pd.DataFrame | None, pd.DataFrame]:
    preferred = output_dir / "spectral_gain_curve.csv"
    candidates = ([preferred] if preferred.exists() else []) + sorted(
        p
        for p in output_dir.glob("*.csv")
        if "tau" in p.name.lower() and p != preferred
    )
    if not candidates:
        return None, pd.DataFrame(columns=["block_name", "tau_sign_reason"])
    for path in candidates:
        frame = pd.read_csv(path)
        block_col = _find_column(frame, "block_name")
        tau_col = _find_column(frame, "tau", "tau_value", "condition_tau")
        delta_col = _find_column(frame, "delta_g", "DeltaG")
        if block_col and tau_col and delta_col:
            primary = _primary_rows(frame, policy)
            reasons = []
            zero_tol = float(policy.get("reliability", {}).get("zero_tolerance", 1e-10))
            for block_name, group in primary.groupby(block_col):
                reasons.append({
                    "block_name": str(block_name),
                    "tau_sign_reason": tau_sign_reason(pd.to_numeric(group[delta_col], errors="coerce"), zero_tol),
                })
            return primary, pd.DataFrame(reasons)
    return None, pd.DataFrame(columns=["block_name", "tau_sign_reason"])


def _hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_csv_atomic(frame: pd.DataFrame, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(tmp, index=False)
    tmp.replace(path)


def extract_shift_overrides(output_dir: Path, destination: Path) -> dict[str, float]:
    """Promote one consistent estimated shift per block from a calibration run."""
    metrics_path = output_dir / "block_metrics.csv"
    if not metrics_path.exists():
        raise ReliabilityError(f"missing calibration metrics: {metrics_path}")
    frame = pd.read_csv(metrics_path)
    block_col = _find_column(frame, "block_name", required=True)
    shift_col = _find_column(frame, "curvature_shift", required=True)
    overrides: dict[str, float] = {}
    for block_name, group in frame.groupby(block_col, sort=True):
        values = pd.to_numeric(group[shift_col], errors="coerce").dropna().unique()
        if len(values) != 1:
            raise ReliabilityError(
                f"calibration produced inconsistent shifts for {block_name!r}: {values.tolist()}"
            )
        shift = float(values[0])
        if not math.isfinite(shift) or shift < 0.0:
            raise ReliabilityError(f"invalid calibrated shift for {block_name!r}: {shift}")
        overrides[str(block_name)] = shift
    if not overrides:
        raise ReliabilityError("calibration produced no curvature shifts")
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "1.0",
        "source_output_dir": str(output_dir.resolve()),
        "blocks": overrides,
    }
    destination.write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    return overrides


def annotate_final_outputs(
    final_dir: Path,
    stage_rows: pd.DataFrame,
    certificates: pd.DataFrame,
    policy: Mapping[str, Any],
) -> dict[str, Any]:
    """Promote only final rows that pass every balanced numerical check."""
    _write_csv_atomic(stage_rows, final_dir / "endpoint_convergence.csv")
    partial_columns = [
        "block_name",
        "stage_index",
        "stage_label",
        "partial_trace_probes",
        "negative_mass_left",
        "negative_mass_right",
        "partial_trace_artifact",
        "output_dir",
    ]
    _write_csv_atomic(
        stage_rows[[c for c in partial_columns if c in stage_rows.columns]],
        final_dir / "partial_trace_convergence.csv",
    )
    _write_csv_atomic(
        certificates, final_dir / "balanced_reliability_certificates.csv"
    )

    tau_table, tau_reasons = find_tau_table(final_dir, policy)
    if tau_table is not None:
        _write_csv_atomic(
            tau_table, final_dir / "canonical_spectral_gain_curve.csv"
        )

    metrics_path = final_dir / "block_metrics.csv"
    if not metrics_path.exists():
        raise ReliabilityError(f"missing {metrics_path}")
    metrics = pd.read_csv(metrics_path)
    block_col = _find_column(metrics, "block_name", required=True)
    annotated = metrics.merge(
        certificates,
        left_on=block_col,
        right_on="block_name",
        how="left",
        suffixes=("", "_certificate"),
    )
    if not tau_reasons.empty:
        annotated = annotated.merge(
            tau_reasons,
            left_on=block_col,
            right_on="block_name",
            how="left",
            suffixes=("", "_tau"),
        )
    else:
        annotated["tau_sign_reason"] = "tau_refinement_unavailable"

    delta_col = _find_column(annotated, "delta_g", required=True)
    k_adam_col = _find_column(annotated, "K_adam", required=True)
    k_shampoo_col = _find_column(annotated, "K_shampoo", required=True)
    ci_low_col = _find_column(annotated, "bootstrap_ci_low")
    ci_high_col = _find_column(annotated, "bootstrap_ci_high")
    finite_reps_col = _find_column(annotated, "bootstrap_reps_finite")
    moment_col = _find_column(annotated, "covariance_moment")
    assignment_col = _find_column(annotated, "assignment")
    alpha_col = _find_column(annotated, "alpha")
    final_endpoint_col = _find_column(
        annotated, "endpoint_numerically_reliable"
    )
    threshold = thresholds_from_policy(policy)
    primary_policy = policy.get("primary", {})

    labels: list[str] = []
    reliable_flags: list[bool] = []
    tail_localized: list[bool] = []
    reasons_out: list[str] = []
    final_agreements: list[bool] = []
    final_k_adam_changes: list[float] = []
    final_k_shampoo_changes: list[float] = []
    final_delta_changes: list[float] = []
    final_endpoint_flags: list[bool] = []

    for _, row in annotated.iterrows():
        primary = True
        if moment_col is not None:
            primary &= str(row[moment_col]) == str(
                primary_policy.get("covariance_moment", "centered")
            )
        if assignment_col is not None:
            primary &= str(row[assignment_col]) == str(
                primary_policy.get("assignment", "observed")
            )
        if alpha_col is not None:
            primary &= math.isclose(
                _finite_float(row[alpha_col]),
                float(primary_policy.get("alpha", 0.25)),
                rel_tol=0.0,
                abs_tol=1e-12,
            )

        reasons: list[str] = []
        delta = _finite_float(row[delta_col])
        endpoint_ok = _as_bool(row.get("adaptive_endpoint_certified"))
        if not endpoint_ok:
            reasons.append("adaptive_endpoint")

        final_endpoint_ok = (
            _as_bool(row[final_endpoint_col])
            if final_endpoint_col is not None
            else False
        )
        final_endpoint_flags.append(final_endpoint_ok)
        if not final_endpoint_ok:
            reasons.append("final_endpoint")

        final_k_adam_change = _relative_change(
            _finite_float(row[k_adam_col]),
            _finite_float(row.get("selected_K_adam")),
        )
        final_k_shampoo_change = _relative_change(
            _finite_float(row[k_shampoo_col]),
            _finite_float(row.get("selected_K_shampoo")),
        )
        final_delta_change = abs(
            delta - _finite_float(row.get("selected_delta_g"))
        )
        final_agreement = bool(
            final_k_adam_change <= threshold.k_relative_change_tolerance
            and final_k_shampoo_change
            <= threshold.k_relative_change_tolerance
            and final_delta_change
            <= threshold.delta_g_absolute_change_tolerance
        )
        final_agreements.append(final_agreement)
        final_k_adam_changes.append(final_k_adam_change)
        final_k_shampoo_changes.append(final_k_shampoo_change)
        final_delta_changes.append(final_delta_change)
        if not final_agreement:
            reasons.append("final_diagnostic_disagreement")

        reps_ok = False
        if finite_reps_col is not None:
            reps_ok = (
                _finite_float(row[finite_reps_col])
                >= threshold.bootstrap_minimum_finite_reps
            )
        if not reps_ok:
            reasons.append("bootstrap_reps")

        ci_sign = 0
        if ci_low_col is not None and ci_high_col is not None:
            low = _finite_float(row[ci_low_col])
            high = _finite_float(row[ci_high_col])
            if math.isfinite(low) and math.isfinite(high):
                if low > 0:
                    ci_sign = 1
                elif high < 0:
                    ci_sign = -1
        if ci_sign == 0:
            reasons.append("bootstrap_ci")
        point_sign = 1 if delta > 0 else (-1 if delta < 0 else 0)
        if ci_sign and point_sign != ci_sign:
            reasons.append("point_ci_disagreement")

        tau_reason = str(
            row.get("tau_sign_reason", "tau_refinement_unavailable")
        )
        tau_ok = tau_reason in {
            "stable_nonzero",
            "one_sided_with_coarse_saturation",
        }
        if not tau_ok:
            reasons.append("tau_" + tau_reason)
        is_tail = tau_reason == "one_sided_with_coarse_saturation"

        reliable = bool(
            primary
            and endpoint_ok
            and final_endpoint_ok
            and final_agreement
            and reps_ok
            and ci_sign != 0
            and point_sign == ci_sign
            and tau_ok
        )
        label = (
            "positive"
            if reliable and ci_sign > 0
            else ("negative" if reliable and ci_sign < 0 else "inconclusive")
        )
        labels.append(label)
        reliable_flags.append(reliable)
        tail_localized.append(is_tail)
        reasons_out.append(",".join(dict.fromkeys(reasons)))

    annotated["final_endpoint_numerically_accepted"] = final_endpoint_flags
    annotated["final_K_adam_relative_change"] = final_k_adam_changes
    annotated["final_K_shampoo_relative_change"] = final_k_shampoo_changes
    annotated["final_delta_g_absolute_change"] = final_delta_changes
    annotated["final_diagnostic_agreement"] = final_agreements
    annotated["balanced_reliable_ordering"] = labels
    annotated["balanced_primary_reliable"] = reliable_flags
    annotated["tail_localized_ordering"] = tail_localized
    annotated["balanced_reliability_reasons"] = reasons_out
    _write_csv_atomic(annotated, final_dir / "balanced_block_metrics.csv")

    canonical_primary = _primary_rows(annotated, policy)
    _write_csv_atomic(
        canonical_primary, final_dir / "canonical_block_metrics.csv"
    )

    canonical_control_names = {
        "interventions.csv": "canonical_interventions.csv",
        "alpha_sweep.csv": "canonical_alpha_sweep.csv",
        "damping_sweep.csv": "canonical_damping_sweep.csv",
    }
    for filename, canonical_name in canonical_control_names.items():
        path = final_dir / filename
        if not path.exists():
            continue
        table = pd.read_csv(path)
        table_block = _find_column(table, "block_name", required=True)
        table = table.merge(
            certificates,
            left_on=table_block,
            right_on="block_name",
            how="left",
            suffixes=("", "_certificate"),
        )
        assignment = _find_column(table, "assignment")
        endpoint_native = _find_column(
            table, "endpoint_numerically_reliable"
        )
        reliable: list[bool] = []
        for _, row in table.iterrows():
            assignment_value = (
                str(row[assignment]) if assignment is not None else "observed"
            )
            needs_basis = assignment_value in {"aligned", "reversed"}
            ok = _as_bool(row.get("adaptive_endpoint_certified"))
            if endpoint_native is not None:
                ok = ok and _as_bool(row[endpoint_native])
            if needs_basis:
                ok = ok and _as_bool(row.get("partial_trace_checks_passed"))
            reliable.append(bool(ok))
        table["balanced_reliable_for_inference"] = reliable
        _write_csv_atomic(table, final_dir / ("balanced_" + filename))
        _write_csv_atomic(table, final_dir / canonical_name)

    primary_available = bool(
        not canonical_primary.empty
        and canonical_primary["balanced_primary_reliable"].map(_as_bool).all()
    )
    scientific_status = "accepted" if primary_available else "inconclusive"
    status = {
        "pipeline_status": "complete",
        "primary_inference_available": primary_available,
        "scientific_status": scientific_status,
    }
    (final_dir / "scientific_status.json").write_text(
        json.dumps(status, indent=2, sort_keys=True), encoding="utf-8"
    )

    summary = {
        "schema_version": 2,
        "reliability_mode": "balanced_canonical",
        "pipeline_status": "complete",
        "scientific_status": scientific_status,
        "primary_inference_available": primary_available,
        "policy": dict(policy),
        "num_blocks": int(certificates["block_name"].nunique()),
        "num_endpoint_certified": int(
            certificates["adaptive_endpoint_certified"].map(_as_bool).sum()
        ),
        "num_partial_trace_certified": int(
            certificates["adaptive_partial_trace_certified"].map(_as_bool).sum()
        ),
        "num_balanced_primary_reliable": int(
            canonical_primary["balanced_primary_reliable"].map(_as_bool).sum()
        )
        if not canonical_primary.empty
        else 0,
        "selected_endpoint_steps": int(
            certificates["selected_endpoint_steps"].max()
        ),
        "selected_endpoint_starts": int(
            certificates["selected_endpoint_starts"].max()
        ),
        "selected_partial_trace_probes": int(
            certificates["selected_partial_trace_probes"].max()
        ),
        "files": {},
    }
    for path in sorted(final_dir.glob("*")):
        if path.is_file():
            summary["files"][path.name] = _hash_file(path)
    (final_dir / "balanced_reliability_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    return summary

def _progress(message: str) -> None:
    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    print(f"[balanced {timestamp} pid={os.getpid()}] {message}", flush=True)


def _run_command(
    command: Sequence[str],
    *,
    cwd: Path,
    log_path: Path,
    env: Mapping[str, str] | None = None,
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    merged_env = os.environ.copy()
    if env:
        merged_env.update({str(k): str(v) for k, v in env.items()})
    _progress(
        "starting child process; "
        f"log={log_path} command={' '.join(command)}"
    )
    with log_path.open("w", encoding="utf-8") as handle:
        process = subprocess.run(
            list(command),
            cwd=str(cwd),
            env=merged_env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
    if process.returncode != 0:
        tail = log_path.read_text(encoding="utf-8", errors="replace")[-6000:]
        raise ReliabilityError(
            f"command failed ({process.returncode}): {' '.join(command)}\n{tail}"
        )
    _progress(f"child process completed; log={log_path}")


def run_balanced_policy(policy_path: str | Path, *, project_root: str | Path | None = None) -> Path:
    policy_path = Path(policy_path).resolve()
    policy = load_yaml(policy_path)
    root = Path(project_root).resolve() if project_root else Path.cwd().resolve()
    base_path = Path(str(policy.get("base_config", "")))
    if not base_path.is_absolute():
        base_path = (root / base_path).resolve()
    if not base_path.exists():
        raise ReliabilityError(f"base_config does not exist: {base_path}")
    base_config = load_yaml(base_path)
    output_root = Path(str(policy.get("output_root", "outputs/balanced_reliability")))
    if not output_root.is_absolute():
        output_root = (root / output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    try:
        runtime_env = runtime_env_from_policy(policy)
    except RuntimeEnvironmentError as exc:
        raise ReliabilityError(str(exc)) from exc
    with exclusive_output_lock(output_root, policy_path=policy_path):
        visible_devices = runtime_env.get(
            "CUDA_VISIBLE_DEVICES",
            os.environ.get("CUDA_VISIBLE_DEVICES", "unset"),
        )
        _progress(
            f"acquired output lock for {output_root}; "
            f"CUDA_VISIBLE_DEVICES={visible_devices}"
        )
        generated_dir = output_root / "generated_configs"
        logs_dir = output_root / "logs"
        stages_dir = output_root / "diagnostic_stages"
        final_dir = output_root / "final"
        python_executable = str(policy.get("python_executable", sys.executable))
        runner = root / "scripts" / "run_frozen_analysis.py"
        validator = root / "scripts" / "validate_run.py"
        if not runner.exists():
            raise ReliabilityError(f"missing core runner: {runner}")

        specs = build_stage_specs(policy)
        thresholds = thresholds_from_policy(policy)
        stage_frames: list[pd.DataFrame] = []
        selected_spec = specs[-1]
        shift_overrides_path = output_root / "curvature_shift_overrides.json"
        for spec in specs:
            stage_output = stages_dir / spec.label
            stage_shift_path = shift_overrides_path if stage_frames else None
            config, modified = prepare_core_config(
                base_config,
                stage=spec,
                output_dir=stage_output,
                diagnostic=True,
                policy=policy,
                shift_overrides_path=stage_shift_path,
            )
            config_path = generated_dir / f"{spec.label}.yaml"
            write_yaml(config_path, config)
            (generated_dir / f"{spec.label}.changes.json").write_text(json.dumps(modified, indent=2, sort_keys=True), encoding="utf-8")
            if stage_output.exists() and not bool(policy.get("reuse_completed_stages", False)):
                shutil.rmtree(stage_output)
            if not stage_output.exists():
                _run_command([python_executable, str(runner), "--config", str(config_path)], cwd=root, log_path=logs_dir / f"{spec.label}.log", env=runtime_env)
            if validator.exists():
                _run_command([python_executable, str(validator), "--output-dir", str(stage_output)], cwd=root, log_path=logs_dir / f"{spec.label}.validate.log", env=runtime_env)
            if not stage_frames:
                extract_shift_overrides(stage_output, shift_overrides_path)
            stage_frame = summarise_stage(stage_output, spec, policy)
            stage_frames.append(stage_frame)
            combined = pd.concat(stage_frames, ignore_index=True)
            _progress(
                f"{spec.label}: starting CPU endpoint/partial-trace certification"
            )
            current_cert = certify_convergence(combined, thresholds)
            endpoint_done = bool(current_cert["adaptive_endpoint_certified"].map(_as_bool).all())
            partial_done = bool(current_cert["adaptive_partial_trace_certified"].map(_as_bool).all())
            selected_spec = spec
            _progress(
                f"{spec.label}: certification finished; "
                f"endpoint_all={endpoint_done} partial_trace_all={partial_done}"
            )
            if endpoint_done and partial_done and len(stage_frames) >= thresholds.minimum_stage_count:
                break

        combined = pd.concat(stage_frames, ignore_index=True)
        certificates = certify_convergence(combined, thresholds)

        refinement_cfg = policy.get("reliability", {}).get("lower_tail_refinement", {})
        need_refinement = not bool(certificates["adaptive_endpoint_certified"].map(_as_bool).all())
        if bool(refinement_cfg.get("enabled", True)) and (need_refinement or bool(refinement_cfg.get("always_run", False))):
            refinement_spec = StageSpec(
                index=int(combined["stage_index"].max()) + 1,
                endpoint_steps=int(refinement_cfg.get("steps", max(s.endpoint_steps for s in specs))),
                endpoint_starts=int(refinement_cfg.get("starts", 6)),
                partial_trace_probes=int(refinement_cfg.get("partial_trace_probes", max(s.partial_trace_probes for s in specs))),
                label="lower_tail_multistart_refinement",
                refinement=True,
            )
            stage_output = stages_dir / refinement_spec.label
            config, modified = prepare_core_config(
                base_config,
                stage=refinement_spec,
                output_dir=stage_output,
                diagnostic=True,
                policy=policy,
                shift_overrides_path=shift_overrides_path,
            )
            config_path = generated_dir / f"{refinement_spec.label}.yaml"
            write_yaml(config_path, config)
            (generated_dir / f"{refinement_spec.label}.changes.json").write_text(json.dumps(modified, indent=2, sort_keys=True), encoding="utf-8")
            if stage_output.exists() and not bool(policy.get("reuse_completed_stages", False)):
                shutil.rmtree(stage_output)
            if not stage_output.exists():
                _run_command([python_executable, str(runner), "--config", str(config_path)], cwd=root, log_path=logs_dir / f"{refinement_spec.label}.log", env=runtime_env)
            if validator.exists():
                _run_command([python_executable, str(validator), "--output-dir", str(stage_output)], cwd=root, log_path=logs_dir / f"{refinement_spec.label}.validate.log", env=runtime_env)
            stage_frames.append(summarise_stage(stage_output, refinement_spec, policy))
            combined = pd.concat(stage_frames, ignore_index=True)
            certificates = certify_convergence(combined, thresholds)
            selected_spec = refinement_spec

        final_spec = StageSpec(
            index=selected_spec.index + 1,
            endpoint_steps=int(certificates["selected_endpoint_steps"].max()),
            endpoint_starts=int(certificates["selected_endpoint_starts"].max()),
            partial_trace_probes=int(certificates["selected_partial_trace_probes"].max()),
            label="final_balanced_confirmatory",
            refinement=False,
        )
        final_config, modified = prepare_core_config(
            base_config,
            stage=final_spec,
            output_dir=final_dir,
            diagnostic=False,
            policy=policy,
            shift_overrides_path=shift_overrides_path,
        )
        final_config_path = generated_dir / "final_balanced_confirmatory.yaml"
        write_yaml(final_config_path, final_config)
        (generated_dir / "final_balanced_confirmatory.changes.json").write_text(json.dumps(modified, indent=2, sort_keys=True), encoding="utf-8")
        if final_dir.exists() and not bool(policy.get("reuse_completed_final", False)):
            shutil.rmtree(final_dir)
        if not final_dir.exists():
            _run_command([python_executable, str(runner), "--config", str(final_config_path)], cwd=root, log_path=logs_dir / "final_balanced_confirmatory.log", env=runtime_env)
        if validator.exists():
            _run_command([python_executable, str(validator), "--output-dir", str(final_dir)], cwd=root, log_path=logs_dir / "final_balanced_confirmatory.validate.log", env=runtime_env)

        _write_csv_atomic(combined, output_root / "endpoint_convergence.csv")
        _write_csv_atomic(certificates, output_root / "balanced_reliability_certificates.csv")
        _progress("final core run complete; promoting canonical balanced outputs")
        summary = annotate_final_outputs(final_dir, combined, certificates, policy)
        (output_root / "balanced_reliability_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
        (output_root / "COMPLETED").write_text("balanced reliability pipeline completed\n", encoding="utf-8")
        _progress(f"balanced pipeline completed: {final_dir}")
        return final_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the balanced frozen-experiment reliability pipeline")
    parser.add_argument("--policy", required=True, help="Balanced reliability policy YAML")
    parser.add_argument("--project-root", default=None, help="Repository root; defaults to current working directory")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    final_dir = run_balanced_policy(args.policy, project_root=args.project_root)
    print(final_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
