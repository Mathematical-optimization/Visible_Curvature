from __future__ import annotations

import copy
import math
import re
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

import yaml

_IMMUTABLE_COMMIT = re.compile(r"^[0-9a-fA-F]{40}$")
_ALLOWED_TIERS = {"debug", "screening", "confirmatory"}
_ALLOWED_ASSIGNMENTS = {"observed", "aligned", "reversed"}
_ALLOWED_DAMPING_SWEEP_MODES = {"joint", "shampoo_only"}
_RELIABILITY_KEYS = {
    "min_r2",
    "max_commutator",
    "max_shift_ratio",
    "max_min_ritz_residual_over_min",
    "max_max_ritz_residual_over_max",
    "sign_zero_tol",
    "bootstrap_alpha",
    "minimum_bootstrap_reps",
    "max_factor_negative_mass",
    "max_factor_eigen_residual",
    "min_elasticity_modes",
    "min_curvature_log_width",
    "max_preconditioner_floored_fraction",
}


def _intervals_overlap(left: tuple[int, int], right: tuple[int, int]) -> bool:
    return max(left[0], right[0]) < min(left[1], right[1])


def _require_commit(value: Any, field: str) -> None:
    if not isinstance(value, str) or _IMMUTABLE_COMMIT.fullmatch(value) is None:
        raise ValueError(f"{field} must be a full 40-character immutable commit SHA")


def _positive_int(value: Any, field: str, *, allow_zero: bool = False) -> int:
    parsed = int(value)
    if parsed < 0 or (parsed == 0 and not allow_zero):
        relation = "nonnegative" if allow_zero else "positive"
        raise ValueError(f"{field} must be {relation}")
    return parsed


def _validate_frozen_config(cfg: Dict[str, Any]) -> None:
    if "online" in cfg:
        raise ValueError("online experiments were removed from the focused package")
    model = cfg.setdefault("model", {})
    data = cfg.setdefault("data", {})
    if str(model.get("backend", "tiny_causal_lm")) not in {"tiny_causal_lm", "hf_causal_lm"}:
        raise ValueError("model.backend must be tiny_causal_lm or hf_causal_lm")
    if str(data.get("backend", "synthetic_tokens")) not in {"synthetic_tokens", "hf_text"}:
        raise ValueError("data.backend must be synthetic_tokens or hf_text")
    data["order_seed"] = int(data.get("order_seed", 0))

    blocks = cfg.setdefault("blocks", {})
    raw_exact_names = blocks.get("exact_names", [])
    if isinstance(raw_exact_names, str):
        raw_exact_names = [raw_exact_names]
    exact_names = [str(value) for value in raw_exact_names]
    if any(not value.strip() for value in exact_names):
        raise ValueError("blocks.exact_names must contain nonempty names")
    if len(set(exact_names)) != len(exact_names):
        raise ValueError("blocks.exact_names must not contain duplicates")
    if exact_names:
        blocks["exact_names"] = exact_names
        blocks["include"] = [f"^{re.escape(name)}$" for name in exact_names]
        blocks["max_blocks"] = len(exact_names)
        blocks["selection_strategy"] = "first"

    analysis = cfg.setdefault("analysis", {})
    forbidden = sorted(set(analysis).intersection({"nci_samples", "nci_skip_batches", "frozen"}))
    if forbidden:
        raise ValueError(f"Removed analysis option(s): {', '.join(forbidden)}")
    tier = str(analysis.get("compute_tier", "debug"))
    if tier not in _ALLOWED_TIERS:
        raise ValueError(f"analysis.compute_tier must be one of {sorted(_ALLOWED_TIERS)}")
    analysis["compute_tier"] = tier

    reliability = analysis.setdefault("reliability", {})
    unknown = sorted(set(reliability).difference(_RELIABILITY_KEYS))
    if unknown:
        raise ValueError(f"Unknown analysis.reliability key(s): {', '.join(unknown)}")

    assignments = [str(value) for value in analysis.get("assignments", ["observed", "aligned", "reversed"])]
    if not assignments or any(value not in _ALLOWED_ASSIGNMENTS for value in assignments):
        raise ValueError("analysis.assignments may contain only observed, aligned, and reversed")
    if len(set(assignments)) != len(assignments):
        raise ValueError("analysis.assignments must not contain duplicates")
    analysis["assignments"] = assignments

    controls = analysis.setdefault("controls", {})
    raw_control_names = controls.get("block_names", [])
    if isinstance(raw_control_names, str):
        raw_control_names = [raw_control_names]
    control_names = [str(value) for value in raw_control_names]
    if any(not value.strip() for value in control_names):
        raise ValueError("analysis.controls.block_names must contain nonempty names")
    if len(set(control_names)) != len(control_names):
        raise ValueError("analysis.controls.block_names must not contain duplicates")
    controls["block_names"] = control_names
    if exact_names and not set(control_names).issubset(exact_names):
        raise ValueError(
            "analysis control block_names must be a subset of blocks.exact_names"
        )

    alpha_values = [float(value) for value in analysis.setdefault("alpha_sweep", {}).get("values", [0.25, 0.5])]
    if not alpha_values or any(not 0.0 < value <= 0.5 for value in alpha_values):
        raise ValueError("analysis.alpha_sweep.values must lie in (0, 0.5]")
    analysis["alpha_sweep"]["values"] = alpha_values

    damping_sweep = analysis.setdefault("damping_sweep", {})
    damping_modes = [
        str(value) for value in damping_sweep.get("modes", ["joint"])
    ]
    if (
        not damping_modes
        or any(value not in _ALLOWED_DAMPING_SWEEP_MODES for value in damping_modes)
        or len(set(damping_modes)) != len(damping_modes)
    ):
        raise ValueError(
            "analysis.damping_sweep.modes may contain unique joint and shampoo_only values"
        )
    damping_sweep["modes"] = damping_modes

    ridge_sensitivity = analysis.setdefault("ridge_sensitivity", {})
    ridge_sensitivity["enabled"] = bool(ridge_sensitivity.get("enabled", False))
    ridge_coefficients = [
        float(value)
        for value in ridge_sensitivity.get(
            "coefficients", [1.0e-6, 1.0e-5, 1.0e-4]
        )
    ]
    if any(not math.isfinite(value) or value < 0.0 for value in ridge_coefficients):
        raise ValueError(
            "analysis.ridge_sensitivity.coefficients must be finite and nonnegative"
        )
    if len(set(ridge_coefficients)) != len(ridge_coefficients):
        raise ValueError(
            "analysis.ridge_sensitivity.coefficients must contain unique values"
        )
    if ridge_sensitivity["enabled"] and not ridge_coefficients:
        raise ValueError(
            "analysis.ridge_sensitivity.coefficients must be nonempty when enabled"
        )
    ridge_sensitivity["coefficients"] = ridge_coefficients

    covariance = analysis.setdefault("covariance", {})
    curvature = analysis.setdefault("curvature", {})
    covariance_start = _positive_int(covariance.get("skip_batches", 0), "analysis.covariance.skip_batches", allow_zero=True)
    covariance_count = _positive_int(covariance.get("num_batches", 16), "analysis.covariance.num_batches")
    covariance_group_size = _positive_int(
        covariance.get("group_size", 4),
        "analysis.covariance.group_size",
    )
    covariance["group_size"] = covariance_group_size
    curvature_start = _positive_int(
        curvature.get("skip_batches", covariance_start + covariance_count),
        "analysis.curvature.skip_batches",
        allow_zero=True,
    )
    curvature_count = _positive_int(curvature.get("num_batches", 1), "analysis.curvature.num_batches")
    if _intervals_overlap(
        (covariance_start, covariance_start + covariance_count),
        (curvature_start, curvature_start + curvature_count),
    ):
        raise ValueError("Frozen stream intervals overlap: covariance and curvature")

    bootstrap = analysis.setdefault("bootstrap", {})
    reps = _positive_int(bootstrap.get("reps", 0), "analysis.bootstrap.reps", allow_zero=True)
    diagnostics = str(bootstrap.get("diagnostics", "delta_only"))
    if diagnostics != "delta_only":
        raise ValueError("analysis.bootstrap.diagnostics must be delta_only")
    bootstrap["diagnostics"] = diagnostics

    scientific = bool(cfg.get("scientific_run", False))
    if scientific:
        if str(model.get("backend")) != "hf_causal_lm" or str(data.get("backend")) != "hf_text":
            raise ValueError("scientific_run requires hf_causal_lm with hf_text")
        _require_commit(model.get("revision"), "model.revision")
        _require_commit(data.get("revision"), "data.revision")
        _require_commit(data.get("tokenizer_revision"), "data.tokenizer_revision")
        if "order_seed" not in data:
            raise ValueError("scientific_run requires data.order_seed")
        if tier == "confirmatory" and not exact_names:
            raise ValueError(
                "Scientific confirmatory runs require blocks.exact_names; "
                "unanchored or discovery-only regex selection is not permitted"
            )
        if tier == "confirmatory" and reps < 100:
            raise ValueError(
                f"Scientific confirmatory runs require at least 100 delta-only bootstrap replicates; got {reps}"
            )
        if tier == "screening" and reps != 0:
            raise ValueError("Scientific screening runs must set bootstrap.reps=0")
        if covariance_count % covariance_group_size != 0:
            raise ValueError(
                "Scientific grouped bootstrap requires analysis.covariance.num_batches "
                "to be divisible by analysis.covariance.group_size"
            )


def validate_config(cfg: Mapping[str, Any], mode: str) -> Dict[str, Any]:
    """Return a validated mutable configuration copy for the focused package."""
    validated: Dict[str, Any] = copy.deepcopy(dict(cfg))
    if str(mode) != "frozen":
        raise ValueError("mode must be 'frozen'")
    _validate_frozen_config(validated)
    return validated


def load_yaml(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as file:
        cfg = yaml.safe_load(file) or {}
    cfg["_config_path"] = str(path.resolve())
    return cfg


def deep_get(cfg: Mapping[str, Any], dotted: str, default: Any = None) -> Any:
    current: Any = cfg
    for key in dotted.split("."):
        if not isinstance(current, Mapping) or key not in current:
            return default
        current = current[key]
    return current


def deep_update(base: Dict[str, Any], updates: Mapping[str, Any]) -> Dict[str, Any]:
    for key, value in updates.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), dict):
            deep_update(base[key], value)
        else:
            base[key] = value
    return base


def ensure_output_dir(cfg: Mapping[str, Any]) -> Path:
    output = Path(str(cfg.get("output_dir", "outputs/run")))
    output.mkdir(parents=True, exist_ok=True)
    return output


def save_resolved_config(cfg: Mapping[str, Any], output_dir: str | Path) -> Path:
    path = Path(output_dir) / "resolved_config.yaml"
    clean = {key: value for key, value in cfg.items() if not str(key).startswith("_")}
    with path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(clean, file, sort_keys=False, allow_unicode=True)
    return path


def resolve_name_rule(name: str, default: float, rules: Iterable[Mapping[str, Any]] | None) -> float:
    """Resolve a numeric hyperparameter from first-matching regex rules."""
    for rule in rules or []:
        if re.search(str(rule["pattern"]), name):
            return float(rule["value"])
    return float(default)
