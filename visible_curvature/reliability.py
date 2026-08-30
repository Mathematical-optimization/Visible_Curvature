from __future__ import annotations

import math
from typing import Iterable, Mapping

import numpy as np


def _signed(value: float, zero_tol: float) -> int:
    if not math.isfinite(float(value)) or abs(float(value)) <= float(zero_tol):
        return 0
    return 1 if float(value) > 0 else -1


def tau_sign_stability(values: Mapping[float | str, float], zero_tol: float = 1e-12) -> dict[str, object]:
    """Check whether every resolved truncation level gives the same sign."""
    finite = [(str(key), float(value), _signed(float(value), zero_tol)) for key, value in values.items() if math.isfinite(float(value))]
    if not finite:
        return {"stable": False, "sign": 0, "reason": "no_finite_values", "num_values": 0}
    if any(sign == 0 for _, _, sign in finite):
        return {"stable": False, "sign": 0, "reason": "unresolved_zero", "num_values": len(finite)}
    signs = {sign for _, _, sign in finite}
    if len(signs) != 1:
        return {"stable": False, "sign": 0, "reason": "sign_flip", "num_values": len(finite)}
    sign = signs.pop()
    return {"stable": True, "sign": int(sign), "reason": "", "num_values": len(finite)}


def calibrated_bootstrap_interval(
    *,
    point: float,
    bootstrap_values: Iterable[float],
    reference: float,
    alpha: float = 0.05,
    minimum_reps: int = 2,
) -> tuple[float, float]:
    """Calibrate a low-budget bootstrap error distribution to a high-budget point.

    The replicate errors are ``bootstrap_value - low_budget_reference``.
    A basic centered-difference interval is then placed around ``point``.
    """
    point = float(point)
    reference = float(reference)
    alpha = float(alpha)
    if not (math.isfinite(point) and math.isfinite(reference)):
        return float("nan"), float("nan")
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must lie strictly between zero and one")
    values = np.asarray(list(bootstrap_values), dtype=float)
    values = values[np.isfinite(values)]
    if values.size < int(minimum_reps):
        return float("nan"), float("nan")
    errors = values - reference
    lower_error = float(np.quantile(errors, alpha / 2.0))
    upper_error = float(np.quantile(errors, 1.0 - alpha / 2.0))
    return float(point - upper_error), float(point - lower_error)


def classify_reliable_ordering(
    *,
    delta: float,
    ci_low: float,
    ci_high: float,
    checks: Mapping[str, bool],
) -> dict[str, object]:
    """Combine point sign, bootstrap inference, and numerical reliability."""
    if not math.isfinite(float(delta)):
        point = "undefined"
    elif delta > 0:
        point = "shampoo_favorable"
    elif delta < 0:
        point = "shampoo_unfavorable"
    else:
        point = "neutral"

    if math.isfinite(float(ci_low)) and math.isfinite(float(ci_high)):
        if ci_low > 0:
            ci_label = "positive"
        elif ci_high < 0:
            ci_label = "negative"
        else:
            ci_label = "inconclusive"
    else:
        ci_label = "inconclusive"

    failed = sorted(str(name) for name, passed in checks.items() if not bool(passed))
    point_sign = 1 if point == "shampoo_favorable" else (-1 if point == "shampoo_unfavorable" else 0)
    ci_sign = 1 if ci_label == "positive" else (-1 if ci_label == "negative" else 0)
    if ci_sign == 0:
        failed.append("bootstrap_interval_inconclusive")
    elif point_sign != ci_sign:
        failed.append("point_bootstrap_sign_mismatch")
    failed = sorted(set(failed))
    reliable = not failed and ci_label in {"positive", "negative"}
    return {
        "point_label": point,
        "ci_label": ci_label,
        "reliable_label": ci_label if reliable else "inconclusive",
        "reliable": bool(reliable),
        "reliability_reasons": ",".join(failed),
    }


def ritz_residual_check(
    min_ritz: float,
    max_ritz: float,
    min_residual: float,
    max_residual: float,
    *,
    relative_to_min: float = 0.25,
    relative_to_max: float = 1e-3,
) -> bool:
    """Conservative endpoint check using the relevant spectral scale."""
    values = [min_ritz, max_ritz, min_residual, max_residual]
    if not all(math.isfinite(float(value)) for value in values):
        return False
    if min_ritz <= 0 or max_ritz <= 0 or min_ritz > max_ritz:
        return False
    return min_residual <= relative_to_min * min_ritz and max_residual <= relative_to_max * max_ritz
