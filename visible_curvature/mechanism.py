from __future__ import annotations

import math
from typing import Iterable, Mapping, Sequence

import numpy as np
import torch
from scipy.stats import spearmanr

from .covariance import CovarianceEstimate


def _safe_ratio(numerator: torch.Tensor | float, denominator: torch.Tensor | float, eps: float) -> float:
    num = float(numerator.detach().cpu()) if torch.is_tensor(numerator) else float(numerator)
    den = float(denominator.detach().cpu()) if torch.is_tensor(denominator) else float(denominator)
    return num / max(abs(den), float(eps))


def mean_gradient_contamination(centered: CovarianceEstimate, eps: float = 1e-30) -> dict[str, float]:
    """Quantify the mean-gradient contribution excluded by centered covariance.

    Ratios with the ``over_centered`` suffix compare the mean outer-product
    contribution with the centered statistic.  Ratios with the
    ``of_uncentered`` suffix report the fraction of the corresponding
    uncentered statistic attributable to the mean.
    """
    mean = centered.mean.detach()
    mean_diag = mean.square()
    mean_left = mean @ mean.T
    mean_right = mean.T @ mean

    diag_centered = torch.sum(torch.abs(centered.diag))
    diag_mean = torch.sum(torch.abs(mean_diag))
    left_centered = torch.linalg.norm(centered.left)
    left_mean = torch.linalg.norm(mean_left)
    right_centered = torch.linalg.norm(centered.right)
    right_mean = torch.linalg.norm(mean_right)

    return {
        "mean_fraction_adam_over_centered": _safe_ratio(diag_mean, diag_centered, eps),
        "mean_fraction_left_over_centered": _safe_ratio(left_mean, left_centered, eps),
        "mean_fraction_right_over_centered": _safe_ratio(right_mean, right_centered, eps),
        "mean_fraction_adam_of_uncentered": _safe_ratio(diag_mean, diag_centered + diag_mean, eps),
        "mean_fraction_left_of_uncentered": _safe_ratio(left_mean, left_centered + left_mean, eps),
        "mean_fraction_right_of_uncentered": _safe_ratio(right_mean, right_centered + right_mean, eps),
        "mean_gradient_fro_norm": float(torch.linalg.norm(mean).detach().cpu()),
        "centered_diag_l1": float(diag_centered.detach().cpu()),
        "centered_left_fro": float(left_centered.detach().cpu()),
        "centered_right_fro": float(right_centered.detach().cpu()),
    }


def resolve_covariance_moments(cfg: Mapping[str, object]) -> list[str]:
    """Resolve v0.4 multi-moment configuration with v0.3 compatibility."""
    raw = cfg.get("moments")
    if raw is None:
        raw = [cfg.get("moment", "centered")]
    elif isinstance(raw, str):
        raw = [raw]
    moments: list[str] = []
    for value in raw:  # type: ignore[union-attr]
        moment = str(value).lower()
        if moment not in {"centered", "uncentered"}:
            raise ValueError("analysis.covariance moments must be centered or uncentered")
        if moment not in moments:
            moments.append(moment)
    if not moments:
        raise ValueError("analysis.covariance.moments must not be empty")
    return moments


def covariance_variants(centered: CovarianceEstimate, moments: Sequence[str]) -> dict[str, CovarianceEstimate]:
    """Build requested moment variants without repeating gradient collection."""
    out: dict[str, CovarianceEstimate] = {}
    for moment in moments:
        if moment == "centered":
            out[moment] = centered
        elif moment == "uncentered":
            out[moment] = centered.uncentered()
        else:
            raise ValueError(f"Unsupported covariance moment: {moment}")
    return out


def normalized_damping_grid(cfg: Mapping[str, object]) -> list[float]:
    """Return a validated, sorted, duplicate-free normalized damping grid."""
    values = cfg.get("coefficients", cfg.get("grid", []))
    if isinstance(values, (int, float)):
        values = [values]
    grid = sorted({float(value) for value in values})  # type: ignore[arg-type]
    if any((not math.isfinite(value)) or value < 0 for value in grid):
        raise ValueError("damping-sweep coefficients must be finite and nonnegative")
    return grid


def summarize_sign_prediction(actual: Sequence[float], predicted: Sequence[float], zero_tol: float = 1e-12) -> dict[str, float | int]:
    """Summarize continuous and signed prediction quality without sklearn."""
    if len(actual) != len(predicted):
        raise ValueError("actual and predicted must have equal length")
    pairs = [
        (float(a), float(p))
        for a, p in zip(actual, predicted)
        if math.isfinite(float(a)) and math.isfinite(float(p))
    ]
    if not pairs:
        return {
            "n_pairs": 0,
            "n_signed_pairs": 0,
            "spearman": float("nan"),
            "sign_accuracy": float("nan"),
            "sign_balanced_accuracy": float("nan"),
            "true_positive": 0,
            "true_negative": 0,
            "false_positive": 0,
            "false_negative": 0,
        }
    a = np.asarray([pair[0] for pair in pairs], dtype=float)
    p = np.asarray([pair[1] for pair in pairs], dtype=float)
    spearman = (
        float(spearmanr(a, p).statistic)
        if len(pairs) >= 2
        and np.unique(a).size >= 2
        and np.unique(p).size >= 2
        else float("nan")
    )
    signed = [(av, pv) for av, pv in pairs if abs(av) > zero_tol and abs(pv) > zero_tol]
    tp = sum(av > 0 and pv > 0 for av, pv in signed)
    tn = sum(av < 0 and pv < 0 for av, pv in signed)
    fp = sum(av < 0 and pv > 0 for av, pv in signed)
    fn = sum(av > 0 and pv < 0 for av, pv in signed)
    total = len(signed)
    accuracy = (tp + tn) / total if total else float("nan")
    pos_total = tp + fn
    neg_total = tn + fp
    tpr = tp / pos_total if pos_total else float("nan")
    tnr = tn / neg_total if neg_total else float("nan")
    balanced = 0.5 * (tpr + tnr) if math.isfinite(tpr) and math.isfinite(tnr) else float("nan")
    return {
        "n_pairs": len(pairs),
        "n_signed_pairs": total,
        "spearman": spearman,
        "sign_accuracy": float(accuracy),
        "sign_balanced_accuracy": float(balanced),
        "true_positive": int(tp),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
    }


def _truthy(value: object) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if value is None:
        return False
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        try:
            if not math.isfinite(float(value)):
                return False
        except (TypeError, ValueError):
            return False
        return bool(value)
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def build_elasticity_prediction_rows(
    metrics: "pd.DataFrame",
    *,
    primary_alpha: float = 0.25,
) -> "pd.DataFrame":
    """Return long-form predictor rows with explicit fail-closed eligibility.

    The consumption-only term and the full commuting--Kronecker proxy are
    reported separately.  Eligibility requires a primary centered/observed
    row, an accepted primary conditioning estimate when that status is
    available, a reliable factor-elasticity diagnostic, and finite values.
    """
    import pandas as pd

    predictors = {
        "consumption": "delta_g_predicted_consumption",
        "full_proxy": "delta_g_predicted_full_proxy",
    }
    if "delta_g_predicted_full_proxy" not in metrics.columns and "delta_g_predicted" in metrics.columns:
        predictors["full_proxy"] = "delta_g_predicted"

    output: list[dict[str, object]] = []
    for _, source in metrics.iterrows():
        primary = True
        reasons: list[str] = []
        if "covariance_moment" in metrics.columns:
            primary &= str(source.get("covariance_moment", "")) == "centered"
        if "assignment" in metrics.columns:
            primary &= str(source.get("assignment", "")) == "observed"
        if "alpha" in metrics.columns:
            try:
                primary &= math.isclose(
                    float(source.get("alpha")),
                    float(primary_alpha),
                    rel_tol=0.0,
                    abs_tol=1.0e-12,
                )
            except (TypeError, ValueError):
                primary = False
        if not primary:
            reasons.append("nonprimary")

        primary_reliable = True
        if "balanced_primary_reliable" in metrics.columns:
            primary_reliable = _truthy(source.get("balanced_primary_reliable"))
        elif "ordering_inferentially_reliable" in metrics.columns:
            primary_reliable = _truthy(source.get("ordering_inferentially_reliable"))
        if not primary_reliable:
            reasons.append("primary_ordering")

        factor_reliable = (
            _truthy(source.get("factor_elasticity_reliable"))
            if "factor_elasticity_reliable" in metrics.columns
            else False
        )
        if not factor_reliable:
            reasons.append("factor_elasticity")

        try:
            actual = float(source.get("delta_g", float("nan")))
        except (TypeError, ValueError):
            actual = float("nan")
        if not math.isfinite(actual):
            reasons.append("actual_nonfinite")

        for predictor_name, column in predictors.items():
            try:
                predicted = float(source.get(column, float("nan")))
            except (TypeError, ValueError):
                predicted = float("nan")
            predictor_reasons = list(reasons)
            if not math.isfinite(predicted):
                predictor_reasons.append("predictor_nonfinite")
            row = source.to_dict()
            row.update(
                {
                    "predictor_name": predictor_name,
                    "predictor_source_column": column,
                    "actual_delta_g": actual,
                    "predicted_delta_g": predicted,
                    "prediction_eligible": not predictor_reasons,
                    "prediction_eligibility_reasons": ",".join(
                        dict.fromkeys(predictor_reasons)
                    ),
                }
            )
            output.append(row)

    columns = [
        *metrics.columns.tolist(),
        "predictor_name",
        "predictor_source_column",
        "actual_delta_g",
        "predicted_delta_g",
        "prediction_eligible",
        "prediction_eligibility_reasons",
    ]
    return pd.DataFrame(output, columns=list(dict.fromkeys(columns)))


def summarize_elasticity_predictions(
    rows: "pd.DataFrame",
    *,
    zero_tol: float = 1.0e-12,
) -> "pd.DataFrame":
    """Summarize sign and rank agreement for each declared predictor."""
    import pandas as pd

    columns = [
        "predictor_name",
        "n_candidate_rows",
        "n_eligible_rows",
        "n_pairs",
        "n_signed_pairs",
        "spearman",
        "sign_accuracy",
        "sign_balanced_accuracy",
        "true_positive",
        "true_negative",
        "false_positive",
        "false_negative",
    ]
    if rows.empty or "predictor_name" not in rows.columns:
        return pd.DataFrame(columns=columns)

    output: list[dict[str, object]] = []
    for predictor_name, group in rows.groupby("predictor_name", dropna=False):
        eligible_mask = group["prediction_eligible"].map(_truthy)
        eligible = group[eligible_mask]
        summary = summarize_sign_prediction(
            eligible.get("actual_delta_g", pd.Series(dtype=float)).tolist(),
            eligible.get("predicted_delta_g", pd.Series(dtype=float)).tolist(),
            zero_tol=zero_tol,
        )
        output.append(
            {
                "predictor_name": str(predictor_name),
                "n_candidate_rows": int(len(group)),
                "n_eligible_rows": int(len(eligible)),
                **summary,
            }
        )
    return pd.DataFrame(output, columns=columns)
