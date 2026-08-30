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
    spearman = float(spearmanr(a, p).statistic) if len(pairs) >= 2 else float("nan")
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
