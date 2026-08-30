from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, Iterable, Sequence

import numpy as np
import torch
from scipy.stats import spearmanr

from .linear_algebra import symmetric_eigendecomp, symmetric_eigendecomp_with_metadata

Tensor = torch.Tensor


@dataclass
class ElasticityFit:
    slope: float
    intercept: float
    r2: float
    residual_rmse: float
    n: int
    sxx: float


def _linear_fit(x: np.ndarray, y: np.ndarray) -> ElasticityFit:
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size < 2 or float(np.var(x)) <= 1e-20:
        return ElasticityFit(float("nan"), float("nan"), float("nan"), float("nan"), int(x.size), 0.0)
    xm = x.mean()
    ym = y.mean()
    sxx = float(np.sum((x - xm) ** 2))
    slope = float(np.sum((x - xm) * (y - ym)) / sxx)
    intercept = float(ym - slope * xm)
    pred = intercept + slope * x
    resid = y - pred
    sst = float(np.sum((y - ym) ** 2))
    sse = float(np.sum(resid ** 2))
    r2 = 1.0 - sse / sst if sst > 0 else float("nan")
    rmse = math.sqrt(sse / x.size)
    return ElasticityFit(slope, intercept, r2, rmse, int(x.size), sxx)


def visible_elasticity(
    curvature_factor: Tensor,
    covariance_factor: Tensor,
    damping: float,
    max_full_eigh_dim: int = 4096,
    approx_k: int = 64,
    curvature_rel_floor: float = 1e-6,
    curvature_eigendecomp: tuple[Tensor, Tensor, dict] | None = None,
    covariance_eigendecomp: tuple[Tensor, Tensor, dict] | None = None,
) -> tuple[ElasticityFit, dict]:
    if curvature_eigendecomp is None:
        hvals, U, hmeta = symmetric_eigendecomp_with_metadata(
            curvature_factor, max_full_dim=max_full_eigh_dim, approx_k=approx_k
        )
    else:
        hvals, U, hmeta = curvature_eigendecomp
    q_rayleigh = torch.sum(U * (covariance_factor @ U), dim=0)
    hmax = max(float(hvals.max().detach().cpu()), 0.0) if hvals.numel() else 0.0
    mask = hvals > max(torch.finfo(hvals.dtype).tiny, curvature_rel_floor * hmax)
    h = hvals[mask].detach().cpu().numpy()
    q = q_rayleigh[mask].clamp_min(0).detach().cpu().numpy()
    fit = _linear_fit(np.log(h), np.log(q + float(damping))) if h.size else ElasticityFit(float("nan"), float("nan"), float("nan"), float("nan"), 0, 0.0)

    # Covariance eigenspace diagnostics.
    if covariance_eigendecomp is None:
        qvals, V, qmeta = symmetric_eigendecomp_with_metadata(
            covariance_factor, max_full_dim=max_full_eigh_dim, approx_k=approx_k
        )
    else:
        qvals, V, qmeta = covariance_eigendecomp
    k = min(16, U.shape[1], V.shape[1])
    if k > 0:
        # Cached curvature and covariance decompositions may intentionally use
        # different precisions (for example float32 H factors and float64
        # covariance-factor roots).  Principal-angle diagnostics are invariant
        # to this storage choice, so evaluate the overlap in one stable dtype.
        overlap_dtype = torch.float64
        Utop = U[:, -k:].to(device=U.device, dtype=overlap_dtype)
        Vtop = V[:, -k:].to(device=U.device, dtype=overlap_dtype)
        overlap = Utop.T @ Vtop
        svals = torch.linalg.svdvals(overlap).clamp(0, 1)
        angles = torch.arccos(svals)
        mean_angle_deg = float(torch.mean(angles).detach().cpu() * 180.0 / math.pi)
        overlap_fro2 = float(torch.sum(overlap.square()).detach().cpu() / k)
    else:
        mean_angle_deg = float("nan")
        overlap_fro2 = float("nan")

    if h.size >= 2:
        rank_corr = float(spearmanr(np.log(h), np.log(q + float(damping))).statistic)
    else:
        rank_corr = float("nan")

    aux = {
        "h_min_used": float(np.min(h)) if h.size else float("nan"),
        "h_max_used": float(np.max(h)) if h.size else float("nan"),
        "q_min_used": float(np.min(q)) if q.size else float("nan"),
        "q_max_used": float(np.max(q)) if q.size else float("nan"),
        "mean_top_principal_angle_deg": mean_angle_deg,
        "top_subspace_overlap_fro2": overlap_fro2,
        "rank_corr_h_q": rank_corr,
        "num_factor_modes": int(h.size),
        "eigenspectrum_scope": str(hmeta["scope"]),
        "eigen_dimension": int(hmeta["dimension"]),
        "eigen_modes_returned": int(hmeta["modes_returned"]),
        "eigen_mode_fraction": float(hmeta["mode_fraction"]),
        "eigen_max_relative_residual": float(hmeta["max_relative_residual"]),
        "covariance_eigenspectrum_scope": str(qmeta["scope"]),
        "covariance_eigen_max_relative_residual": float(qmeta["max_relative_residual"]),
        "negative_mode_fraction": float((hvals < 0).float().mean().detach().cpu()) if hvals.numel() else float("nan"),
        "negative_spectral_mass": float(
            (-hvals.clamp_max(0)).sum().detach().cpu()
            / hvals.abs().sum().clamp_min(torch.finfo(hvals.dtype).tiny).detach().cpu()
        ) if hvals.numel() else float("nan"),
    }
    return fit, aux


def combine_factor_elasticities(
    left: ElasticityFit,
    right: ElasticityFit,
    left_log_width: float | None = None,
    right_log_width: float | None = None,
) -> float:
    """Combine left/right factor elasticities into one signed predictor.

    For a Kronecker product the total log condition width is additive across
    factors.  When factor log-widths are available, weighting each elasticity
    by its curvature log-width matches that additive geometry.  The older SXX
    weighting is retained as a fallback when widths are unavailable.
    """
    if not math.isfinite(left.slope) and not math.isfinite(right.slope):
        return float("nan")
    if not math.isfinite(left.slope):
        return right.slope
    if not math.isfinite(right.slope):
        return left.slope
    if left_log_width is not None and right_log_width is not None:
        wl = max(float(left_log_width), 0.0) if math.isfinite(float(left_log_width)) else 0.0
        wr = max(float(right_log_width), 0.0) if math.isfinite(float(right_log_width)) else 0.0
        if wl + wr > 0:
            return (left.slope * wl + right.slope * wr) / (wl + wr)
    denom = left.sxx + right.sxx
    if denom <= 0:
        return 0.5 * (left.slope + right.slope)
    return (left.slope * left.sxx + right.slope * right.sxx) / denom


def normalized_commutator(A: Tensor, B: Tensor, eps: float = 1e-30) -> float:
    C = A @ B - B @ A
    denom = torch.linalg.norm(A) * torch.linalg.norm(B)
    return float((torch.linalg.norm(C) / denom.clamp_min(eps)).detach().cpu())


def coefficient_of_variation(x: Tensor, eps: float = 1e-30) -> float:
    x = x.detach().float().flatten()
    mean = torch.mean(x).abs()
    return float((torch.std(x, unbiased=False) / mean.clamp_min(eps)).cpu())


def relative_spectrum_error(A: Tensor, B: Tensor, eps: float = 1e-30) -> float:
    ea = torch.linalg.eigvalsh(0.5 * (A + A.T))
    eb = torch.linalg.eigvalsh(0.5 * (B + B.T))
    denom = torch.linalg.norm(ea).clamp_min(eps)
    return float((torch.linalg.norm(torch.sort(ea).values - torch.sort(eb).values) / denom).detach().cpu())


def relative_tensor_error(a: Tensor, b: Tensor, eps: float = 1e-30) -> float:
    return float((torch.linalg.norm(a - b) / torch.linalg.norm(a).clamp_min(eps)).detach().cpu())


def safe_log_condition(K: float) -> float:
    return math.log(K) if math.isfinite(K) and K > 0 else float("nan")


def adam_coordinate_elasticity(
    curvature_diagonal: Tensor,
    covariance_diagonal: Tensor,
    damping: float,
    curvature_rel_floor: float = 1e-6,
) -> tuple[ElasticityFit, dict]:
    """Fit the coordinate-visible response retained by Adam.

    The regression uses matched coordinates rather than independently sorted
    spectra, so its sign is an assignment statistic.  Negative or unresolved
    curvature diagonal estimates are removed before taking logarithms.
    """
    h_t = curvature_diagonal.detach().double().flatten()
    q_t = covariance_diagonal.detach().double().flatten().clamp_min(0)
    if h_t.numel() != q_t.numel():
        raise ValueError("curvature and covariance diagonals must have the same number of entries")
    if h_t.numel() == 0:
        return ElasticityFit(float("nan"), float("nan"), float("nan"), float("nan"), 0, 0.0), {
            "curvature_log_width": 0.0,
            "num_coordinate_modes": 0,
        }
    hmax = max(float(h_t.max().cpu()), 0.0)
    floor = max(torch.finfo(h_t.dtype).tiny, float(curvature_rel_floor) * hmax)
    mask = torch.isfinite(h_t) & torch.isfinite(q_t) & (h_t > floor)
    h = h_t[mask].cpu().numpy()
    q = q_t[mask].cpu().numpy()
    if h.size:
        y = np.log(np.maximum(q + float(damping), np.finfo(np.float64).tiny))
        fit = _linear_fit(np.log(h), y)
        lo, hi = float(np.min(h)), float(np.max(h))
        width = math.log(hi / lo) if hi > lo > 0 else 0.0
        rank = float(spearmanr(np.log(h), y).statistic) if h.size >= 2 else float("nan")
    else:
        fit = ElasticityFit(float("nan"), float("nan"), float("nan"), float("nan"), 0, 0.0)
        lo = hi = float("nan")
        width = 0.0
        rank = float("nan")
    return fit, {
        "curvature_diag_min_used": lo,
        "curvature_diag_max_used": hi,
        "curvature_log_width": width,
        "num_coordinate_modes": int(h.size),
        "rank_corr_coordinate": rank,
    }



def predicted_delta_g_components(
    *,
    r_left: float,
    width_left: float,
    r_right: float,
    width_right: float,
    r_adam: float,
    width_adam: float,
    factor_exponent: float = 0.25,
) -> dict[str, float]:
    """Return the commuting--Kronecker proxy decomposition for ``Delta G``.

    The response-consumption term alone is not a prediction of
    ``G_Shampoo - G_Adam`` unless the Adam coordinate-curvature width equals
    the sum of the two Shampoo factor-curvature widths.  The explicit
    baseline term keeps this structural assumption auditable.
    """
    vals = [
        r_left,
        width_left,
        r_right,
        width_right,
        r_adam,
        width_adam,
        factor_exponent,
    ]
    if not all(math.isfinite(float(v)) for v in vals):
        return {
            "baseline_width_mismatch": float("nan"),
            "delta_g_predicted_consumption": float("nan"),
            "delta_g_predicted_full_proxy": float("nan"),
        }
    alpha = float(factor_exponent)
    if not 0.0 < alpha <= 0.5:
        raise ValueError("factor_exponent must lie in (0, 0.5]")
    width_l = float(width_left)
    width_r = float(width_right)
    width_a = float(width_adam)
    baseline = width_a - (width_l + width_r)
    consumption = alpha * (
        float(r_left) * width_l + float(r_right) * width_r
    ) - 0.5 * float(r_adam) * width_a
    return {
        "baseline_width_mismatch": baseline,
        "delta_g_predicted_consumption": consumption,
        "delta_g_predicted_full_proxy": baseline + consumption,
    }


def predicted_delta_g(
    *,
    r_left: float,
    width_left: float,
    r_right: float,
    width_right: float,
    r_adam: float,
    width_adam: float,
    factor_exponent: float = 0.25,
) -> float:
    """Return the full commuting--Kronecker proxy for ``G_Shampoo-G_Adam``.

    This compatibility wrapper returns ``delta_g_predicted_full_proxy``.
    Call :func:`predicted_delta_g_components` when the baseline and
    response-consumption contributions must be reported separately.
    """
    return predicted_delta_g_components(
        r_left=r_left,
        width_left=width_left,
        r_right=r_right,
        width_right=width_right,
        r_adam=r_adam,
        width_adam=width_adam,
        factor_exponent=factor_exponent,
    )["delta_g_predicted_full_proxy"]
