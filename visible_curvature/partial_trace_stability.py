"""Persistence and cross-budget stability checks for partial-trace geometry.

The aligned/reversed intervention depends on curvature-factor invariant
subspaces, not only on whether a partial trace is approximately PSD.  This
module persists the matrices and the induced factor interventions and compares
consecutive probe budgets using matrix, clustered-subspace, and intervention
stability diagnostics.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from .interventions import build_factor_intervention


@dataclass(frozen=True)
class PartialTraceStabilityThresholds:
    matrix_relative_tolerance: float = 0.10
    subspace_projector_tolerance: float = 0.10
    intervention_factor_relative_tolerance: float = 0.10
    cluster_relative_gap: float = 1.0e-4
    maximum_negative_mass: float = 0.05
    negative_mass_change_tolerance: float = 0.02


def _safe_block_id(block_name: str) -> str:
    return hashlib.sha256(str(block_name).encode("utf-8")).hexdigest()[:24]


def partial_trace_artifact_path(root: str | Path, block_name: str) -> Path:
    return Path(root) / f"{_safe_block_id(block_name)}.npz"


def _symmetrize(value: torch.Tensor) -> torch.Tensor:
    work = value.detach().to(device="cpu", dtype=torch.float64)
    return 0.5 * (work + work.T)


def save_partial_trace_artifact(
    root: str | Path,
    *,
    block_name: str,
    left: torch.Tensor,
    right: torch.Tensor,
    covariance_left: torch.Tensor,
    covariance_right: torch.Tensor,
) -> Path:
    """Persist partial traces and the exact aligned/reversed factor controls."""
    root_path = Path(root)
    root_path.mkdir(parents=True, exist_ok=True)
    left_work = _symmetrize(left)
    right_work = _symmetrize(right)
    covariance_left_work = _symmetrize(covariance_left)
    covariance_right_work = _symmetrize(covariance_right)

    left_values, left_vectors = torch.linalg.eigh(left_work)
    right_values, right_vectors = torch.linalg.eigh(right_work)
    aligned_left, aligned_right = build_factor_intervention(
        covariance_left_work,
        covariance_right_work,
        left_work,
        right_work,
        mode="aligned",
        left_curvature_eigendecomp=(left_values, left_vectors),
        right_curvature_eigendecomp=(right_values, right_vectors),
    )
    reversed_left, reversed_right = build_factor_intervention(
        covariance_left_work,
        covariance_right_work,
        left_work,
        right_work,
        mode="reversed",
        left_curvature_eigendecomp=(left_values, left_vectors),
        right_curvature_eigendecomp=(right_values, right_vectors),
    )

    path = partial_trace_artifact_path(root_path, block_name)
    np.savez_compressed(
        path,
        block_name=np.asarray(str(block_name)),
        left=left_work.numpy(),
        right=right_work.numpy(),
        left_eigenvalues=left_values.numpy(),
        right_eigenvalues=right_values.numpy(),
        aligned_left=aligned_left.numpy(),
        aligned_right=aligned_right.numpy(),
        reversed_left=reversed_left.numpy(),
        reversed_right=reversed_right.numpy(),
    )

    index_path = root_path / "index.json"
    index: dict[str, Any]
    if index_path.exists():
        loaded = json.loads(index_path.read_text(encoding="utf-8"))
        index = dict(loaded) if isinstance(loaded, Mapping) else {}
    else:
        index = {}
    index[_safe_block_id(block_name)] = {
        "block_name": str(block_name),
        "artifact": path.name,
    }
    temporary = index_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(index, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(index_path)
    return path


def load_partial_trace_artifact(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    with np.load(source, allow_pickle=False) as payload:
        result: dict[str, Any] = {}
        for key in payload.files:
            value = payload[key]
            if value.ndim == 0 and value.dtype.kind in {"U", "S"}:
                result[key] = str(value.item())
            else:
                result[key] = np.asarray(value, dtype=np.float64)
    return result


def _relative_frobenius(current: np.ndarray, previous: np.ndarray) -> float:
    denominator = max(float(np.linalg.norm(previous, ord="fro")), np.finfo(float).tiny)
    return float(np.linalg.norm(current - previous, ord="fro") / denominator)


def _negative_spectral_mass(matrix: np.ndarray) -> float:
    values = np.linalg.eigvalsh(0.5 * (matrix + matrix.T))
    denominator = max(float(np.abs(values).sum()), np.finfo(float).tiny)
    return float(np.maximum(-values, 0.0).sum() / denominator)


def _cluster_slices(values: np.ndarray, relative_gap: float) -> list[slice]:
    values = np.asarray(values, dtype=float)
    if values.ndim != 1:
        raise ValueError("eigenvalues must be one-dimensional")
    if values.size == 0:
        return []
    starts = [0]
    for index in range(values.size - 1):
        scale = max(abs(float(values[index])), abs(float(values[index + 1])), np.finfo(float).tiny)
        gap = abs(float(values[index + 1] - values[index])) / scale
        if gap > float(relative_gap):
            starts.append(index + 1)
    starts.append(values.size)
    return [slice(starts[i], starts[i + 1]) for i in range(len(starts) - 1)]


def _subspace_projector_distance(
    previous_basis: np.ndarray,
    current_basis: np.ndarray,
) -> float:
    """Return ``||UU^T - VV^T||_2`` without forming full projectors.

    For equal-dimensional orthonormal bases ``U`` and ``V``, the projector
    distance equals the sine of the largest principal angle:

    ``sqrt(1 - sigma_min(U.T @ V)**2)``.

    The overlap matrix has the cluster dimension rather than the ambient
    dimension, which avoids an ambient-size SVD for every spectral cluster.
    """
    previous_work = np.asarray(previous_basis, dtype=float)
    current_work = np.asarray(current_basis, dtype=float)
    if previous_work.ndim != 2 or current_work.ndim != 2:
        raise ValueError("subspace bases must be two-dimensional")
    if previous_work.shape != current_work.shape:
        raise ValueError("subspace bases must have the same shape")
    if previous_work.shape[1] == 0:
        return 0.0

    overlap = previous_work.T @ current_work
    singular_values = np.linalg.svd(overlap, compute_uv=False)
    sigma_min = float(np.clip(np.min(singular_values), 0.0, 1.0))
    return math.sqrt(max(0.0, 1.0 - sigma_min * sigma_min))


def _projector_distance_for_clusters(
    previous: np.ndarray,
    current: np.ndarray,
    relative_gap: float,
) -> tuple[float, bool, str]:
    previous_values, previous_vectors = np.linalg.eigh(
        0.5 * (previous + previous.T)
    )
    current_values, current_vectors = np.linalg.eigh(
        0.5 * (current + current.T)
    )
    previous_clusters = _cluster_slices(previous_values, relative_gap)
    current_clusters = _cluster_slices(current_values, relative_gap)
    previous_sizes = [item.stop - item.start for item in previous_clusters]
    current_sizes = [item.stop - item.start for item in current_clusters]
    if previous_sizes != current_sizes:
        return float("inf"), False, "spectral_cluster_partition_changed"

    distances = [
        _subspace_projector_distance(
            previous_vectors[:, previous_slice],
            current_vectors[:, current_slice],
        )
        for previous_slice, current_slice in zip(
            previous_clusters,
            current_clusters,
        )
    ]
    return (max(distances, default=0.0), True, "")


def compare_partial_trace_artifacts(
    previous: str | Path | Mapping[str, Any],
    current: str | Path | Mapping[str, Any],
    thresholds: PartialTraceStabilityThresholds | None = None,
) -> dict[str, float | bool | str]:
    """Compare consecutive artifacts and return fail-closed geometry checks."""
    thresholds = thresholds or PartialTraceStabilityThresholds()
    previous_data = (
        load_partial_trace_artifact(previous)
        if isinstance(previous, (str, Path))
        else dict(previous)
    )
    current_data = (
        load_partial_trace_artifact(current)
        if isinstance(current, (str, Path))
        else dict(current)
    )
    if str(previous_data.get("block_name")) != str(current_data.get("block_name")):
        raise ValueError("partial-trace artifacts belong to different blocks")

    left_previous = np.asarray(previous_data["left"], dtype=float)
    left_current = np.asarray(current_data["left"], dtype=float)
    right_previous = np.asarray(previous_data["right"], dtype=float)
    right_current = np.asarray(current_data["right"], dtype=float)

    left_matrix_change = _relative_frobenius(left_current, left_previous)
    right_matrix_change = _relative_frobenius(right_current, right_previous)
    max_matrix_change = max(left_matrix_change, right_matrix_change)
    matrix_stable = max_matrix_change <= thresholds.matrix_relative_tolerance

    left_projector, left_partition_ok, left_partition_reason = _projector_distance_for_clusters(
        left_previous, left_current, thresholds.cluster_relative_gap
    )
    right_projector, right_partition_ok, right_partition_reason = _projector_distance_for_clusters(
        right_previous, right_current, thresholds.cluster_relative_gap
    )
    max_projector = max(left_projector, right_projector)
    subspace_stable = (
        left_partition_ok
        and right_partition_ok
        and max_projector <= thresholds.subspace_projector_tolerance
    )

    intervention_changes = {
        key: _relative_frobenius(
            np.asarray(current_data[key], dtype=float),
            np.asarray(previous_data[key], dtype=float),
        )
        for key in (
            "aligned_left",
            "aligned_right",
            "reversed_left",
            "reversed_right",
        )
    }
    max_intervention_change = max(intervention_changes.values(), default=float("inf"))
    intervention_stable = (
        max_intervention_change
        <= thresholds.intervention_factor_relative_tolerance
    )

    previous_negative_left = _negative_spectral_mass(left_previous)
    previous_negative_right = _negative_spectral_mass(right_previous)
    current_negative_left = _negative_spectral_mass(left_current)
    current_negative_right = _negative_spectral_mass(right_current)
    negative_change = max(
        abs(current_negative_left - previous_negative_left),
        abs(current_negative_right - previous_negative_right),
    )
    psd_checks_passed = (
        current_negative_left <= thresholds.maximum_negative_mass
        and current_negative_right <= thresholds.maximum_negative_mass
        and negative_change <= thresholds.negative_mass_change_tolerance
    )

    checks_passed = bool(
        psd_checks_passed
        and matrix_stable
        and subspace_stable
        and intervention_stable
    )
    reasons: list[str] = []
    if not psd_checks_passed:
        reasons.append("negative_spectral_mass")
    if not matrix_stable:
        reasons.append("matrix_change")
    if not subspace_stable:
        reasons.append("subspace_change")
    if not intervention_stable:
        reasons.append("intervention_change")
    for reason in (left_partition_reason, right_partition_reason):
        if reason and reason not in reasons:
            reasons.append(reason)

    return {
        "partial_trace_psd_checks_passed": bool(psd_checks_passed),
        "partial_trace_matrix_stable": bool(matrix_stable),
        "partial_trace_subspace_stable": bool(subspace_stable),
        "intervention_factors_stable": bool(intervention_stable),
        "partial_trace_checks_passed": checks_passed,
        "partial_trace_stability_reasons": ",".join(reasons),
        "left_matrix_relative_change": left_matrix_change,
        "right_matrix_relative_change": right_matrix_change,
        "max_matrix_relative_change": max_matrix_change,
        "left_subspace_projector_distance": left_projector,
        "right_subspace_projector_distance": right_projector,
        "max_subspace_projector_distance": max_projector,
        "max_intervention_factor_relative_change": max_intervention_change,
        "negative_mass_left": current_negative_left,
        "negative_mass_right": current_negative_right,
        "negative_mass_max_change": negative_change,
        **{f"{key}_relative_change": value for key, value in intervention_changes.items()},
    }
