from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
import torch

from visible_curvature.partial_trace_stability import (
    PartialTraceStabilityThresholds,
    compare_partial_trace_artifacts,
    load_partial_trace_artifact,
    save_partial_trace_artifact,
)


def _rotation(theta: float) -> torch.Tensor:
    return torch.tensor(
        [[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]],
        dtype=torch.float64,
    )


def _save(
    root: Path,
    name: str,
    left: torch.Tensor,
    right: torch.Tensor | None = None,
) -> Path:
    right = left if right is None else right
    covariance_left = torch.diag(
        torch.linspace(1.0, 2.0, left.shape[0], dtype=torch.float64)
    )
    covariance_right = torch.diag(
        torch.linspace(1.0, 2.0, right.shape[0], dtype=torch.float64)
    )
    return save_partial_trace_artifact(
        root,
        block_name=name,
        left=left,
        right=right,
        covariance_left=covariance_left,
        covariance_right=covariance_right,
    )


def test_rotated_well_separated_eigenspaces_fail_subspace_check(tmp_path: Path):
    previous_matrix = torch.diag(torch.tensor([1.0, 4.0], dtype=torch.float64))
    rotation = _rotation(math.pi / 4.0)
    current_matrix = rotation @ previous_matrix @ rotation.T
    previous = _save(tmp_path / "previous", "block", previous_matrix)
    current = _save(tmp_path / "current", "block", current_matrix)

    result = compare_partial_trace_artifacts(
        previous,
        current,
        PartialTraceStabilityThresholds(
            matrix_relative_tolerance=2.0,
            subspace_projector_tolerance=0.2,
            intervention_factor_relative_tolerance=2.0,
        ),
    )

    assert result["partial_trace_subspace_stable"] is False
    assert result["partial_trace_checks_passed"] is False
    assert result["max_subspace_projector_distance"] > 0.5


def test_rotation_inside_degenerate_cluster_is_accepted(tmp_path: Path):
    previous_matrix = torch.diag(torch.tensor([2.0, 2.0, 5.0], dtype=torch.float64))
    theta = math.pi / 3.0
    block_rotation = torch.eye(3, dtype=torch.float64)
    block_rotation[:2, :2] = _rotation(theta)
    current_matrix = block_rotation @ previous_matrix @ block_rotation.T
    previous = _save(tmp_path / "previous", "block", previous_matrix)
    current = _save(tmp_path / "current", "block", current_matrix)

    result = compare_partial_trace_artifacts(
        previous,
        current,
        PartialTraceStabilityThresholds(cluster_relative_gap=1.0e-6),
    )

    assert result["partial_trace_subspace_stable"] is True
    assert result["partial_trace_matrix_stable"] is True
    assert result["partial_trace_checks_passed"] is True


def test_artifact_roundtrip_records_block_and_interventions(tmp_path: Path):
    matrix = torch.diag(torch.tensor([1.0, 3.0], dtype=torch.float64))
    path = _save(tmp_path, "layer.0.weight", matrix)
    artifact = load_partial_trace_artifact(path)
    assert artifact["block_name"] == "layer.0.weight"
    assert artifact["left"].shape == (2, 2)
    assert artifact["aligned_left"].shape == (2, 2)
    assert artifact["reversed_right"].shape == (2, 2)
    index = (tmp_path / "index.json").read_text(encoding="utf-8")
    assert "layer.0.weight" in index


def test_clustered_subspace_distance_avoids_full_projector_spectral_norm(monkeypatch):
    from visible_curvature.partial_trace_stability import (
        _projector_distance_for_clusters,
    )

    previous = np.diag(np.arange(1.0, 9.0))
    theta = 0.2
    rotation = np.eye(8)
    rotation[:2, :2] = np.array(
        [
            [math.cos(theta), -math.sin(theta)],
            [math.sin(theta), math.cos(theta)],
        ]
    )
    current = rotation @ previous @ rotation.T

    original_norm = np.linalg.norm

    def guarded_norm(value, ord=None, *args, **kwargs):
        array = np.asarray(value)
        if ord == 2 and array.ndim == 2 and array.shape == (8, 8):
            raise AssertionError(
                "cluster comparison formed a full-size projector spectral norm"
            )
        return original_norm(value, ord=ord, *args, **kwargs)

    monkeypatch.setattr(np.linalg, "norm", guarded_norm)
    distance, partition_ok, reason = _projector_distance_for_clusters(
        previous,
        current,
        relative_gap=1.0e-8,
    )

    assert partition_ok is True
    assert reason == ""
    assert math.isfinite(distance)


def test_overlap_formula_matches_explicit_projector_distance():
    from visible_curvature.partial_trace_stability import (
        _projector_distance_for_clusters,
    )

    rng = np.random.default_rng(7)
    previous_basis, _ = np.linalg.qr(rng.normal(size=(6, 6)))
    current_basis, _ = np.linalg.qr(rng.normal(size=(6, 6)))
    values = np.array([1.0, 1.0, 3.0, 3.0, 8.0, 8.0])
    previous = previous_basis @ np.diag(values) @ previous_basis.T
    current = current_basis @ np.diag(values) @ current_basis.T

    actual, partition_ok, reason = _projector_distance_for_clusters(
        previous,
        current,
        relative_gap=1.0e-8,
    )

    expected = 0.0
    for cluster in (slice(0, 2), slice(2, 4), slice(4, 6)):
        left = previous_basis[:, cluster]
        right = current_basis[:, cluster]
        expected = max(
            expected,
            float(np.linalg.norm(left @ left.T - right @ right.T, ord=2)),
        )

    assert partition_ok is True
    assert reason == ""
    assert actual == pytest.approx(expected, abs=1.0e-10)
