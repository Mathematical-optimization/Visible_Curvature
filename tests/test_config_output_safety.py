from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
import torch
import yaml

from ovc_experiments.config import load_config
from ovc_experiments.hardened_runner import HardenedBlockConfig, analyze_block_streaming
from ovc_experiments.hardened_spectral import ConditionDiagnosticsWriter
from ovc_experiments.io import append_dataframe_row
from ovc_experiments.safe_operators import DiagonalOperator


def test_unknown_nested_config_key_fails_with_full_path(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text(
        yaml.safe_dump({"curvature": {"lancoz_steps": 17}}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=r"curvature\.lancoz_steps"):
        load_config(path)


def test_unknown_root_config_key_fails_with_full_path(tmp_path: Path) -> None:
    path = tmp_path / "bad-root.yaml"
    path.write_text(yaml.safe_dump({"output_directory": "x"}), encoding="utf-8")

    with pytest.raises(ValueError, match=r"output_directory"):
        load_config(path)


def test_append_dataframe_row_preserves_rows_and_merges_schema(tmp_path: Path) -> None:
    path = tmp_path / "geometry.csv"

    append_dataframe_row({"block_name": "a", "K": 3.0}, path)
    append_dataframe_row({"block_name": "b", "K": 4.0, "extra": "kept"}, path)

    frame = pd.read_csv(path)
    assert frame["block_name"].tolist() == ["a", "b"]
    assert frame["K"].tolist() == [3.0, 4.0]
    assert pd.isna(frame.loc[0, "extra"])
    assert frame.loc[1, "extra"] == "kept"


def test_condition_diagnostics_writer_is_append_only_and_strict_json(tmp_path: Path) -> None:
    path = tmp_path / "solver.jsonl"

    writer = ConditionDiagnosticsWriter(path)
    writer.write({"block": "a", "condition": float("nan")})
    ConditionDiagnosticsWriter(path).write({"block": "b", "condition": float("inf")})

    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    payloads = [
        json.loads(
            line,
            parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
        )
        for line in lines
    ]
    assert payloads == [
        {"block": "a", "condition": None},
        {"block": "b", "condition": None},
    ]


def test_multiple_hardened_blocks_append_geometry_and_diagnostics(tmp_path: Path) -> None:
    curvature = DiagonalOperator(torch.tensor([1.0, 2.0], dtype=torch.float64))
    gradients = [
        torch.tensor([[1.0], [2.0]], dtype=torch.float64),
        torch.tensor([[2.0], [1.0]], dtype=torch.float64),
        torch.tensor([[3.0], [4.0]], dtype=torch.float64),
    ]
    config = HardenedBlockConfig(
        centered_moments=False,
        adam_damping=1e-3,
        shampoo_damping_ratio=1e-3,
        exact_condition_max_dim=8,
    )

    for block_name in ("a", "b"):
        analyze_block_streaming(
            curvature_operator=curvature,
            gradient_factory=lambda: iter(gradients),
            rows=2,
            cols=1,
            example_count=len(gradients),
            config=config,
            output_dir=tmp_path,
            metadata={"block_name": block_name},
        )

    geometry = pd.read_csv(tmp_path / "geometry.csv")
    diagnostics = (tmp_path / "solver_diagnostics.jsonl").read_text(encoding="utf-8").splitlines()
    assert geometry["block_name"].tolist() == ["a", "b"]
    assert len(diagnostics) == 6


def test_hardened_streaming_censors_invalid_adam_without_aborting_block(tmp_path: Path) -> None:
    curvature = DiagonalOperator(torch.tensor([1.0, 2.0], dtype=torch.float64))
    # The second coordinate is identically zero, so zero-damped centered Adam is singular.
    gradients = [
        torch.tensor([[1.0], [0.0]], dtype=torch.float64),
        torch.tensor([[2.0], [0.0]], dtype=torch.float64),
        torch.tensor([[3.0], [0.0]], dtype=torch.float64),
    ]

    result = analyze_block_streaming(
        curvature_operator=curvature,
        gradient_factory=lambda: iter(gradients),
        rows=2,
        cols=1,
        example_count=len(gradients),
        config=HardenedBlockConfig(
            centered_moments=True,
            adam_damping=0.0,
            shampoo_damping_ratio=1e-3,
            exact_condition_max_dim=8,
        ),
        output_dir=tmp_path,
        metadata={"block_name": "singular-adam"},
    )

    assert result.row["adam_censored"] is True
    assert str(result.row["adam_censor_reason"]).startswith("invalid_preconditioner:")
    assert pd.isna(result.row["K_adam"])
    assert result.row["curvature_censored"] is False


def test_hardened_streaming_censors_unresolved_shampoo_damping_without_aborting(
    tmp_path: Path,
) -> None:
    curvature = DiagonalOperator(torch.tensor([1.0, 2.0], dtype=torch.float64))
    # Identical gradients have zero centered row/column covariance, so a
    # factor-normalized damping ratio has no resolved positive reference scale.
    gradients = [
        torch.tensor([[1.0], [2.0]], dtype=torch.float64),
        torch.tensor([[1.0], [2.0]], dtype=torch.float64),
        torch.tensor([[1.0], [2.0]], dtype=torch.float64),
    ]

    result = analyze_block_streaming(
        curvature_operator=curvature,
        gradient_factory=lambda: iter(gradients),
        rows=2,
        cols=1,
        example_count=len(gradients),
        config=HardenedBlockConfig(
            centered_moments=True,
            adam_damping=1e-3,
            shampoo_damping_ratio=1e-3,
            exact_condition_max_dim=8,
        ),
        output_dir=tmp_path,
        metadata={"block_name": "zero-centered-factors"},
    )

    assert result.row["curvature_censored"] is False
    assert result.row["shampoo_censored"] is True
    assert str(result.row["shampoo_censor_reason"]).startswith(
        "invalid_factor_damping:"
    )
    assert pd.isna(result.row["K_shampoo"])
