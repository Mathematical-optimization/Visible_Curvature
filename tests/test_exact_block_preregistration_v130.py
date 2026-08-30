from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from visible_curvature.analysis_runner import _controls_enabled_for_block
from visible_curvature.config import validate_config


ROOT = Path(__file__).resolve().parents[1]


def _confirmatory() -> dict:
    return yaml.safe_load(
        (ROOT / "configs" / "hf_opt125m_confirmatory.yaml").read_text(
            encoding="utf-8"
        )
    )


def test_confirmatory_config_preregisters_twelve_exact_blocks_and_control_subset():
    validated = validate_config(_confirmatory(), mode="frozen")
    exact = validated["blocks"]["exact_names"]
    controls = validated["analysis"]["controls"]["block_names"]
    assert len(exact) == 12
    assert len(set(exact)) == 12
    assert validated["blocks"]["max_blocks"] == 12
    assert all(pattern.startswith("^") and pattern.endswith("$") for pattern in validated["blocks"]["include"])
    assert set(controls).issubset(exact)
    assert len(controls) == 4


def test_scientific_confirmatory_rejects_unanchored_regex_selection():
    cfg = _confirmatory()
    cfg["blocks"].pop("exact_names", None)
    cfg["blocks"]["include"] = [r"model\\.decoder\\.layers\\..*"]
    with pytest.raises(ValueError, match="exact_names|anchored"):
        validate_config(cfg, mode="frozen")


def test_control_subset_must_be_part_of_exact_primary_blocks():
    cfg = _confirmatory()
    cfg["analysis"]["controls"]["block_names"] = ["not.a.real.block"]
    with pytest.raises(ValueError, match="control.*subset"):
        validate_config(cfg, mode="frozen")


def test_control_subset_limits_expensive_controls_without_suppressing_primary_blocks():
    analysis = {"controls": {"block_names": ["block.a", "block.c"]}}
    assert _controls_enabled_for_block(analysis, "block.a") is True
    assert _controls_enabled_for_block(analysis, "block.b") is False
    assert _controls_enabled_for_block({"controls": {"block_names": []}}, "block.b") is True
