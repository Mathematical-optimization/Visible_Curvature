from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from visible_curvature.analysis_runner import run_frozen_analysis
from visible_curvature.interventions import reassign_factor


def _config(tmp_path: Path) -> dict:
    return {
        "seed": 3,
        "deterministic": True,
        "device": "cpu",
        "scientific_run": False,
        "output_dir": str(tmp_path / "focused"),
        "model": {
            "backend": "tiny_causal_lm",
            "dtype": "float32",
            "vocab_size": 24,
            "d_model": 4,
            "n_heads": 1,
            "hidden": 6,
            "n_layers": 1,
        },
        "data": {
            "backend": "synthetic_tokens",
            "content_seed": 11,
            "order_seed": 17,
            "num_examples": 12,
            "batch_size": 2,
            "seq_len": 5,
            "vocab_size": 24,
            "shuffle": False,
        },
        "blocks": {
            "include": [r"layers\.0\.attn\.q_proj\.weight"],
            "max_blocks": 1,
        },
        "analysis": {
            "compute_tier": "debug",
            "assignments": ["observed", "aligned", "reversed"],
            "covariance": {
                "skip_batches": 0,
                "num_batches": 4,
                "group_size": 2,
                "ddof": 0,
            },
            "curvature": {
                "kind": "ggn",
                "skip_batches": 4,
                "num_batches": 1,
                "dtype": "float32",
                "psd_mode": "shift",
                "ridge": 1.0e-4,
                "ridge_mode": "relative_max",
                "stabilize_lanczos_steps": 4,
                "stabilize_lanczos_starts": 1,
                "stabilize_rounds": 1,
                "lanczos_steps": 5,
                "lanczos_starts": 1,
                "partial_trace_probes": 3,
            },
            "condition": {
                "relative_floor": 1.0e-7,
                "fallback_tau": 1.0e-4,
                "tau_sweep": [1.0e-3, 1.0e-4],
            },
            "preconditioners": {
                "adam": {
                    "damping_coefficient": 1.0e-3,
                    "damping_statistic": "median",
                    "damping_min": 1.0e-8,
                },
                "shampoo": {
                    "damping_coefficient": 1.0e-3,
                    "damping_statistic": "lambda_max",
                    "damping_min": 1.0e-8,
                    "factor_exponent": 0.25,
                    "eig_floor": 0.0,
                    "relative_eig_floor": 64.0,
                },
            },
            "factor_diagnostics": {
                "max_full_eigh_dim": 64,
                "approx_eig_k": 8,
                "curvature_rel_floor": 1.0e-6,
            },
            "bootstrap": {
                "reps": 2,
                "diagnostics": "delta_only",
                "lanczos_steps": 3,
                "lanczos_starts": 1,
                "alpha": 0.1,
                "minimum_reps": 2,
            },
            "interventions": {"enabled": True},
            "alpha_sweep": {"enabled": True, "values": [0.25, 0.5]},
            "damping_sweep": {"enabled": True, "coefficients": [0.0, 0.1]},
            "reliability": {
                "max_shift_ratio": 1.0,
                "max_min_ritz_residual_over_min": 1.0,
                "max_max_ritz_residual_over_max": 1.0,
                "min_r2": -1.0,
                "max_commutator": 2.0,
                "max_factor_negative_mass": 1.0,
                "max_factor_eigen_residual": 1.0,
            },
        },
    }


def test_focused_runner_writes_only_required_experiment_tables(tmp_path):
    output = run_frozen_analysis(_config(tmp_path))
    required = {
        "block_metrics.csv",
        "bootstrap_metrics.csv",
        "interventions.csv",
        "alpha_sweep.csv",
        "damping_sweep.csv",
        "block_failures.csv",
        "run_manifest.json",
        "summary.json",
        "resolved_config.yaml",
    }
    assert required.issubset({path.name for path in output.iterdir()})
    assert not (output / "frozen_trajectories.csv").exists()

    blocks = pd.read_csv(output / "block_metrics.csv")
    assert set(blocks["covariance_moment"]) == {"centered", "uncentered"}
    assert set(blocks["assignment"]) == {"observed"}
    assert set(blocks["alpha"]) == {0.25}
    for column in (
        "delta_g",
        "endpoint_numerically_reliable",
        "ordering_inferentially_reliable",
        "reliable_ordering",
        "adam_condition_saturated",
        "shampoo_condition_saturated",
    ):
        assert column in blocks.columns
    assert not any(column.lower().startswith("nci_") for column in blocks.columns)

    interventions = pd.read_csv(output / "interventions.csv")
    assert set(interventions["assignment"]) == {"observed", "aligned", "reversed"}
    alpha = pd.read_csv(output / "alpha_sweep.csv")
    assert set(alpha["alpha"]) == {0.25, 0.5}
    assert set(alpha["assignment"]) == {"observed", "aligned", "reversed"}
    damping = pd.read_csv(output / "damping_sweep.csv")
    assert set(damping["damping_coefficient"]) == {0.0, 0.1}

    bootstrap = pd.read_csv(output / "bootstrap_metrics.csv")
    assert set(bootstrap["covariance_moment"]) == {"centered"}
    assert set(bootstrap["assignment"]) == {"observed"}
    assert set(bootstrap["alpha"]) == {0.25}
    assert not any(column.startswith("r_") for column in bootstrap.columns)

    manifest = json.loads((output / "run_manifest.json").read_text())
    assert manifest["status"] == "complete"
    assert manifest["data"]["order_seed"] == 17


def test_removed_intervention_modes_are_rejected():
    import torch

    factor = torch.eye(2)
    for mode in ("random", "scrambled"):
        with pytest.raises(ValueError, match="observed, aligned, or reversed"):
            reassign_factor(factor, factor, mode=mode)


def test_csv_writer_emits_declared_header_for_empty_rows(tmp_path):
    from visible_curvature.analysis_runner import _write_csv

    path = tmp_path / "empty.csv"
    _write_csv(path, [], columns=["block_name", "delta_g"])
    frame = pd.read_csv(path)
    assert list(frame.columns) == ["block_name", "delta_g"]
    assert frame.empty
