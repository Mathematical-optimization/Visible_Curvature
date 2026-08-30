from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from visible_curvature.reliability_balanced import annotate_final_outputs


def _policy() -> dict:
    return {
        "primary": {
            "covariance_moment": "centered",
            "assignment": "observed",
            "alpha": 0.25,
            "sweep_mode": "primary",
        },
        "reliability": {
            "bootstrap_minimum_finite_reps": 2,
            "k_relative_change_tolerance": 0.05,
            "delta_g_absolute_change_tolerance": 0.05,
        },
    }


def _certificates() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "block_name": "layer.weight",
                "adaptive_endpoint_certified": True,
                "adaptive_partial_trace_certified": True,
                "partial_trace_checks_passed": True,
                "selected_K_adam": 10.0,
                "selected_K_shampoo": 5.0,
                "selected_delta_g": 0.6931471805599453,
                "selected_condition_metric": "ordinary",
                "selected_fallback_tau": 1.0e-4,
                "condition_metric_consistent": True,
                "fallback_tau_consistent": True,
                "selected_endpoint_steps": 64,
                "selected_endpoint_starts": 2,
                "selected_partial_trace_probes": 32,
            }
        ]
    )


def _stage_rows() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "block_name": "layer.weight",
                "stage_index": 1,
                "stage_label": "selected",
                "partial_trace_probes": 32,
                "negative_mass_left": 0.0,
                "negative_mass_right": 0.0,
                "output_dir": "diagnostic",
            }
        ]
    )


def _write_final_tables(final_dir: Path, *, endpoint_reliable: bool) -> None:
    final_dir.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "block_name": "layer.weight",
                "covariance_moment": "centered",
                "assignment": "observed",
                "alpha": 0.25,
                "K_adam": 10.0,
                "K_shampoo": 5.0,
                "delta_g": 0.6931471805599453,
                "condition_metric": "ordinary",
                "fallback_tau": 1.0e-4,
                "endpoint_numerically_reliable": endpoint_reliable,
                "bootstrap_ci_low": 0.2,
                "bootstrap_ci_high": 1.0,
                "bootstrap_reps_finite": 4,
            },
            {
                "block_name": "layer.weight",
                "covariance_moment": "uncentered",
                "assignment": "observed",
                "alpha": 0.25,
                "K_adam": 9.0,
                "K_shampoo": 6.0,
                "delta_g": 0.4054651081081644,
                "condition_metric": "ordinary",
                "fallback_tau": 1.0e-4,
                "endpoint_numerically_reliable": True,
                "bootstrap_ci_low": float("nan"),
                "bootstrap_ci_high": float("nan"),
                "bootstrap_reps_finite": 0,
            },
        ]
    ).to_csv(final_dir / "block_metrics.csv", index=False)
    pd.DataFrame(
        [
            {
                "block_name": "layer.weight",
                "covariance_moment": "centered",
                "assignment": "observed",
                "alpha": 0.25,
                "sweep_mode": "primary",
                "tau": 1.0e-3,
                "delta_g": 0.69,
            },
            {
                "block_name": "layer.weight",
                "covariance_moment": "centered",
                "assignment": "observed",
                "alpha": 0.25,
                "sweep_mode": "primary",
                "tau": 1.0e-4,
                "delta_g": 0.70,
            },
        ]
    ).to_csv(final_dir / "spectral_gain_curve.csv", index=False)
    for name in ("interventions.csv", "alpha_sweep.csv", "damping_sweep.csv"):
        pd.DataFrame(
            [
                {
                    "block_name": "layer.weight",
                    "assignment": "observed",
                    "condition_metric": "ordinary",
                    "fallback_tau": 1.0e-4,
                    "endpoint_numerically_reliable": True,
                },
                {
                    "block_name": "layer.weight",
                    "assignment": "aligned",
                    "condition_metric": "ordinary",
                    "fallback_tau": 1.0e-4,
                    "endpoint_numerically_reliable": True,
                },
            ]
        ).to_csv(final_dir / name, index=False)


def test_final_endpoint_failure_overrides_adaptive_acceptance(tmp_path: Path):
    final_dir = tmp_path / "final"
    _write_final_tables(final_dir, endpoint_reliable=False)

    summary = annotate_final_outputs(
        final_dir, _stage_rows(), _certificates(), _policy()
    )

    row = pd.read_csv(final_dir / "canonical_block_metrics.csv").iloc[0]
    assert not bool(row["balanced_primary_reliable"])
    assert "final_endpoint" in row["balanced_reliability_reasons"]
    assert summary["scientific_status"] == "inconclusive"


def test_canonical_table_contains_only_primary_rows_and_status(tmp_path: Path):
    final_dir = tmp_path / "final"
    _write_final_tables(final_dir, endpoint_reliable=True)

    summary = annotate_final_outputs(
        final_dir, _stage_rows(), _certificates(), _policy()
    )

    canonical = pd.read_csv(final_dir / "canonical_block_metrics.csv")
    assert len(canonical) == 1
    assert canonical.iloc[0]["covariance_moment"] == "centered"
    assert bool(canonical.iloc[0]["balanced_primary_reliable"])
    status = json.loads((final_dir / "scientific_status.json").read_text())
    assert status == {
        "pipeline_status": "complete",
        "primary_inference_available": True,
        "scientific_status": "accepted",
    }
    assert summary["scientific_status"] == "accepted"
    assert (final_dir / "canonical_interventions.csv").exists()
