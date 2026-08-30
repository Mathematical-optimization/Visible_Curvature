from pathlib import Path

import pandas as pd

from visible_curvature.aggregate import _paired_seed_summary
from visible_curvature.reliability_balanced import (
    ReliabilityThresholds,
    annotate_final_outputs,
    certify_convergence,
)


def _stage(metric0="ordinary", metric1="ordinary", tau0=1e-4, tau1=1e-4):
    return pd.DataFrame(
        [
            {
                "block_name": "b",
                "stage_index": 0,
                "stage_label": "s0",
                "endpoint_steps": 64,
                "endpoint_starts": 2,
                "partial_trace_probes": 32,
                "K_adam": 100.0,
                "K_shampoo": 50.0,
                "delta_g": 0.693,
                "native_endpoint_reliable": True,
                "negative_mass_left": 0.0,
                "negative_mass_right": 0.0,
                "condition_metric": metric0,
                "fallback_tau": tau0,
            },
            {
                "block_name": "b",
                "stage_index": 1,
                "stage_label": "s1",
                "endpoint_steps": 96,
                "endpoint_starts": 2,
                "partial_trace_probes": 64,
                "K_adam": 101.0,
                "K_shampoo": 50.5,
                "delta_g": 0.693,
                "native_endpoint_reliable": True,
                "negative_mass_left": 0.0,
                "negative_mass_right": 0.0,
                "condition_metric": metric1,
                "fallback_tau": tau1,
            },
        ]
    )


def test_stage_metric_transition_prevents_endpoint_certification():
    cert = certify_convergence(
        _stage("ordinary", "truncated"),
        ReliabilityThresholds(require_partial_trace_artifacts=False),
    )
    assert not bool(cert.loc[0, "adaptive_endpoint_certified"])
    assert not bool(cert.loc[0, "condition_metric_consistent"])
    assert "condition_metric_changed" in cert.loc[0, "endpoint_certification_reasons"]


def test_truncated_stage_requires_same_fallback_tau():
    cert = certify_convergence(
        _stage("truncated", "truncated", 1e-4, 1e-5),
        ReliabilityThresholds(require_partial_trace_artifacts=False),
    )
    assert not bool(cert.loc[0, "adaptive_endpoint_certified"])
    assert not bool(cert.loc[0, "fallback_tau_consistent"])
    assert "fallback_tau_changed" in cert.loc[0, "endpoint_certification_reasons"]


def test_cross_seed_metric_disagreement_forces_inconclusive_consensus():
    frame = pd.DataFrame(
        [
            {
                "protocol_hash": "p",
                "runtime_identity_sha256": "r",
                "block_name": "b",
                "block_type": "attn_q",
                "covariance_moment": "centered",
                "assignment": "observed",
                "alpha": 0.25,
                "seed": 0,
                "delta_g": 0.4,
                "condition_metric": "ordinary",
                "fallback_tau": 1e-4,
                "balanced_reliable_ordering": "positive",
            },
            {
                "protocol_hash": "p",
                "runtime_identity_sha256": "r",
                "block_name": "b",
                "block_type": "attn_q",
                "covariance_moment": "centered",
                "assignment": "observed",
                "alpha": 0.25,
                "seed": 1,
                "delta_g": 0.5,
                "condition_metric": "truncated",
                "fallback_tau": 1e-4,
                "balanced_reliable_ordering": "positive",
            },
        ]
    )
    summary = _paired_seed_summary(frame, minimum_seed_count=2)
    assert summary.loc[0, "condition_metric_consensus"] == "incompatible"
    assert summary.loc[0, "reliable_ordering"] == "inconclusive"
    assert summary.loc[0, "reliable_ordering_consensus_reason"] == "condition_metric_disagreement"


def _write_final(final_dir: Path, metric: str, tau: float = 1e-4):
    final_dir.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "block_name": "b",
                "covariance_moment": "centered",
                "assignment": "observed",
                "alpha": 0.25,
                "K_adam": 10.0,
                "K_shampoo": 5.0,
                "delta_g": 0.6931471805599453,
                "condition_metric": metric,
                "fallback_tau": tau,
                "endpoint_numerically_reliable": True,
                "bootstrap_ci_low": 0.2,
                "bootstrap_ci_high": 1.0,
                "bootstrap_reps_finite": 4,
            }
        ]
    ).to_csv(final_dir / "block_metrics.csv", index=False)
    pd.DataFrame(
        [
            {
                "block_name": "b",
                "covariance_moment": "centered",
                "assignment": "observed",
                "alpha": 0.25,
                "sweep_mode": "primary",
                "tau": 1e-3,
                "delta_g": 0.69,
            },
            {
                "block_name": "b",
                "covariance_moment": "centered",
                "assignment": "observed",
                "alpha": 0.25,
                "sweep_mode": "primary",
                "tau": 1e-4,
                "delta_g": 0.70,
            },
        ]
    ).to_csv(final_dir / "spectral_gain_curve.csv", index=False)


def _certificate(metric: str, tau: float = 1e-4):
    return pd.DataFrame(
        [
            {
                "block_name": "b",
                "adaptive_endpoint_certified": True,
                "adaptive_partial_trace_certified": True,
                "partial_trace_checks_passed": True,
                "selected_K_adam": 10.0,
                "selected_K_shampoo": 5.0,
                "selected_delta_g": 0.6931471805599453,
                "selected_condition_metric": metric,
                "selected_fallback_tau": tau,
                "condition_metric_consistent": True,
                "fallback_tau_consistent": True,
                "selected_endpoint_steps": 64,
                "selected_endpoint_starts": 2,
                "selected_partial_trace_probes": 32,
            }
        ]
    )


def _policy(allow_truncated: bool):
    return {
        "primary": {"covariance_moment": "centered", "assignment": "observed", "alpha": 0.25},
        "reliability": {
            "bootstrap_minimum_finite_reps": 2,
            "allow_truncated_primary": allow_truncated,
        },
    }


def test_truncated_primary_is_inconclusive_by_default(tmp_path: Path):
    final = tmp_path / "final"
    _write_final(final, "truncated")
    summary = annotate_final_outputs(final, _stage().iloc[-1:], _certificate("truncated"), _policy(False))
    row = pd.read_csv(final / "canonical_block_metrics.csv").iloc[0]
    assert not bool(row["balanced_primary_reliable"])
    assert "truncated_primary_not_allowed" in row["balanced_reliability_reasons"]
    assert summary["scientific_status"] == "inconclusive"


def test_truncated_primary_can_only_be_enabled_explicitly(tmp_path: Path):
    final = tmp_path / "final"
    _write_final(final, "truncated")
    summary = annotate_final_outputs(final, _stage().iloc[-1:], _certificate("truncated"), _policy(True))
    row = pd.read_csv(final / "canonical_block_metrics.csv").iloc[0]
    assert bool(row["balanced_primary_reliable"])
    assert summary["scientific_status"] == "accepted"


def test_control_metric_must_match_selected_primary_metric(tmp_path: Path):
    final = tmp_path / "final"
    _write_final(final, "ordinary")
    pd.DataFrame(
        [
            {
                "block_name": "b",
                "assignment": "observed",
                "condition_metric": "truncated",
                "fallback_tau": 1e-4,
                "endpoint_numerically_reliable": True,
            }
        ]
    ).to_csv(final / "interventions.csv", index=False)
    annotate_final_outputs(final, _stage().iloc[-1:], _certificate("ordinary"), _policy(False))
    control = pd.read_csv(final / "canonical_interventions.csv").iloc[0]
    assert not bool(control["balanced_reliable_for_inference"])
    assert "condition_metric" in control["balanced_control_reliability_reasons"]


def test_paired_seed_summary_preserves_bootstrap_interval_and_tail_consensus():
    import pandas as pd

    from visible_curvature.aggregate import _paired_seed_summary

    rows = []
    for seed in range(3):
        rows.append(
            {
                "protocol_hash": "p",
                "runtime_identity_sha256": "r",
                "block_name": "b",
                "block_type": "attn_q",
                "seed": seed,
                "covariance_moment": "centered",
                "assignment": "observed",
                "alpha": 0.25,
                "delta_g": 0.5,
                "K_adam": 10.0,
                "K_shampoo": 6.0,
                "condition_metric": "ordinary",
                "bootstrap_ci_low": 0.1 + 0.01 * seed,
                "bootstrap_ci_high": 0.8 + 0.01 * seed,
                "tail_localized_ordering": False,
                "balanced_reliable_ordering": "positive",
            }
        )
    summary = _paired_seed_summary(pd.DataFrame(rows), minimum_seed_count=3)
    row = summary.iloc[0]
    assert row["bootstrap_ci_low_median"] == 0.11
    assert row["bootstrap_ci_high_median"] == 0.81
    assert row["tail_localized_consensus"] == "no"
