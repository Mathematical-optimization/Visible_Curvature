from __future__ import annotations
import pandas as pd
from visible_curvature.reliability_balanced import (
    ReliabilityThresholds,
    build_stage_specs,
    certify_convergence,
    prepare_core_config,
    tau_sign_reason,
)


def test_stage_schedule_extends_shorter_probe_schedule():
    specs = build_stage_specs({"reliability": {
        "endpoint_steps": [64, 96, 128, 192, 256],
        "endpoint_starts": [2, 2, 2, 3, 4],
        "partial_trace_probes": [32, 64, 128, 256],
    }})
    assert [s.endpoint_steps for s in specs] == [64, 96, 128, 192, 256]
    assert [s.partial_trace_probes for s in specs] == [32, 64, 128, 256, 256]


def test_prepare_diagnostic_config_disables_expensive_controls():
    base = {
        "scientific_run": True,
        "output_dir": "old",
        "analysis": {
            "compute_tier": "confirmatory",
            "curvature": {
                "lanczos_steps": 64,
                "lanczos_starts": 2,
                "partial_trace_probes": 32,
                "stabilize_lanczos_steps": 16,
                "stabilize_lanczos_starts": 1,
                "ridge": 1e-6,
                "ridge_mode": "absolute",
            },
            "bootstrap": {"reps": 100, "minimum_reps": 100},
            "interventions": {"enabled": True},
            "alpha_sweep": {"enabled": True},
            "damping_sweep": {"enabled": True},
        },
    }
    stage = build_stage_specs({"reliability": {"endpoint_steps": [96], "endpoint_starts": [3], "partial_trace_probes": [64]}})[0]
    cfg, changed = prepare_core_config(base, stage=stage, output_dir=__import__('pathlib').Path('new'), diagnostic=True, policy={"reliability": {"fixed_relative_ridge": 1e-5}})
    assert cfg["scientific_run"] is False
    assert cfg["analysis"]["compute_tier"] == "debug"
    assert cfg["analysis"]["curvature"]["lanczos_steps"] == 96
    assert cfg["analysis"]["curvature"]["partial_trace_probes"] == 64
    assert cfg["analysis"]["bootstrap"]["reps"] == 0
    assert cfg["analysis"]["interventions"]["enabled"] is False
    assert cfg["analysis"]["curvature"]["ridge"] == 1e-5
    assert cfg["analysis"]["curvature"]["ridge_mode"] == "relative_max"
    assert changed["analysis.curvature.lanczos_steps"] == 96


def test_endpoint_certificate_requires_native_and_cross_budget_stability():
    rows = pd.DataFrame([
        {"block_name": "b", "stage_index": 0, "stage_label": "s0", "endpoint_steps": 64, "endpoint_starts": 2, "partial_trace_probes": 32,
         "K_adam": 100.0, "K_shampoo": 50.0, "delta_g": 0.693, "native_endpoint_reliable": True, "negative_mass_left": .04, "negative_mass_right": .04},
        {"block_name": "b", "stage_index": 1, "stage_label": "s1", "endpoint_steps": 96, "endpoint_starts": 2, "partial_trace_probes": 64,
         "K_adam": 102.0, "K_shampoo": 51.0, "delta_g": 0.693, "native_endpoint_reliable": True, "negative_mass_left": .03, "negative_mass_right": .03},
    ])
    cert = certify_convergence(rows, ReliabilityThresholds())
    assert bool(cert.loc[0, "adaptive_endpoint_certified"])
    assert bool(cert.loc[0, "adaptive_partial_trace_certified"])


def test_native_failure_cannot_be_overridden_by_stability():
    rows = pd.DataFrame([
        {"block_name": "b", "stage_index": 0, "stage_label": "s0", "endpoint_steps": 64, "endpoint_starts": 2, "partial_trace_probes": 32,
         "K_adam": 100.0, "K_shampoo": 50.0, "delta_g": 0.693, "native_endpoint_reliable": False, "negative_mass_left": .2, "negative_mass_right": .2},
        {"block_name": "b", "stage_index": 1, "stage_label": "s1", "endpoint_steps": 96, "endpoint_starts": 2, "partial_trace_probes": 64,
         "K_adam": 100.0, "K_shampoo": 50.0, "delta_g": 0.693, "native_endpoint_reliable": False, "negative_mass_left": .2, "negative_mass_right": .2},
    ])
    cert = certify_convergence(rows, ReliabilityThresholds())
    assert not bool(cert.loc[0, "adaptive_endpoint_certified"])
    assert not bool(cert.loc[0, "adaptive_partial_trace_certified"])


def test_tau_reason_distinguishes_saturation_from_flip():
    assert tau_sign_reason([0.0, 0.0, 0.2, 0.4]) == "one_sided_with_coarse_saturation"
    assert tau_sign_reason([0.2, -0.1]) == "sign_flip"
    assert tau_sign_reason([0.0, 0.0]) == "all_saturated"
    assert tau_sign_reason([0.2, 0.3]) == "stable_nonzero"


def test_partial_trace_certificate_rejects_rotated_intervention_basis(tmp_path):
    import math
    import torch
    from visible_curvature.partial_trace_stability import save_partial_trace_artifact

    diagonal = torch.diag(torch.tensor([1.0, 4.0], dtype=torch.float64))
    rotation = torch.tensor(
        [[math.cos(math.pi / 4), -math.sin(math.pi / 4)],
         [math.sin(math.pi / 4), math.cos(math.pi / 4)]],
        dtype=torch.float64,
    )
    rotated = rotation @ diagonal @ rotation.T
    covariance = torch.diag(torch.tensor([1.0, 2.0], dtype=torch.float64))
    first = save_partial_trace_artifact(
        tmp_path / "s0", block_name="b", left=diagonal, right=diagonal,
        covariance_left=covariance, covariance_right=covariance,
    )
    second = save_partial_trace_artifact(
        tmp_path / "s1", block_name="b", left=rotated, right=rotated,
        covariance_left=covariance, covariance_right=covariance,
    )
    rows = pd.DataFrame([
        {"block_name": "b", "stage_index": 0, "stage_label": "s0", "endpoint_steps": 64,
         "endpoint_starts": 2, "partial_trace_probes": 32, "K_adam": 100.0,
         "K_shampoo": 50.0, "delta_g": 0.693, "native_endpoint_reliable": True,
         "negative_mass_left": 0.0, "negative_mass_right": 0.0,
         "partial_trace_artifact": str(first)},
        {"block_name": "b", "stage_index": 1, "stage_label": "s1", "endpoint_steps": 96,
         "endpoint_starts": 2, "partial_trace_probes": 64, "K_adam": 100.0,
         "K_shampoo": 50.0, "delta_g": 0.693, "native_endpoint_reliable": True,
         "negative_mass_left": 0.0, "negative_mass_right": 0.0,
         "partial_trace_artifact": str(second)},
    ])
    cert = certify_convergence(
        rows,
        ReliabilityThresholds(
            require_partial_trace_artifacts=True,
            partial_trace_matrix_relative_tolerance=2.0,
            partial_trace_subspace_projector_tolerance=0.2,
            intervention_factor_relative_tolerance=2.0,
        ),
    )
    assert not bool(cert.loc[0, "adaptive_partial_trace_certified"])
    assert not bool(cert.loc[0, "partial_trace_subspace_stable"])
