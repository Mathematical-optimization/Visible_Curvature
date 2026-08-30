from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from visible_curvature.aggregate import aggregate_frozen_runs
from visible_curvature.paper_export import export_paper_assets


def _write_balanced_root(root: Path, seed: int, label: str = "negative") -> Path:
    balanced = root / f"balanced-{seed}"
    final = balanced / "final"
    final.mkdir(parents=True)
    common = {
        "schema_version": "1.0",
        "config_hash": f"config-{seed}",
        "protocol_hash": "protocol-shared",
        "runtime_identity_sha256": "runtime-shared",
        "runtime_environment_sha256": "environment-shared",
        "seed": seed,
        "block_name": "model.layers.0.attn.weight",
        "block_type": "attn",
    }
    pd.DataFrame(
        [
            {
                **common,
                "covariance_moment": "centered",
                "assignment": "observed",
                "alpha": 0.25,
                "delta_g": -0.4,
                "delta_g_predicted": -0.3,
                "delta_g_predicted_consumption": -0.2,
                "delta_g_predicted_full_proxy": -0.3,
                "factor_elasticity_reliable": True,
                "K_adam": 10.0,
                "K_shampoo": 15.0,
                "final_endpoint_numerically_accepted": True,
                "final_diagnostic_agreement": True,
                "balanced_primary_reliable": True,
                "balanced_reliable_ordering": label,
                "condition_metric": "ordinary",
                "fallback_tau": 1.0e-4,
            }
        ]
    ).to_csv(final / "canonical_block_metrics.csv", index=False)
    controls = pd.DataFrame(
        [
            {
                **common,
                "covariance_moment": "centered",
                "assignment": assignment,
                "alpha": 0.25,
                "damping_coefficient": 0.001,
                "delta_g": value,
                "condition_metric": "ordinary",
                "fallback_tau": 1.0e-4,
                "balanced_reliable_for_inference": True,
            }
            for assignment, value in [
                ("observed", -0.4),
                ("aligned", 0.8),
                ("reversed", -0.9),
            ]
        ]
    )
    controls.to_csv(final / "canonical_interventions.csv", index=False)
    pd.concat(
        [controls.assign(alpha=0.25), controls.assign(alpha=0.5, delta_g=controls["delta_g"] * 2)],
        ignore_index=True,
    ).to_csv(final / "canonical_alpha_sweep.csv", index=False)
    pd.concat(
        [
            controls.assign(damping_coefficient=coefficient, delta_g=controls["delta_g"] / (1 + coefficient))
            for coefficient in (0.0, 0.1, 1.0)
        ],
        ignore_index=True,
    ).to_csv(final / "canonical_damping_sweep.csv", index=False)
    pd.DataFrame(columns=[*common.keys(), "replicate", "delta_g"]).to_csv(
        final / "bootstrap_metrics.csv", index=False
    )
    pd.DataFrame(columns=[*common.keys(), "error_type", "error_message"]).to_csv(
        final / "block_failures.csv", index=False
    )
    manifest = {
        "schema_version": "1.0",
        "status": "complete",
        "protocol_hash": "protocol-shared",
        "runtime_identity_sha256": "runtime-shared",
        "runtime_environment_sha256": "environment-shared",
        "scientific_run": True,
        "synthetic_backend": False,
        "experiment_tier": "confirmatory",
        "num_failed_blocks": 0,
        "model": {"resolved_model_commit": "a" * 40},
        "data": {
            "dataset_revision": "b" * 40,
            "tokenizer_revision": "c" * 40,
            "source_order_sha256": "source",
            "selected_chunk_content_sha256": "selected",
        },
    }
    (final / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    balanced_summary = {
        "reliability_mode": "balanced_canonical",
        "pipeline_status": "complete",
        "scientific_status": "accepted",
        "primary_inference_available": True,
    }
    (final / "balanced_reliability_summary.json").write_text(
        json.dumps(balanced_summary), encoding="utf-8"
    )
    (balanced / "balanced_reliability_summary.json").write_text(
        json.dumps(balanced_summary), encoding="utf-8"
    )
    (balanced / "COMPLETED").write_text("complete\n", encoding="utf-8")
    return balanced


def test_aggregate_uses_balanced_ordering_from_canonical_table(tmp_path: Path):
    roots = [_write_balanced_root(tmp_path, seed) for seed in range(3)]
    output = aggregate_frozen_runs(
        roots, tmp_path / "aggregate", minimum_seed_count=3
    )
    manifest = json.loads((output / "aggregate_manifest.json").read_text())
    assert manifest["reliability_mode"] == "balanced_canonical"
    assert manifest["canonical_tables_used"] is True
    assert manifest["all_sources_balanced"] is True
    assert manifest["all_primary_rows_numerically_accepted"] is True
    summary = pd.read_csv(output / "paired_seed_summary.csv")
    assert summary.iloc[0]["reliable_ordering"] == "negative"
    assert (output / "canonical_block_metrics.csv").exists()
    prediction_rows = pd.read_csv(output / "elasticity_prediction_rows.csv")
    prediction_summary = pd.read_csv(output / "elasticity_prediction_summary.csv")
    contrasts = pd.read_csv(output / "paired_control_contrasts.csv")
    assert prediction_rows["prediction_eligible"].all()
    assert set(prediction_summary["predictor_name"]) == {"consumption", "full_proxy"}
    assert {
        "assignment_aligned_minus_reversed",
        "alpha_signed_change_from_practical",
        "damping_change_from_minimum",
    }.issubset(set(contrasts["contrast_type"]))


def test_scientific_export_rejects_legacy_aggregate(tmp_path: Path):
    legacy = tmp_path / "legacy"
    legacy.mkdir()
    (legacy / "aggregate_manifest.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "source_manifests_present": True,
                "compatible_protocol_hashes": True,
                "compatible_runtime_identities": True,
                "no_block_failures": True,
                "scientific_run": True,
                "synthetic_backend": False,
                "immutable_revisions": True,
                "experiment_tier": "confirmatory",
                "minimum_seed_count_met": True,
                "reliability_mode": "legacy",
            }
        ),
        encoding="utf-8",
    )
    pd.DataFrame(
        [
            {
                "block_name": "b",
                "covariance_moment": "centered",
                "assignment": "observed",
                "alpha": 0.25,
                "reliable_ordering": "positive",
                "delta_g_median": 0.2,
                "n_seeds": 3,
            }
        ]
    ).to_csv(legacy / "paired_seed_summary.csv", index=False)
    with pytest.raises(ValueError, match="balanced_canonical"):
        export_paper_assets(legacy, tmp_path / "paper")


def test_balanced_aggregate_is_eligible_for_export(tmp_path: Path):
    roots = [_write_balanced_root(tmp_path, seed) for seed in range(3)]
    output = aggregate_frozen_runs(
        roots, tmp_path / "aggregate", minimum_seed_count=3
    )
    paths = export_paper_assets(output, tmp_path / "paper")
    assert paths
    assert all(path.exists() for path in paths)
    names = {path.name for path in paths}
    assert {
        "elasticity_prediction_autogen.tex",
        "paired_controls_autogen.tex",
    }.issubset(names)
    block_table = (tmp_path / "paper" / "block_gain_autogen.tex").read_text()
    assert "Metric" in block_table
    assert "$K_{\\mathrm{Ad}}$" in block_table
