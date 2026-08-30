from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from visible_curvature.aggregate import aggregate_frozen_runs
from visible_curvature.figures import make_all_frozen_figures
from visible_curvature.paper_export import DEBUG_WATERMARK, export_paper_assets


def _write_run(root: Path, seed: int, *, scientific: bool = True, failed: bool = False, environment: str = "environment-shared") -> Path:
    run = root / f"run{seed}"
    run.mkdir(parents=True)
    protocol = "protocol-shared"
    runtime = "runtime-shared"
    common = {
        "schema_version": "1.0",
        "config_hash": f"config-{seed}",
        "protocol_hash": protocol,
        "seed": seed,
        "block_name": "model.decoder.layers.0.self_attn.q_proj.weight",
        "block_type": "attn_q",
    }
    pd.DataFrame([
        {
            **common,
            "covariance_moment": "centered",
            "assignment": "observed",
            "alpha": 0.25,
            "delta_g": 0.4 + 0.01 * seed,
            "delta_g_predicted": 0.3,
            "K_adam": 10.0,
            "K_shampoo": 6.7,
            "endpoint_numerically_reliable": True,
            "ordering_inferentially_reliable": True,
            "reliable_ordering": "positive",
        },
        {
            **common,
            "covariance_moment": "uncentered",
            "assignment": "observed",
            "alpha": 0.25,
            "delta_g": 0.2,
            "K_adam": 10.0,
            "K_shampoo": 8.2,
            "endpoint_numerically_reliable": True,
            "ordering_inferentially_reliable": False,
            "reliable_ordering": "inconclusive",
        },
    ]).to_csv(run / "block_metrics.csv", index=False)
    pd.DataFrame([
        {**common, "covariance_moment": "centered", "assignment": a, "alpha": 0.25,
         "damping_coefficient": 0.001, "delta_g": value}
        for a, value in [("observed", 0.4), ("aligned", 0.8), ("reversed", -0.5)]
    ]).to_csv(run / "interventions.csv", index=False)
    pd.DataFrame([
        {**common, "covariance_moment": "centered", "assignment": a, "alpha": alpha,
         "damping_coefficient": 0.001, "delta_g": value * (2 if alpha == 0.5 else 1)}
        for a, value in [("observed", 0.4), ("aligned", 0.8), ("reversed", -0.5)]
        for alpha in (0.25, 0.5)
    ]).to_csv(run / "alpha_sweep.csv", index=False)
    pd.DataFrame([
        {**common, "covariance_moment": "centered", "assignment": a, "alpha": 0.25,
         "damping_coefficient": coefficient, "delta_g": value / (1.0 + coefficient)}
        for a, value in [("observed", 0.4), ("aligned", 0.8), ("reversed", -0.5)]
        for coefficient in (0.0, 0.1, 1.0)
    ]).to_csv(run / "damping_sweep.csv", index=False)
    pd.DataFrame([
        {**common, "replicate": rep, "covariance_moment": "centered", "assignment": "observed",
         "alpha": 0.25, "delta_g": 0.35 + 0.01 * rep}
        for rep in range(4)
    ]).to_csv(run / "bootstrap_metrics.csv", index=False)
    pd.DataFrame(
        [{**common, "error_type": "RuntimeError", "error_message": "boom"}] if failed else [],
        columns=[*common.keys(), "error_type", "error_message"],
    ).to_csv(run / "block_failures.csv", index=False)
    manifest = {
        "schema_version": "1.0",
        "status": "complete",
        "config_hash": f"config-{seed}",
        "protocol_hash": protocol,
        "runtime_identity_sha256": runtime,
        "runtime_environment_sha256": environment,
        "scientific_run": scientific,
        "synthetic_backend": not scientific,
        "experiment_tier": "confirmatory" if scientific else "debug",
        "num_failed_blocks": int(failed),
        "model": {"resolved_model_commit": "a" * 40},
        "data": {
            "dataset_revision": "b" * 40,
            "tokenizer_revision": "a" * 40,
            "source_order_sha256": "source",
            "packed_token_stream_sha256": "packed",
            "selected_chunk_content_sha256": "selected",
        },
    }
    (run / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return run



def test_aggregate_rejects_incompatible_runtime_environments(tmp_path: Path):
    runs = [
        _write_run(tmp_path, 0, environment="environment-a"),
        _write_run(tmp_path, 1, environment="environment-b"),
    ]
    with pytest.raises(ValueError, match="runtime environments"):
        aggregate_frozen_runs(runs, tmp_path / "aggregate", minimum_seed_count=2)

def test_aggregate_preserves_focused_tables_and_seed_consensus(tmp_path: Path):
    runs = [_write_run(tmp_path, seed) for seed in range(3)]
    output = aggregate_frozen_runs(runs, tmp_path / "aggregate", minimum_seed_count=3)
    manifest = json.loads((output / "aggregate_manifest.json").read_text())
    assert manifest["source_manifests_present"] is True
    assert manifest["compatible_protocol_hashes"] is True
    assert manifest["compatible_runtime_identities"] is True
    assert manifest["no_block_failures"] is True
    assert manifest["minimum_seed_count_met"] is True
    summary = pd.read_csv(output / "paired_seed_summary.csv")
    primary = summary.query("covariance_moment == 'centered' and assignment == 'observed' and alpha == 0.25")
    assert len(primary) == 1
    assert primary.iloc[0]["reliable_ordering"] == "positive"
    assert primary.iloc[0]["n_seeds"] == 3


def test_figures_cover_only_the_four_retained_empirical_questions(tmp_path: Path):
    runs = [_write_run(tmp_path, seed) for seed in range(3)]
    output = aggregate_frozen_runs(runs, tmp_path / "aggregate", minimum_seed_count=3)
    figures = make_all_frozen_figures(output, tmp_path / "figures")
    assert {path.name for path in figures} == {
        "block_signed_gain.pdf",
        "assignment_intervention.pdf",
        "alpha_response.pdf",
        "damping_attenuation.pdf",
    }
    assert all(path.exists() and path.stat().st_size > 0 for path in figures)


def test_scientific_export_is_fail_closed_and_debug_export_is_watermarked(tmp_path: Path):
    runs = [_write_run(tmp_path, seed) for seed in range(3)]
    aggregate = aggregate_frozen_runs(runs, tmp_path / "aggregate", minimum_seed_count=3)
    with pytest.raises(ValueError, match="balanced_canonical"):
        export_paper_assets(aggregate, tmp_path / "paper")
    legacy_debug_paths = export_paper_assets(
        aggregate, tmp_path / "legacy-debug-paper", allow_debug_export=True
    )
    assert all(
        DEBUG_WATERMARK in path.read_text(encoding="utf-8")
        for path in legacy_debug_paths
    )

    debug_runs = [_write_run(tmp_path / "debug", seed, scientific=False) for seed in range(3)]
    debug = aggregate_frozen_runs(debug_runs, tmp_path / "debug-aggregate", minimum_seed_count=3)
    with pytest.raises(ValueError, match="not eligible for scientific paper export"):
        export_paper_assets(debug, tmp_path / "rejected")
    debug_paths = export_paper_assets(debug, tmp_path / "debug-paper", allow_debug_export=True)
    assert all(DEBUG_WATERMARK in path.read_text(encoding="utf-8") for path in debug_paths)


def test_scientific_export_rejects_any_captured_block_failure(tmp_path: Path):
    runs = [_write_run(tmp_path, seed, failed=(seed == 2)) for seed in range(3)]
    aggregate = aggregate_frozen_runs(runs, tmp_path / "aggregate", minimum_seed_count=3)
    with pytest.raises(ValueError, match="no_block_failures"):
        export_paper_assets(aggregate, tmp_path / "paper")
