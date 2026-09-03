from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from ovc_experiments.config import ExperimentConfig, save_config
from ovc_experiments.runners import run_smoke, run_synthetic


def _smoke_config(tmp_path: Path) -> ExperimentConfig:
    cfg = ExperimentConfig()
    cfg.run.name = "pytest-decoder"
    cfg.run.seed = 11
    cfg.run.dtype = "float64"
    cfg.output_dir = str(tmp_path / "outputs")
    cfg.model.family = "decoder"
    cfg.model.kwargs = {
        "vocab_size": 8,
        "max_seq_len": 4,
        "d_model": 4,
        "n_heads": 1,
        "n_layers": 1,
        "mlp_ratio": 1,
        "dropout": 0.0,
    }
    cfg.data.family = "synthetic_language"
    cfg.data.num_examples = 6
    cfg.data.batch_size = 3
    cfg.data.kwargs = {"seq_len": 3, "vocab_size": 8}
    cfg.blocks.include = [r"blocks\.0\.q_proj\.weight$"]
    cfg.blocks.max_numel = 32
    cfg.curvature.kind = "fisher"
    cfg.curvature.shift = 1e-3
    cfg.curvature.exact_max_dim = 64
    cfg.curvature.lanczos_steps = 12
    cfg.curvature.slq_probes = 2
    cfg.curvature.slq_steps = 8
    cfg.moments.backend = "loop"
    cfg.geometry.alpha_values = [0.0, 0.25, 0.5]
    cfg.geometry.damping_ratios = [0.0, 1.0, 100.0]
    cfg.geometry.response_vectors = 6
    cfg.geometry.exact_condition_max_dim = 64
    cfg.training.optimizer = "adamw"
    cfg.training.steps = 2
    cfg.training.checkpoint_steps = [0, 2]
    cfg.training.learning_rate = 0.02
    return cfg


def test_end_to_end_smoke_runner_writes_all_core_artifacts(tmp_path: Path) -> None:
    cfg = _smoke_config(tmp_path)
    config_path = tmp_path / "smoke.yaml"
    save_config(cfg, config_path)

    result = run_smoke(cfg)
    run_dir = Path(result["run_dir"])

    expected = [
        run_dir / "resolved_config.yaml",
        run_dir / "manifest.json",
        run_dir / "training" / "training.csv",
        run_dir / "geometry" / "geometry.csv",
        run_dir / "interventions" / "interventions.csv",
        run_dir / "dynamics" / "dynamics.csv",
        run_dir / "continuations" / "continuations.csv",
        run_dir / "summary" / "hypotheses.json",
        run_dir / "figures" / "geometry_delta_gain.pdf",
        run_dir / "figures" / "intervention_conditions.pdf",
        run_dir / "figures" / "dynamics_curves.pdf",
        run_dir / "figures" / "continuation_curves.pdf",
    ]
    for path in expected:
        assert path.exists(), path

    geometry = pd.read_csv(run_dir / "geometry" / "geometry.csv")
    assert {
        "block_name",
        "K_curvature",
        "K_adam",
        "K_shampoo_0.25",
        "G_adam",
        "G_shampoo_0.25",
        "delta_G_0.25",
        "response_shampoo",
        "projected_commutator_shampoo",
        "optimizer_state_kind",
        "K_optimizer_state",
        "G_optimizer_state",
    }.issubset(geometry.columns)
    assert len(geometry) == 1
    assert geometry["optimizer_state_kind"].iloc[0] == "adamw"
    assert geometry["K_optimizer_state"].notna().all()

    interventions = pd.read_csv(run_dir / "interventions" / "interventions.csv")
    assert {"run_name", "block_name", "intervention", "branch"}.issubset(
        interventions.columns
    )

    dynamics = pd.read_csv(run_dir / "dynamics" / "dynamics.csv")
    assert {
        "run_name",
        "block_name",
        "preconditioner",
        "method",
        "iteration",
        "relative_objective",
        "gradient_norm",
        "step_size",
    }.issubset(dynamics.columns)

    continuations = pd.read_csv(run_dir / "continuations" / "continuations.csv")
    assert {
        "run_name",
        "block_name",
        "preconditioner",
        "iteration",
        "loss",
        "relative_loss",
    }.issubset(continuations.columns)
    assert not continuations.empty

    with (run_dir / "summary" / "hypotheses.json").open("r", encoding="utf-8") as handle:
        summary = json.load(handle)
    assert set(summary) >= {"H1", "H2", "H3", "H4", "H5", "H6"}


def test_synthetic_runner_recovers_reciprocal_product_identity(tmp_path: Path) -> None:
    cfg = _smoke_config(tmp_path)
    cfg.run.name = "synthetic"
    result = run_synthetic(cfg)
    frame = pd.read_csv(Path(result["results_csv"]))
    paired = frame[frame["experiment"] == "flat_kron_pair"]
    assert not paired.empty
    relative_error = (
        (paired["K_plus"] * paired["K_minus"] - paired["K_H"] ** 2).abs()
        / (paired["K_H"] ** 2)
    )
    assert relative_error.max() < 1e-10


def test_checkpoint_sweep_records_all_checkpoints_and_stale_factors(tmp_path: Path) -> None:
    from ovc_experiments.runners import run_checkpoint_sweep

    cfg = _smoke_config(tmp_path)
    cfg.run.name = "pytest-checkpoint-sweep"
    cfg.training.steps = 2
    cfg.training.checkpoint_steps = [0, 1, 2]
    cfg.sweep.run_interventions = False
    cfg.sweep.run_dynamics = False
    cfg.sweep.staleness_checkpoint_lags = [0, 1]

    result = run_checkpoint_sweep(cfg)
    geometry = pd.read_csv(result["geometry_csv"])
    staleness = pd.read_csv(result["staleness_csv"])

    assert set(geometry["checkpoint_step"]) == {0, 1, 2}
    assert set(staleness["checkpoint_lag"]) == {0, 1}
    assert (staleness["source_step"] <= staleness["target_step"]).all()
    assert Path(result["checkpoint_figure"]).exists()
    assert Path(result["staleness_figure"]).exists()
    assert Path(result["hypotheses_json"]).exists()


def test_aggregate_runner_writes_cluster_bootstrap_and_mechanism_statistics(
    tmp_path: Path,
) -> None:
    from ovc_experiments.runners import run_aggregate

    rows = []
    for run_index, run_name in enumerate(("seed-0", "seed-1", "seed-2")):
        for block_index in range(4):
            response = -0.6 + 0.4 * block_index + 0.1 * run_index
            rows.append(
                {
                    "run_name": run_name,
                    "checkpoint_step": block_index,
                    "block_name": f"block-{block_index}",
                    "response_shampoo": response,
                    "projected_commutator_shampoo": 0.05 * block_index,
                    "delta_G_0.25": 1.5 * response,
                }
            )
    frame = pd.DataFrame(rows)
    paths = []
    for run_name, group in frame.groupby("run_name"):
        path = tmp_path / f"{run_name}.csv"
        group.to_csv(path, index=False)
        paths.append(path)

    intervention_paths = []
    for run_name in ("seed-0", "seed-1", "seed-2"):
        intervention = pd.DataFrame(
            [
                {
                    "run_name": run_name,
                    "block_name": f"block-{block_index}",
                    "intervention": "assignment",
                    "branch": branch,
                    "condition_number": (2.0 + block_index)
                    * (1.0 if branch == "aligned" else 3.0),
                }
                for block_index in range(4)
                for branch in ("aligned", "reversed")
            ]
        )
        path = tmp_path / f"{run_name}-interventions.csv"
        intervention.to_csv(path, index=False)
        intervention_paths.append(path)

    result = run_aggregate(
        paths,
        output_dir=tmp_path / "aggregate",
        intervention_paths=intervention_paths,
    )
    statistics_path = Path(result["statistics_json"])
    assert statistics_path.exists()
    statistics = json.loads(statistics_path.read_text(encoding="utf-8"))
    assert set(statistics["sign_fractions"]) == {"positive", "negative", "neutral"}
    assert statistics["mechanism_prediction"]["rows"] == len(frame)
    assert statistics["assignment_paired"]["success_fraction"] == 1.0
    assert Path(result["interventions_csv"]).exists()
    assert Path(result["assignment_effects_csv"]).exists()


def test_checkpoint_sweep_combines_optional_continuations(tmp_path: Path) -> None:
    from ovc_experiments.runners import run_checkpoint_sweep

    cfg = _smoke_config(tmp_path)
    cfg.run.name = "pytest-checkpoint-continuations"
    cfg.training.steps = 1
    cfg.training.checkpoint_steps = [0, 1]
    cfg.sweep.run_interventions = False
    cfg.sweep.run_dynamics = False
    cfg.sweep.run_continuations = True
    cfg.sweep.staleness_checkpoint_lags = [0]
    cfg.continuation.steps = 1
    cfg.continuation.preconditioners = ["identity"]

    result = run_checkpoint_sweep(cfg)

    assert result["continuations_csv"] is not None
    continuation_path = Path(result["continuations_csv"])
    assert continuation_path.exists()
    frame = pd.read_csv(continuation_path)
    assert set(frame["checkpoint_step"]) == {0, 1}
    assert set(frame["preconditioner"]) == {"identity"}


def test_cli_aggregate_exposes_statistical_controls() -> None:
    from ovc_experiments.cli import build_parser

    args = build_parser().parse_args(
        [
            "aggregate",
            "--geometry",
            "a.csv",
            "b.csv",
            "--output-dir",
            "aggregate-out",
            "--sign-threshold",
            "0.2",
            "--bootstrap-replicates",
            "17",
            "--seed",
            "9",
        ]
    )
    assert args.sign_threshold == 0.2
    assert args.bootstrap_replicates == 17
    assert args.seed == 9


def test_aggregate_runner_accepts_canonical_streaming_delta_gain(tmp_path: Path) -> None:
    from ovc_experiments.runners import run_aggregate

    paths: list[Path] = []
    for run_index, run_name in enumerate(("streaming-seed-0", "streaming-seed-1")):
        frame = pd.DataFrame(
            [
                {
                    "run_name": run_name,
                    "checkpoint_step": block_index,
                    "block_name": f"block-{block_index}",
                    "delta_G": (-1.0 if block_index % 2 else 1.0)
                    * (0.3 + 0.1 * run_index),
                    "curvature_censored": False,
                    "adam_censored": False,
                    "shampoo_censored": False,
                }
                for block_index in range(3)
            ]
        )
        path = tmp_path / f"{run_name}.csv"
        frame.to_csv(path, index=False)
        paths.append(path)

    result = run_aggregate(
        paths,
        output_dir=tmp_path / "canonical-aggregate",
        bootstrap_replicates=20,
    )

    statistics = json.loads(Path(result["statistics_json"]).read_text(encoding="utf-8"))
    assert set(statistics["sign_fractions"]) == {"positive", "negative", "neutral"}
    assert Path(result["figure"]).exists()
