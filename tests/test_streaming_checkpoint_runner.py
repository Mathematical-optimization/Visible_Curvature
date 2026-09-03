from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import torch

from ovc_experiments.blocks import BlockSpec, MatrixLayout, discover_matrix_blocks
from ovc_experiments.config import ExperimentConfig
from ovc_experiments.hardened_runner import HardenedBlockConfig, analyze_block_streaming
from ovc_experiments.functional import FunctionalBlockModel, iter_per_example_gradients
from ovc_experiments.models import TinyDecoderLM
from ovc_experiments.runners import build_task
from ovc_experiments.streaming_runner import (
    _positive_eigenvalue_mask,
    _run_interventions,
    ReplayableMeanOperator,
    run_streaming_geometry,
)
from ovc_experiments.safe_operators import DiagonalOperator


def _streaming_config(tmp_path: Path) -> ExperimentConfig:
    config = ExperimentConfig()
    config.run.name = "streaming-pytest"
    config.run.seed = 19
    config.run.dtype = "float64"
    config.output_dir = str(tmp_path / "outputs")
    config.model.family = "decoder"
    config.model.kwargs = {
        "vocab_size": 8,
        "max_seq_len": 4,
        "d_model": 4,
        "n_heads": 1,
        "n_layers": 1,
        "mlp_ratio": 1,
        "dropout": 0.0,
    }
    config.data.family = "synthetic_language"
    config.data.num_examples = 6
    config.data.batch_size = 2
    config.data.kwargs = {"seq_len": 3, "vocab_size": 8}
    config.blocks.include = [r"blocks\.0\.[qk]_proj\.weight$"]
    config.blocks.max_numel = 32
    config.curvature.kind = "ggn"
    config.curvature.shift = 1e-3
    config.curvature.exact_max_dim = 64
    config.curvature.lanczos_steps = 8
    config.curvature.lanczos_starts = 1
    config.curvature.residual_tolerance = 1e-6
    config.moments.backend = "loop"
    config.moments.max_examples = 4
    config.geometry.adam_damping = 1e-3
    config.geometry.shampoo_damping_ratio = 1e-3
    config.geometry.alpha_values = [0.25, 0.5]
    config.geometry.damping_ratios = [1e-3, 1.0]
    config.geometry.random_assignment_repeats = 0
    config.geometry.exact_condition_max_dim = 64
    config.streaming.curvature_batch_size = 2
    config.streaming.max_factor_elements = 10_000
    config.streaming.run_interventions = True
    config.streaming.assignment_max_dim = 64
    return config



def test_replayable_mean_operator_uses_all_weighted_batches() -> None:
    batches = [
        (DiagonalOperator(torch.tensor([1.0, 3.0], dtype=torch.float64)), 1),
        (DiagonalOperator(torch.tensor([5.0, 7.0], dtype=torch.float64)), 3),
    ]

    operator = ReplayableMeanOperator(
        batch_factory=lambda: iter(batches),
        operator_factory=lambda item: item[0],
        weight_function=lambda item: item[1],
        expected_weight=4,
        dimension=2,
        dtype=torch.float64,
        device="cpu",
        name="weighted-test",
    )

    result = operator.matvec(torch.ones(2, dtype=torch.float64))

    assert torch.allclose(result, torch.tensor([4.0, 6.0], dtype=torch.float64))

def test_iter_per_example_gradients_replays_a_batch_factory() -> None:
    torch.manual_seed(3)
    model = TinyDecoderLM(
        vocab_size=8,
        max_seq_len=4,
        d_model=4,
        n_heads=1,
        n_layers=1,
        mlp_ratio=1,
        dropout=0.0,
    ).to(dtype=torch.float64)
    task = build_task(ExperimentConfig())
    block = discover_matrix_blocks(
        model,
        include=[r"blocks\.0\.q_proj\.weight$"],
    )[0]
    functional = FunctionalBlockModel(model, block, task)
    batch = {
        "input_ids": torch.tensor(
            [[0, 1, 2], [1, 3, 5], [2, 4, 6], [3, 5, 7]], dtype=torch.long
        ),
        "labels": torch.tensor(
            [[1, 2, 3], [3, 5, 7], [4, 6, 0], [5, 7, 1]], dtype=torch.long
        ),
    }

    def batch_factory():
        yield {key: value[:2] for key, value in batch.items()}
        yield {key: value[2:] for key, value in batch.items()}

    first = list(iter_per_example_gradients(functional, batch_factory, backend="loop"))
    second = list(iter_per_example_gradients(functional, batch_factory, backend="loop"))

    assert len(first) == len(second) == 4
    assert all(item.shape == block.shape for item in first)
    assert all(torch.allclose(left, right) for left, right in zip(first, second, strict=True))


def test_streaming_geometry_processes_blocks_one_at_a_time_and_writes_interventions(
    tmp_path: Path,
) -> None:
    config = _streaming_config(tmp_path)

    result = run_streaming_geometry(config)

    geometry_path = Path(result["geometry_csv"])
    interventions_path = Path(result["interventions_csv"])
    run_dir = Path(result["run_dir"])
    geometry = pd.read_csv(geometry_path)
    interventions = pd.read_csv(interventions_path)

    assert result["blocks_total"] == 2
    assert result["blocks_completed"] == 2
    manifest = json.loads((run_dir / "streaming" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["curvature_example_count"] == manifest["moment_example_count"] == 4
    assert manifest["curvature_batch_size"] == 2
    assert geometry["block_name"].nunique() == 2
    assert set(interventions["intervention"]) == {"alpha", "damping", "assignment"}
    assert set(interventions["block_name"]) == set(geometry["block_name"])
    assert not any(run_dir.rglob("*.pt"))

    summaries = [
        json.loads(line)
        for line in (run_dir / "streaming" / "moments_summary.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert len(summaries) == 2
    assert all(record["raw_gradients_retained"] is False for record in summaries)
    assert "moments" not in result


def test_streaming_geometry_censors_oversized_factors_before_accumulation(
    tmp_path: Path,
) -> None:
    config = _streaming_config(tmp_path)
    config.blocks.include = [r"blocks\.0\.q_proj\.weight$"]
    config.streaming.max_factor_elements = 1
    config.streaming.run_interventions = False

    result = run_streaming_geometry(config)
    geometry = pd.read_csv(result["geometry_csv"])

    assert result["blocks_completed"] == 0
    assert result["blocks_censored"] == 1
    assert geometry.loc[0, "curvature_censored"]
    assert geometry.loc[0, "adam_censored"]
    assert geometry.loc[0, "shampoo_censored"]
    assert geometry.loc[0, "censor_reason"] == "factor_storage_exceeds_limit"



def test_streaming_proxy_active_mask_is_scale_invariant() -> None:
    small = torch.tensor([1e-12, 2e-12, 4e-12], dtype=torch.float64)
    large = small * 1e18

    small_mask = _positive_eigenvalue_mask(small, relative_threshold=1e-10)
    large_mask = _positive_eigenvalue_mask(large, relative_threshold=1e-10)

    assert torch.equal(small_mask, large_mask)
    assert small_mask.all()

def test_cli_exposes_streaming_geometry_command() -> None:
    from ovc_experiments.cli import build_parser

    args = build_parser().parse_args(["streaming-geometry", "--config", "config.yaml"])

    assert args.command == "streaming-geometry"


def test_streaming_interventions_record_censors_when_base_shampoo_is_unresolved(
    tmp_path: Path,
) -> None:
    config = _streaming_config(tmp_path)
    curvature = DiagonalOperator(torch.tensor([1.0, 2.0], dtype=torch.float64))
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
        example_count=3,
        config=HardenedBlockConfig(
            centered_moments=True,
            adam_damping=1e-3,
            shampoo_damping_ratio=1e-3,
            exact_condition_max_dim=8,
        ),
        metadata={"block_name": "zero-centered-factors"},
    )
    block = BlockSpec(
        name="zero-centered-factors",
        shape=(2, 1),
        layout=MatrixLayout.from_shape((2, 1)),
        numel=2,
    )
    output = tmp_path / "interventions.csv"

    _run_interventions(
        config,
        block,
        curvature,
        result,
        output_path=output,
        metadata={"block_name": block.name},
        block_index=0,
    )

    frame = pd.read_csv(output)
    assert set(frame["intervention"]) == {"alpha", "damping", "assignment"}
    assert frame["censored"].all()
    assert frame["censor_reason"].str.startswith("base_shampoo_unresolved:").all()
