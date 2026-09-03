from __future__ import annotations

from pathlib import Path

import torch

from ovc_experiments.blocks import BlockSpec, MatrixLayout, discover_matrix_blocks
from ovc_experiments.config import ExperimentConfig, load_config, save_config


def test_matrix_layout_round_trip_for_matrix_and_convolution() -> None:
    matrix = torch.arange(12, dtype=torch.float64).reshape(3, 4)
    matrix_layout = MatrixLayout.from_shape(matrix.shape)
    assert matrix_layout.matrix_shape == (3, 4)
    assert torch.equal(matrix_layout.from_matrix(matrix_layout.to_matrix(matrix)), matrix)

    conv = torch.arange(2 * 3 * 2 * 2, dtype=torch.float64).reshape(2, 3, 2, 2)
    conv_layout = MatrixLayout.from_shape(conv.shape)
    assert conv_layout.matrix_shape == (2, 12)
    assert torch.equal(conv_layout.from_matrix(conv_layout.to_matrix(conv)), conv)


def test_block_discovery_respects_min_dimensions_and_patterns() -> None:
    model = torch.nn.Sequential(
        torch.nn.Linear(4, 3),
        torch.nn.LayerNorm(3),
        torch.nn.Linear(3, 2, bias=False),
    )
    blocks = discover_matrix_blocks(model, include=[r"0\.weight", r"2\.weight"])
    names = [block.name for block in blocks]
    assert names == ["0.weight", "2.weight"]
    assert all(isinstance(block, BlockSpec) for block in blocks)
    assert [block.layout.matrix_shape for block in blocks] == [(3, 4), (2, 3)]


def test_experiment_config_yaml_round_trip(tmp_path: Path) -> None:
    cfg = ExperimentConfig()
    cfg.run.name = "roundtrip"
    cfg.model.family = "decoder"
    cfg.geometry.alpha_values = [0.0, 0.25, 0.5]
    cfg.output_dir = str(tmp_path / "outputs")

    path = tmp_path / "config.yaml"
    save_config(cfg, path)
    loaded = load_config(path)

    assert loaded.run.name == "roundtrip"
    assert loaded.model.family == "decoder"
    assert loaded.geometry.alpha_values == [0.0, 0.25, 0.5]
    assert loaded.output_dir == str(tmp_path / "outputs")


def test_explicit_task_config_supports_custom_model_factories() -> None:
    from ovc_experiments.runners import build_task
    from ovc_experiments.tasks import CausalLMTask

    config = ExperimentConfig()
    config.model.family = "python"
    config.task.family = "causal_lm"
    config.task.input_key = "tokens"
    config.task.target_key = "next_tokens"
    task = build_task(config)
    assert isinstance(task, CausalLMTask)
    assert task.input_key == "tokens"
    assert task.target_key == "next_tokens"


def test_all_shipped_configs_load() -> None:
    config_dir = Path(__file__).resolve().parents[1] / "configs"
    paths = sorted(config_dir.glob("*.yaml"))
    assert paths
    for path in paths:
        loaded = load_config(path)
        assert loaded.run.name
        assert loaded.output_dir
