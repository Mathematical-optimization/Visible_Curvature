from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, TypeVar, get_args, get_origin, get_type_hints

import yaml


@dataclass
class RunConfig:
    name: str = "ovc-run"
    seed: int = 0
    device: str = "cpu"
    dtype: str = "float64"
    deterministic: bool = True


@dataclass
class ModelConfig:
    family: str = "decoder"
    checkpoint: str | None = None
    kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class DataConfig:
    family: str = "synthetic"
    path: str | None = None
    num_examples: int = 32
    batch_size: int = 8
    kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class TaskConfig:
    family: str = "auto"
    input_key: str = "inputs"
    target_key: str = "targets"
    ignore_index: int = -100
    kwargs: dict[str, Any] = field(default_factory=dict)

@dataclass
class BlockConfig:
    include: list[str] = field(default_factory=lambda: [r".*\.weight$"])
    exclude: list[str] = field(default_factory=list)
    min_numel: int = 2
    max_numel: int | None = None
    tile_rows: int | None = None
    tile_cols: int | None = None


@dataclass
class CurvatureConfig:
    kind: str = "ggn"
    shift: float = 1e-4
    exact_max_dim: int = 512
    lanczos_steps: int = 32
    lanczos_starts: int = 2
    positive_threshold: float = 1e-10
    residual_tolerance: float = 1e-5
    subspace_policy: str = "strict_spd"
    slq_probes: int = 8
    slq_steps: int = 24


@dataclass
class MomentConfig:
    centered: bool = True
    backend: str = "loop"
    accumulation_dtype: str = "float64"
    max_examples: int | None = None


@dataclass
class GeometryConfig:
    alpha_values: list[float] = field(
        default_factory=lambda: [0.0, 0.125, 0.25, 0.375, 0.5]
    )
    damping_ratios: list[float] = field(
        default_factory=lambda: [0.0, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]
    )
    adam_damping: float = 0.0
    response_vectors: int = 12
    sign_threshold: float = 0.22314355131420976  # log(1.25)
    exact_condition_max_dim: int = 512
    shampoo_damping_ratio: float = 1e-3
    random_assignment_repeats: int = 4
    dynamics_steps: int = 12
    grafting_scales: list[float] = field(default_factory=lambda: [0.1, 1.0, 10.0])


@dataclass
class ContinuationConfig:
    steps: int = 5
    step_fraction: float = 0.5
    preconditioners: list[str] = field(
        default_factory=lambda: [
            "identity",
            "adam",
            "shampoo_0.25",
            "shampoo_0.5",
            "optimizer_state",
        ]
    )

@dataclass
class SweepConfig:
    run_interventions: bool = True
    run_dynamics: bool = True
    run_continuations: bool = False
    staleness_checkpoint_lags: list[int] = field(default_factory=lambda: [0, 1, 2])
    staleness_alpha: float = 0.25


@dataclass
class StreamingConfig:
    """Controls for the memory-bounded checkpoint/block geometry path."""

    curvature_batch_size: int | None = None
    max_factor_elements: int = 50_000_000
    run_interventions: bool = True
    assignment_max_dim: int = 512


@dataclass
class TrainingConfig:
    optimizer: str = "adamw"
    steps: int = 20
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    checkpoint_steps: list[int] = field(default_factory=lambda: [0, 5, 10, 20])
    root_frequency: int = 5
    beta1: float = 0.9
    beta2: float = 0.99
    epsilon: float = 1e-8
    alpha: float = 0.25
    grafting: str = "none"


@dataclass
class ExperimentConfig:
    run: RunConfig = field(default_factory=RunConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    task: TaskConfig = field(default_factory=TaskConfig)
    blocks: BlockConfig = field(default_factory=BlockConfig)
    curvature: CurvatureConfig = field(default_factory=CurvatureConfig)
    moments: MomentConfig = field(default_factory=MomentConfig)
    geometry: GeometryConfig = field(default_factory=GeometryConfig)
    continuation: ContinuationConfig = field(default_factory=ContinuationConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    sweep: SweepConfig = field(default_factory=SweepConfig)
    streaming: StreamingConfig = field(default_factory=StreamingConfig)
    output_dir: str = "experiments/outputs"


T = TypeVar("T")


def _construct_dataclass(cls: type[T], values: dict[str, Any], *, path: str = "") -> T:
    known = {item.name for item in fields(cls)}
    unknown = sorted(set(values) - known)
    if unknown:
        full_paths = [f"{path}.{name}" if path else name for name in unknown]
        raise ValueError(f"Unknown configuration key(s): {', '.join(full_paths)}")
    hints = get_type_hints(cls)
    kwargs: dict[str, Any] = {}
    for item in fields(cls):
        if item.name not in values:
            continue
        value = values[item.name]
        hinted = hints.get(item.name, item.type)
        if isinstance(value, dict):
            origin = get_origin(hinted)
            args = get_args(hinted)
            nested_type = hinted
            if origin is not None and origin is not dict:
                candidates = [arg for arg in args if isinstance(arg, type) and is_dataclass(arg)]
                if candidates:
                    nested_type = candidates[0]
            if isinstance(nested_type, type) and is_dataclass(nested_type):
                nested_path = f"{path}.{item.name}" if path else item.name
                value = _construct_dataclass(nested_type, value, path=nested_path)
        kwargs[item.name] = value
    return cls(**kwargs)


def validate_config(config: ExperimentConfig) -> ExperimentConfig:
    if config.curvature.subspace_policy not in {
        "strict_spd",
        "positive_active",
        "pseudoinverse",
    }:
        raise ValueError(
            "curvature.subspace_policy must be one of "
            "strict_spd, positive_active, pseudoinverse"
        )
    nonnegative = {
        "curvature.shift": config.curvature.shift,
        "curvature.positive_threshold": config.curvature.positive_threshold,
        "curvature.residual_tolerance": config.curvature.residual_tolerance,
        "geometry.adam_damping": config.geometry.adam_damping,
        "geometry.shampoo_damping_ratio": config.geometry.shampoo_damping_ratio,
    }
    for name, value in nonnegative.items():
        if float(value) < 0:
            raise ValueError(f"{name} must be nonnegative")
    positive = {
        "curvature.lanczos_steps": config.curvature.lanczos_steps,
        "curvature.lanczos_starts": config.curvature.lanczos_starts,
        "curvature.exact_max_dim": config.curvature.exact_max_dim,
        "geometry.exact_condition_max_dim": config.geometry.exact_condition_max_dim,
        "streaming.max_factor_elements": config.streaming.max_factor_elements,
        "streaming.assignment_max_dim": config.streaming.assignment_max_dim,
    }
    for name, value in positive.items():
        if int(value) < 1:
            raise ValueError(f"{name} must be positive")
    if (config.blocks.tile_rows is None) != (config.blocks.tile_cols is None):
        raise ValueError("blocks.tile_rows and blocks.tile_cols must be set together")
    for name, value in (
        ("blocks.tile_rows", config.blocks.tile_rows),
        ("blocks.tile_cols", config.blocks.tile_cols),
    ):
        if value is not None and int(value) < 1:
            raise ValueError(f"{name} must be positive")
    if config.streaming.curvature_batch_size is not None and int(
        config.streaming.curvature_batch_size
    ) < 1:
        raise ValueError("streaming.curvature_batch_size must be positive")
    if not config.output_dir:
        raise ValueError("output_dir must be nonempty")
    return config


def load_config(path: str | Path) -> ExperimentConfig:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Configuration root must be a mapping: {config_path}")
    return validate_config(_construct_dataclass(ExperimentConfig, payload))


def save_config(config: ExperimentConfig, path: str | Path) -> None:
    validate_config(config)
    config_path = Path(path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with config_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(asdict(config), handle, sort_keys=False, allow_unicode=True)
