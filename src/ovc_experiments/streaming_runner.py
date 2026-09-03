from __future__ import annotations

from dataclasses import asdict
from hashlib import sha256
import math
from pathlib import Path
from typing import Any, Callable, Iterable

import torch
from torch.utils.data import DataLoader, Dataset, Subset

from .blocks import BlockSpec, discover_matrix_blocks
from .config import ExperimentConfig, validate_config
from .curvature import CrossEntropyGGNOperator, ExactHessianOperator, RegularizedOperator
from .curvature_policy import empirical_fisher_operator
from .diagnostics import curvature_factor_proxies_from_ritz
from .functional import FunctionalBlockModel, iter_per_example_gradients
from .hardened_runner import HardenedBlockConfig, HardenedBlockResult, analyze_block_streaming
from .interventions import alpha_sweep, assignment_sweep, damping_sweep
from .io import append_dataframe_row, append_jsonl_strict, atomic_write_json, file_sha256
from .operators import SymmetricLinearOperator, materialize
from .runners import _dtype, _prepare_run, _seed_everything, build_dataset, build_model, build_task
from .training import load_checkpoint, move_batch



class ReplayableMeanOperator(SymmetricLinearOperator):
    """Weighted mean of per-batch symmetric operators with bounded memory."""

    def __init__(
        self,
        *,
        batch_factory: Callable[[], Iterable[Any]],
        operator_factory: Callable[[Any], Any],
        weight_function: Callable[[Any], int | float],
        expected_weight: int | float,
        dimension: int,
        dtype: torch.dtype,
        device: torch.device | str,
        name: str = "replayable-mean-operator",
    ) -> None:
        if dimension < 1:
            raise ValueError("dimension must be positive")
        if float(expected_weight) <= 0:
            raise ValueError("expected_weight must be positive")
        self.batch_factory = batch_factory
        self.operator_factory = operator_factory
        self.weight_function = weight_function
        self.expected_weight = float(expected_weight)
        self.dimension = int(dimension)
        self.dtype = dtype
        self.device = torch.device(device)
        self.name = name

    def matvec(self, vector: torch.Tensor) -> torch.Tensor:
        self._validate_vector(vector)
        vector = vector.to(device=self.device, dtype=self.dtype)
        output = torch.zeros_like(vector)
        observed_weight = 0.0
        for batch in self.batch_factory():
            weight = float(self.weight_function(batch))
            if not math.isfinite(weight) or weight <= 0:
                raise ValueError("batch weights must be finite and positive")
            operator = self.operator_factory(batch)
            if int(getattr(operator, "dimension", getattr(operator, "dim", -1))) != self.dimension:
                raise ValueError("batch operator dimension changed between replays")
            output.add_(operator.matvec(vector), alpha=weight)
            observed_weight += weight
        tolerance = 1e-12 * max(1.0, abs(self.expected_weight))
        if abs(observed_weight - self.expected_weight) > tolerance:
            raise ValueError(
                f"expected total batch weight {self.expected_weight:g}, "
                f"observed {observed_weight:g}"
            )
        return output / observed_weight

def _positive_eigenvalue_mask(
    eigenvalues: torch.Tensor,
    *,
    relative_threshold: float,
) -> torch.Tensor:
    """Select resolved positive modes using a scale-relative threshold."""

    if eigenvalues.ndim != 1 or eigenvalues.numel() == 0:
        raise ValueError("eigenvalues must be a nonempty vector")
    if relative_threshold < 0:
        raise ValueError("relative_threshold must be nonnegative")
    if not torch.isfinite(eigenvalues).all():
        raise ValueError("eigenvalues contain non-finite values")
    dtype = eigenvalues.dtype if eigenvalues.dtype.is_floating_point else torch.float64
    values = eigenvalues.to(dtype=dtype)
    scale = float(values.abs().max().item())
    finfo = torch.finfo(dtype)
    threshold = max(
        float(relative_threshold) * scale,
        64.0 * finfo.eps * values.numel() * max(scale, finfo.tiny),
    )
    return values > threshold

def _indices_hash(indices: list[int]) -> str:
    payload = ",".join(str(value) for value in indices).encode("utf-8")
    return sha256(payload).hexdigest()


def _batch_factory(
    dataset: Dataset,
    indices: list[int],
    *,
    batch_size: int,
    device: torch.device | str,
    dtype: torch.dtype,
) -> Callable[[], Iterable[Any]]:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")

    def factory() -> Iterable[Any]:
        loader = DataLoader(
            Subset(dataset, indices),
            batch_size=min(batch_size, len(indices)),
            shuffle=False,
            drop_last=False,
        )
        for batch in loader:
            yield move_batch(batch, device, dtype=dtype)

    return factory


def _resolve_optional_checkpoint(
    config: ExperimentConfig,
    directory: Path,
    model: torch.nn.Module,
) -> tuple[str, dict[str, Any], str | None]:
    path: Path | None = None
    if config.model.checkpoint:
        path = Path(config.model.checkpoint).expanduser().resolve()
    else:
        pointer = directory / "latest_checkpoint.txt"
        if pointer.exists():
            candidate = Path(pointer.read_text(encoding="utf-8").strip()).expanduser().resolve()
            if candidate.exists():
                path = candidate
    if path is None:
        return "initialization", {"step": 0, "checkpoint_format": "initialization"}, None
    if not path.exists():
        raise FileNotFoundError(path)
    metadata = load_checkpoint(model, path, map_location=config.run.device)
    return str(path), metadata, file_sha256(path)


def _curvature_operator(
    config: ExperimentConfig,
    functional: FunctionalBlockModel,
    curvature_batch_factory: Callable[[], Iterable[Any]],
    gradient_factory: Callable[[], Iterable[torch.Tensor]],
    example_count: int,
) -> SymmetricLinearOperator:
    kind = config.curvature.kind.lower()
    if kind in {"ggn", "generalized_gauss_newton"}:
        operator: SymmetricLinearOperator = ReplayableMeanOperator(
            batch_factory=curvature_batch_factory,
            operator_factory=lambda batch: CrossEntropyGGNOperator(functional, batch),
            weight_function=functional.task.batch_size,
            expected_weight=example_count,
            dimension=functional.block.numel,
            dtype=functional.block_parameter.dtype,
            device=functional.block_parameter.device,
            name="streaming-cross-entropy-ggn",
        )
    elif kind in {"hessian", "exact_hessian"}:
        operator = ReplayableMeanOperator(
            batch_factory=curvature_batch_factory,
            operator_factory=lambda batch: ExactHessianOperator(functional, batch),
            weight_function=functional.task.batch_size,
            expected_weight=example_count,
            dimension=functional.block.numel,
            dtype=functional.block_parameter.dtype,
            device=functional.block_parameter.device,
            name="streaming-exact-hessian",
        )
    elif kind in {"fisher", "empirical_fisher"}:
        operator = empirical_fisher_operator(
            gradient_factory,
            functional.block.numel,
            count=example_count,
            centered=False,
            dtype=functional.block_parameter.dtype,
            device=functional.block_parameter.device,
        )
    else:
        raise ValueError(f"Unsupported curvature kind: {config.curvature.kind}")
    if config.curvature.shift > 0:
        operator = RegularizedOperator(operator, config.curvature.shift)
    return operator


def _hardened_config(config: ExperimentConfig) -> HardenedBlockConfig:
    primary = config.curvature.kind.lower() not in {"fisher", "empirical_fisher"}
    return HardenedBlockConfig(
        curvature_kind=config.curvature.kind,
        primary_analysis=primary,
        centered_moments=config.moments.centered,
        population_moments=True,
        adam_damping=config.geometry.adam_damping,
        shampoo_damping_ratio=config.geometry.shampoo_damping_ratio,
        shampoo_exponent=0.25,
        subspace_policy=config.curvature.subspace_policy,
        exact_condition_max_dim=config.geometry.exact_condition_max_dim,
        lanczos_steps=config.curvature.lanczos_steps,
        lanczos_starts=config.curvature.lanczos_starts,
        residual_tolerance=config.curvature.residual_tolerance,
        relative_positive_threshold=config.curvature.positive_threshold,
        retain_raw_gradients=False,
    )


def _censored_block_row(
    config: ExperimentConfig,
    block: BlockSpec,
    *,
    checkpoint: str,
    reason: str,
    example_count: int,
) -> dict[str, Any]:
    return {
        "run_name": config.run.name,
        "model_family": config.model.family,
        "seed": config.run.seed,
        "checkpoint": checkpoint,
        "block_name": block.name,
        "block_shape": "x".join(map(str, block.shape)),
        "dimension": block.numel,
        "example_count": example_count,
        "curvature_kind": config.curvature.kind,
        "K_curvature": math.nan,
        "K_adam": math.nan,
        "K_shampoo": math.nan,
        "G_adam": math.nan,
        "G_shampoo": math.nan,
        "delta_G": math.nan,
        "curvature_censored": True,
        "adam_censored": True,
        "shampoo_censored": True,
        "curvature_censor_reason": reason,
        "adam_censor_reason": reason,
        "shampoo_censor_reason": reason,
        "censor_reason": reason,
    }


def _sweep_censor_reason(point: Any) -> str | None:
    geometry = getattr(point, "geometry", None)
    if geometry is None:
        return None
    effective = getattr(geometry, "effective_condition", None)
    return getattr(effective, "censor_reason", None)


def _append_sweep_points(
    path: Path,
    *,
    intervention: str,
    branch_getter: Callable[[Any], str],
    points: Iterable[Any],
    common: dict[str, Any],
    adam_gain: float,
) -> None:
    for point in points:
        gain = point.gain
        valid_delta = (
            gain is not None
            and math.isfinite(float(gain))
            and math.isfinite(adam_gain)
            and not point.censored
        )
        append_dataframe_row(
            {
                **common,
                "intervention": intervention,
                "branch": branch_getter(point),
                "alpha": point.alpha,
                "condition_number": point.condition_number,
                "K_shampoo": point.condition_number,
                "gain": gain,
                "G_shampoo": gain,
                "G_adam": adam_gain,
                "delta_G": float(gain) - adam_gain if valid_delta else math.nan,
                "damping_left": point.damping_left,
                "damping_right": point.damping_right,
                "rho_over_min": point.rho_over_min,
                "rho_over_max": point.rho_over_max,
                "rho_left_over_min": point.rho_left_over_min,
                "rho_left_over_max": point.rho_left_over_max,
                "rho_right_over_min": point.rho_right_over_min,
                "rho_right_over_max": point.rho_right_over_max,
                "censored": bool(point.censored),
                "censor_reason": _sweep_censor_reason(point),
            },
            path,
        )


def _append_censored_intervention_rows(
    path: Path,
    *,
    config: ExperimentConfig,
    common: dict[str, Any],
    adam_gain: float,
    reason: str,
    interventions: set[str] | None = None,
) -> None:
    selected = interventions or {"alpha", "damping", "assignment"}
    rows: list[tuple[str, str, float | None, float | None]] = []
    if "alpha" in selected:
        rows.extend(("alpha", "natural", float(alpha), None) for alpha in config.geometry.alpha_values)
    if "damping" in selected:
        rows.extend(
            ("damping", "natural", 0.25, float(ratio))
            for ratio in config.geometry.damping_ratios
        )
    if "assignment" in selected:
        rows.extend(
            ("assignment", branch, 0.25, None)
            for branch in (
                "aligned",
                "reversed",
                *(f"random-{index}" for index in range(config.geometry.random_assignment_repeats)),
            )
        )
    for intervention, branch, alpha, ratio in rows:
        append_dataframe_row(
            {
                **common,
                "intervention": intervention,
                "branch": branch,
                "alpha": alpha,
                "condition_number": math.nan,
                "K_shampoo": math.nan,
                "gain": math.nan,
                "G_shampoo": math.nan,
                "G_adam": adam_gain,
                "delta_G": math.nan,
                "rho_over_min": ratio,
                "rho_over_max": math.nan,
                "rho_left_over_min": ratio,
                "rho_left_over_max": math.nan,
                "rho_right_over_min": ratio,
                "rho_right_over_max": math.nan,
                "censored": True,
                "censor_reason": reason,
            },
            path,
        )


def _run_interventions(
    config: ExperimentConfig,
    block: BlockSpec,
    curvature: SymmetricLinearOperator,
    result: HardenedBlockResult,
    *,
    output_path: Path,
    metadata: dict[str, Any],
    block_index: int,
) -> None:
    moments = result.moments
    left = moments.left_centered if config.moments.centered else moments.left_uncentered
    right = moments.right_centered if config.moments.centered else moments.right_uncentered
    damping_left = float(result.row["shampoo_damping_left"])
    damping_right = float(result.row["shampoo_damping_right"])
    adam_gain = float(result.row["G_adam"])
    common = {
        **metadata,
        "K_curvature": result.row["K_curvature"],
        "K_adam": result.row["K_adam"],
        "curvature_censored": result.row["curvature_censored"],
        "adam_censored": result.row["adam_censored"],
    }
    sweep_common = dict(
        exact_max_dim=config.geometry.exact_condition_max_dim,
        lanczos_steps=config.curvature.lanczos_steps,
        lanczos_starts=config.curvature.lanczos_starts,
        positive_threshold=config.curvature.positive_threshold,
        residual_tolerance=config.curvature.residual_tolerance,
        subspace_policy=config.curvature.subspace_policy,
    )
    if bool(result.row.get("curvature_censored", True)):
        reason = f"base_curvature_unresolved:{result.row.get('curvature_censor_reason')}"
        _append_censored_intervention_rows(
            output_path,
            config=config,
            common=common,
            adam_gain=adam_gain,
            reason=reason,
        )
        return
    if not (math.isfinite(damping_left) and math.isfinite(damping_right)):
        reason = f"base_shampoo_unresolved:{result.row.get('shampoo_censor_reason')}"
        _append_censored_intervention_rows(
            output_path,
            config=config,
            common=common,
            adam_gain=adam_gain,
            reason=reason,
        )
        return
    alpha_points = alpha_sweep(
        curvature,
        left,
        right,
        layout=block.layout,
        alphas=config.geometry.alpha_values,
        damping_left=damping_left,
        damping_right=damping_right,
        seed=config.run.seed + 10_000 * block_index,
        **sweep_common,
    )
    _append_sweep_points(
        output_path,
        intervention="alpha",
        branch_getter=lambda _point: "natural",
        points=alpha_points,
        common=common,
        adam_gain=adam_gain,
    )
    damping_points = damping_sweep(
        curvature,
        left,
        right,
        layout=block.layout,
        alpha=0.25,
        normalized_ratios=config.geometry.damping_ratios,
        seed=config.run.seed + 20_000 * block_index,
        **sweep_common,
    )
    _append_sweep_points(
        output_path,
        intervention="damping",
        branch_getter=lambda _point: "natural",
        points=damping_points,
        common=common,
        adam_gain=adam_gain,
    )

    if block.numel > config.streaming.assignment_max_dim:
        append_dataframe_row(
            {
                **common,
                "intervention": "assignment",
                "branch": "unresolved",
                "alpha": 0.25,
                "condition_number": math.nan,
                "K_shampoo": math.nan,
                "gain": math.nan,
                "G_shampoo": math.nan,
                "G_adam": adam_gain,
                "delta_G": math.nan,
                "censored": True,
                "censor_reason": "assignment_dimension_exceeds_limit",
            },
            output_path,
        )
        return

    matrix = materialize(curvature, max_dimension=config.streaming.assignment_max_dim)
    eigenvalues, eigenvectors = torch.linalg.eigh(matrix)
    active = _positive_eigenvalue_mask(
        eigenvalues,
        relative_threshold=config.curvature.positive_threshold,
    )
    if not bool(active.any()):
        append_dataframe_row(
            {
                **common,
                "intervention": "assignment",
                "branch": "unresolved",
                "alpha": 0.25,
                "condition_number": math.nan,
                "K_shampoo": math.nan,
                "gain": math.nan,
                "G_shampoo": math.nan,
                "G_adam": adam_gain,
                "delta_G": math.nan,
                "censored": True,
                "censor_reason": "no_positive_curvature_proxy_modes",
            },
            output_path,
        )
        return
    left_proxy, right_proxy = curvature_factor_proxies_from_ritz(
        eigenvalues[active],
        eigenvectors[:, active],
        block.layout,
        positive_only=True,
    )
    assignment_points = assignment_sweep(
        curvature,
        left,
        right,
        left_proxy,
        right_proxy,
        layout=block.layout,
        alpha=0.25,
        damping_left=damping_left,
        damping_right=damping_right,
        random_repeats=config.geometry.random_assignment_repeats,
        seed=config.run.seed + 30_000 * block_index,
        **sweep_common,
    )
    _append_sweep_points(
        output_path,
        intervention="assignment",
        branch_getter=lambda point: str(point.label),
        points=assignment_points,
        common=common,
        adam_gain=adam_gain,
    )


def run_streaming_geometry(
    config: ExperimentConfig,
    *,
    directory: Path | None = None,
) -> dict[str, Any]:
    """Run the primary one-block-at-a-time, no-N-by-d geometry pipeline."""

    validate_config(config)
    _seed_everything(config.run.seed, config.run.deterministic)
    directory = directory or _prepare_run(config)
    streaming_dir = directory / "streaming"
    geometry_path = streaming_dir / "geometry.csv"
    interventions_path = streaming_dir / "interventions.csv"
    if geometry_path.exists() and geometry_path.stat().st_size > 0:
        raise FileExistsError(
            f"streaming output already exists: {geometry_path}; choose a new run.name"
        )
    streaming_dir.mkdir(parents=True, exist_ok=True)

    dtype = _dtype(config.run.dtype)
    model = build_model(config).to(device=config.run.device, dtype=dtype)
    checkpoint, checkpoint_metadata, checkpoint_hash = _resolve_optional_checkpoint(
        config, directory, model
    )
    model.eval()
    dataset = build_dataset(config)
    task = build_task(config)
    example_count = min(
        len(dataset),
        int(config.moments.max_examples or config.data.num_examples),
    )
    if example_count < 1:
        raise ValueError("streaming geometry requires at least one example")
    indices = list(range(example_count))
    batches = _batch_factory(
        dataset,
        indices,
        batch_size=config.data.batch_size,
        device=config.run.device,
        dtype=dtype,
    )
    curvature_batch_size = min(
        example_count,
        int(config.streaming.curvature_batch_size or config.data.batch_size),
    )
    curvature_batches = _batch_factory(
        dataset,
        indices,
        batch_size=curvature_batch_size,
        device=config.run.device,
        dtype=dtype,
    )

    blocks = discover_matrix_blocks(
        model,
        include=config.blocks.include,
        exclude=config.blocks.exclude,
        min_numel=config.blocks.min_numel,
        max_numel=config.blocks.max_numel,
    )
    if not blocks:
        raise ValueError("No matrix-shaped blocks matched the configured patterns")

    atomic_write_json(
        {
            "run_name": config.run.name,
            "seed": config.run.seed,
            "checkpoint": checkpoint,
            "checkpoint_sha256": checkpoint_hash,
            "checkpoint_step": checkpoint_metadata.get("step"),
            "example_indices": indices,
            "example_indices_sha256": _indices_hash(indices),
            "curvature_example_count": example_count,
            "curvature_batch_size": curvature_batch_size,
            "moment_example_count": example_count,
            "blocks": [block.name for block in blocks],
            "streaming_config": asdict(config.streaming),
            "raw_gradients_retained": False,
        },
        streaming_dir / "manifest.json",
    )

    completed = 0
    censored = 0
    block_config = _hardened_config(config)
    for block_index, block in enumerate(blocks):
        factor_elements = block.layout.matrix_shape[0] ** 2 + block.layout.matrix_shape[1] ** 2
        metadata = {
            "run_name": config.run.name,
            "model_family": config.model.family,
            "seed": config.run.seed,
            "checkpoint": checkpoint,
            "checkpoint_step": checkpoint_metadata.get("step", 0),
            "block_name": block.name,
            "block_shape": "x".join(map(str, block.shape)),
            "factor_elements": factor_elements,
        }
        if factor_elements > config.streaming.max_factor_elements:
            row = _censored_block_row(
                config,
                block,
                checkpoint=checkpoint,
                reason="factor_storage_exceeds_limit",
                example_count=example_count,
            )
            append_dataframe_row(row, geometry_path)
            append_jsonl_strict(
                {**metadata, "status": "censored", "reason": row["censor_reason"]},
                streaming_dir / "progress.jsonl",
            )
            censored += 1
            continue

        functional = FunctionalBlockModel(model, block, task)
        gradient_factory = lambda functional=functional: iter_per_example_gradients(
            functional,
            batches,
            backend=config.moments.backend,
        )
        curvature = _curvature_operator(
            config,
            functional,
            curvature_batches,
            gradient_factory,
            example_count,
        )
        result = analyze_block_streaming(
            curvature_operator=curvature,
            gradient_factory=gradient_factory,
            rows=block.layout.matrix_shape[0],
            cols=block.layout.matrix_shape[1],
            example_count=example_count,
            config=block_config,
            output_dir=streaming_dir,
            metadata=metadata,
        )
        if config.streaming.run_interventions:
            _run_interventions(
                config,
                block,
                curvature,
                result,
                output_path=interventions_path,
                metadata=metadata,
                block_index=block_index,
            )
        append_jsonl_strict(
            {
                **metadata,
                "status": "completed",
                "raw_gradients_retained": False,
                "curvature_censored": result.row["curvature_censored"],
                "adam_censored": result.row["adam_censored"],
                "shampoo_censored": result.row["shampoo_censored"],
            },
            streaming_dir / "progress.jsonl",
        )
        completed += 1
        del result, curvature, functional
        if torch.cuda.is_available() and torch.device(config.run.device).type == "cuda":
            torch.cuda.empty_cache()

    return {
        "run_dir": str(directory),
        "geometry_csv": str(geometry_path),
        "interventions_csv": str(interventions_path) if interventions_path.exists() else None,
        "manifest_json": str(streaming_dir / "manifest.json"),
        "progress_jsonl": str(streaming_dir / "progress.jsonl"),
        "blocks_total": len(blocks),
        "blocks_completed": completed,
        "blocks_censored": censored,
        "checkpoint": checkpoint,
    }
