from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
import importlib
import json
import math
from pathlib import Path
import random
import re
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from .blocks import BlockSpec, MatrixLayout, discover_matrix_blocks
from .config import ExperimentConfig, save_config
from .curvature import (
    CrossEntropyGGNOperator,
    EmpiricalFisherOperator,
    ExactHessianOperator,
    RegularizedOperator,
)
from .data import SyntheticLanguageDataset, SyntheticVisionDataset, TensorFileDataset
from .diagnostics import (
    adam_statistic_operator,
    curvature_factor_proxies_from_ritz,
    eigenspace_overlap,
    matched_response,
    nci,
    projected_commutator,
    shampoo_statistic_operator,
)
from .dynamics import (
    quadratic_energy,
    run_chebyshev,
    run_conjugate_gradient,
    run_frozen_block_continuation,
    run_gradient_descent,
)
from .same_error_dynamics import original_error_fingerprint, transformed_initial_for_same_error
from .functional import FunctionalBlockModel, collect_per_example_gradients
from .geometry import GeometryEstimate, measure_frozen_geometry
from .interventions import (
    alpha_sweep,
    assignment_sweep,
    damping_sweep,
    finite_sample_sweep,
    reassign_factor_spectrum,
    staleness_sweep,
)
from .io import (
    atomic_write_json,
    environment_manifest,
    safe_block_name,
    save_tensor_bundle,
    write_dataframe,
)
from .models import TinyDecoderLM, TinyVisionTransformer
from .moments import MomentEstimate, estimate_moments_from_gradients
from .operators import CongruenceOperator, DenseOperator, SymmetricLinearOperator, materialize
from .optimizer_state import OptimizerStateSnapshot, extract_frozen_preconditioner
from .preconditioners import (
    AdamPreconditioner,
    FrozenPreconditioner,
    IdentityPreconditioner,
    ScaledPreconditioner,
    ShampooPreconditioner,
    TiledShampooPreconditioner,
)
from .reporting import (
    aggregate_geometry_files,
    plot_checkpoint_heatmap,
    plot_continuation_curves,
    plot_dynamics_curves,
    delta_gain_column,
    plot_geometry_delta_gain,
    plot_intervention_conditions,
    plot_staleness,
    plot_synthetic_fan,
    summarize_hypotheses,
)
from .spectral import lanczos, slq_spectrum
from .statistics import (
    cluster_bootstrap_sign_fractions,
    leave_one_cluster_out_prediction,
    paired_intervention_effects,
)
from .spectrum_utils import factorwise_damping, positive_spectrum_summary
from .tasks import CausalLMTask, ClassificationTask, TaskAdapter
from .training import load_checkpoint, move_batch, train_and_checkpoint


@dataclass
class BlockAnalysis:
    block: BlockSpec
    gradients: torch.Tensor
    moments: MomentEstimate
    curvature: SymmetricLinearOperator
    curvature_matrix: torch.Tensor | None
    eigenvalues: torch.Tensor
    eigenvectors: torch.Tensor
    centered: bool
    shampoo_damping_left: float
    shampoo_damping_right: float
    preconditioners: dict[str, FrozenPreconditioner]
    geometries: dict[str, GeometryEstimate]
    functional: FunctionalBlockModel
    batch: Any
    optimizer_state: OptimizerStateSnapshot | None = None

    @property
    def shampoo_damping(self) -> float:
        """Backward-compatible scalar summary; do not use for new analyses."""

        return max(self.shampoo_damping_left, self.shampoo_damping_right)


def _dtype(name: str) -> torch.dtype:
    mapping = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float64": torch.float64,
        "fp64": torch.float64,
        "bfloat16": torch.bfloat16,
    }
    try:
        return mapping[name.lower()]
    except KeyError as error:
        raise ValueError(f"Unsupported dtype: {name}") from error


def _seed_everything(seed: int, deterministic: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)


def _import_object(path: str) -> Any:
    module_name, _, attribute = path.rpartition(".")
    if not module_name:
        raise ValueError(f"Expected dotted import path, got {path!r}")
    module = importlib.import_module(module_name)
    return getattr(module, attribute)


def build_model(config: ExperimentConfig) -> torch.nn.Module:
    family = config.model.family.lower()
    kwargs = dict(config.model.kwargs)
    if family in {"decoder", "tiny_decoder", "causal_lm"}:
        return TinyDecoderLM(**kwargs)
    if family in {"vit", "tiny_vit", "vision_transformer"}:
        return TinyVisionTransformer(**kwargs)
    if family in {"python", "factory"}:
        factory_path = kwargs.pop("factory")
        return _import_object(factory_path)(**kwargs)
    raise ValueError(f"Unsupported model family: {config.model.family}")


def build_dataset(config: ExperimentConfig) -> Dataset:
    family = config.data.family.lower()
    kwargs = dict(config.data.kwargs)
    kwargs.setdefault("num_examples", config.data.num_examples)
    kwargs.setdefault("seed", config.run.seed)
    if family in {"synthetic_language", "language", "causal_lm"}:
        return SyntheticLanguageDataset(**kwargs)
    if family in {"synthetic_vision", "vision", "classification"}:
        return SyntheticVisionDataset(**kwargs)
    if family in {"tensor_file", "pt"}:
        if config.data.path is None:
            raise ValueError("data.path is required for tensor_file datasets")
        return TensorFileDataset(config.data.path)
    if family in {"python", "factory"}:
        factory_path = kwargs.pop("factory")
        return _import_object(factory_path)(**kwargs)
    raise ValueError(f"Unsupported data family: {config.data.family}")


def build_task(config: ExperimentConfig) -> TaskAdapter:
    family = config.task.family.lower()
    if family in {"causal_lm", "language", "lm"}:
        input_key = (
            config.task.input_key
            if config.task.input_key != "inputs"
            else "input_ids"
        )
        target_key = (
            config.task.target_key
            if config.task.target_key != "targets"
            else "labels"
        )
        return CausalLMTask(
            input_key=input_key,
            target_key=target_key,
            ignore_index=config.task.ignore_index,
        )
    if family in {"classification", "vision"}:
        return ClassificationTask(
            input_key=config.task.input_key,
            target_key=config.task.target_key,
        )
    if family in {"python", "factory"}:
        kwargs = dict(config.task.kwargs)
        factory_path = kwargs.pop("factory")
        return _import_object(factory_path)(**kwargs)
    if family != "auto":
        raise ValueError(f"Unsupported task family: {config.task.family}")
    if config.model.family.lower() in {"decoder", "tiny_decoder", "causal_lm"} or config.data.family.lower() in {
        "synthetic_language",
        "language",
        "causal_lm",
    }:
        return CausalLMTask()
    return ClassificationTask()


def run_directory(config: ExperimentConfig) -> Path:
    return Path(config.output_dir).expanduser().resolve() / config.run.name


def _prepare_run(config: ExperimentConfig) -> Path:
    directory = run_directory(config)
    directory.mkdir(parents=True, exist_ok=True)
    resolved = directory / "resolved_config.yaml"
    save_config(config, resolved)
    manifest = environment_manifest(
        project_dir=Path(__file__).resolve().parents[3],
        config_path=resolved,
    )
    manifest.update(
        {
            "run_name": config.run.name,
            "seed": config.run.seed,
            "device": config.run.device,
            "dtype": config.run.dtype,
        }
    )
    atomic_write_json(manifest, directory / "manifest.json")
    return directory


def _cast_batch(batch: Any, dtype: torch.dtype) -> Any:
    if isinstance(batch, torch.Tensor):
        return batch.to(dtype=dtype) if batch.is_floating_point() else batch
    if isinstance(batch, dict):
        return {key: _cast_batch(value, dtype) for key, value in batch.items()}
    if isinstance(batch, tuple):
        return tuple(_cast_batch(value, dtype) for value in batch)
    if isinstance(batch, list):
        return [_cast_batch(value, dtype) for value in batch]
    return batch


def _probe_batch(config: ExperimentConfig, dataset: Dataset) -> Any:
    maximum = config.moments.max_examples or config.data.num_examples
    batch_size = min(int(maximum), len(dataset))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    batch = next(iter(loader))
    batch = move_batch(batch, config.run.device)
    return _cast_batch(batch, _dtype(config.run.dtype))


def run_training(config: ExperimentConfig, *, directory: Path | None = None) -> dict[str, Any]:
    _seed_everything(config.run.seed, config.run.deterministic)
    directory = directory or _prepare_run(config)
    model = build_model(config).to(device=config.run.device, dtype=_dtype(config.run.dtype))
    dataset = build_dataset(config)
    task = build_task(config)
    training_dir = directory / "training"
    result = train_and_checkpoint(
        model,
        dataset,
        task,
        output_dir=training_dir / "checkpoints",
        optimizer_name=config.training.optimizer,
        steps=config.training.steps,
        batch_size=config.data.batch_size,
        learning_rate=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
        checkpoint_steps=config.training.checkpoint_steps,
        seed=config.run.seed,
        device=config.run.device,
        beta1=config.training.beta1,
        beta2=config.training.beta2,
        epsilon=config.training.epsilon,
        alpha=config.training.alpha,
        root_frequency=config.training.root_frequency,
        grafting=config.training.grafting,
    )
    training_frame = pd.DataFrame(
        {
            "step": np.arange(1, len(result.losses) + 1, dtype=int),
            "loss": result.losses,
            "optimizer": result.optimizer_name,
        }
    )
    write_dataframe(training_frame, training_dir / "training.csv")
    final_checkpoint = result.checkpoints[-1]
    (directory / "latest_checkpoint.txt").write_text(str(final_checkpoint), encoding="utf-8")
    return {
        "run_dir": str(directory),
        "final_checkpoint": str(final_checkpoint),
        "training_csv": str(training_dir / "training.csv"),
        "checkpoints": [str(path) for path in result.checkpoints],
    }


def _resolve_checkpoint(config: ExperimentConfig, directory: Path) -> Path:
    if config.model.checkpoint:
        path = Path(config.model.checkpoint).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(path)
        return path
    pointer = directory / "latest_checkpoint.txt"
    if pointer.exists():
        path = Path(pointer.read_text(encoding="utf-8").strip())
        if path.exists():
            return path
    raise ValueError("No checkpoint specified and no latest_checkpoint.txt found")


def _curvature_operator(
    config: ExperimentConfig,
    functional: FunctionalBlockModel,
    batch: Any,
    gradients: torch.Tensor,
) -> SymmetricLinearOperator:
    kind = config.curvature.kind.lower()
    if kind in {"fisher", "empirical_fisher"}:
        operator: SymmetricLinearOperator = EmpiricalFisherOperator(
            gradients, functional.block.layout
        )
    elif kind in {"ggn", "generalized_gauss_newton"}:
        operator = CrossEntropyGGNOperator(functional, batch)
    elif kind in {"hessian", "exact_hessian"}:
        operator = ExactHessianOperator(functional, batch)
    else:
        raise ValueError(f"Unsupported curvature kind: {config.curvature.kind}")
    if config.curvature.shift > 0:
        operator = RegularizedOperator(operator, config.curvature.shift)
    return operator


def _factor_damping_or_zero(
    left: torch.Tensor,
    right: torch.Tensor,
    *,
    normalized_ratio: float,
    relative_threshold: float,
) -> tuple[float, float, str | None]:
    try:
        damping = factorwise_damping(
            left,
            right,
            normalized_ratio=normalized_ratio,
            relative_threshold=relative_threshold,
        )
    except ValueError as error:
        return 0.0, 0.0, str(error)
    return damping.left, damping.right, None


def _curvature_eigenpairs(
    operator: SymmetricLinearOperator,
    config: ExperimentConfig,
) -> tuple[torch.Tensor | None, torch.Tensor, torch.Tensor, SymmetricLinearOperator]:
    if operator.dimension <= config.curvature.exact_max_dim:
        matrix = materialize(operator)
        values, vectors = torch.linalg.eigh(matrix)
        return matrix, values, vectors, DenseOperator(matrix, name=operator.name)
    result = lanczos(
        operator,
        steps=min(operator.dimension, max(config.curvature.lanczos_steps, config.geometry.response_vectors * 2)),
        seed=config.run.seed,
    )
    return None, result.ritz_values, result.ritz_vectors, operator


def _select_response_vectors(
    eigenvalues: torch.Tensor,
    eigenvectors: torch.Tensor,
    count: int,
    threshold: float,
) -> torch.Tensor:
    positive_indices = torch.nonzero(eigenvalues > threshold, as_tuple=False).flatten()
    if positive_indices.numel() < 2:
        return eigenvectors[:, -min(2, eigenvectors.shape[1]) :]
    target = min(count, int(positive_indices.numel()))
    positions = torch.linspace(0, positive_indices.numel() - 1, target, device=positive_indices.device)
    chosen = positive_indices[positions.round().to(torch.long)]
    return eigenvectors[:, chosen]


def _covariance_matrix(gradients: torch.Tensor, *, centered: bool) -> torch.Tensor:
    flat = gradients.reshape(gradients.shape[0], -1)
    if centered:
        flat = flat - flat.mean(dim=0, keepdim=True)
    return flat.T @ flat / flat.shape[0]


def _safe_response(
    curvature: SymmetricLinearOperator,
    statistic: SymmetricLinearOperator,
    vectors: torch.Tensor,
) -> tuple[float, float]:
    try:
        result = matched_response(curvature, statistic, vectors)
        return result.slope, result.spearman
    except ValueError:
        return math.nan, math.nan


def _analyze_blocks(
    config: ExperimentConfig,
    *,
    directory: Path,
) -> tuple[pd.DataFrame, list[BlockAnalysis]]:
    _seed_everything(config.run.seed, config.run.deterministic)
    model = build_model(config).to(device=config.run.device, dtype=_dtype(config.run.dtype))
    checkpoint = _resolve_checkpoint(config, directory)
    checkpoint_payload = load_checkpoint(model, checkpoint, map_location=config.run.device)
    model.eval()
    dataset = build_dataset(config)
    task = build_task(config)
    batch = _probe_batch(config, dataset)
    save_tensor_bundle({"batch": move_batch(batch, "cpu")}, directory / "probe_batch.pt")

    blocks = discover_matrix_blocks(
        model,
        include=config.blocks.include,
        exclude=config.blocks.exclude,
        min_numel=config.blocks.min_numel,
        max_numel=config.blocks.max_numel,
    )
    if not blocks:
        raise ValueError("No matrix-shaped blocks matched the configured patterns")

    geometry_dir = directory / "geometry"
    artifact_dir = geometry_dir / "blocks"
    spectra_dir = geometry_dir / "spectra"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    spectra_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    analyses: list[BlockAnalysis] = []

    for block_index, block in enumerate(blocks):
        functional = FunctionalBlockModel(model, block, task)
        gradients = collect_per_example_gradients(
            functional, batch, backend=config.moments.backend
        ).to(torch.float64)
        moments = estimate_moments_from_gradients(
            gradients,
            block.layout,
            accumulation_dtype=_dtype(config.moments.accumulation_dtype),
        )
        centered = bool(config.moments.centered)
        left, right = moments.factors(centered=centered)
        diagonal = moments.diagonal(centered=centered)

        curvature_raw = _curvature_operator(config, functional, batch, gradients)
        curvature_matrix, eigenvalues, eigenvectors, curvature = _curvature_eigenpairs(
            curvature_raw, config
        )
        response_vectors = _select_response_vectors(
            eigenvalues,
            eigenvectors,
            config.geometry.response_vectors,
            config.curvature.positive_threshold,
        )
        shampoo_damping_left, shampoo_damping_right, damping_censor_reason = (
            _factor_damping_or_zero(
                left,
                right,
                normalized_ratio=config.geometry.shampoo_damping_ratio,
                relative_threshold=config.curvature.positive_threshold,
            )
        )
        left_summary = positive_spectrum_summary(
            left, relative_threshold=config.curvature.positive_threshold
        )
        right_summary = positive_spectrum_summary(
            right, relative_threshold=config.curvature.positive_threshold
        )

        preconditioners: dict[str, FrozenPreconditioner] = {
            "identity": IdentityPreconditioner(
                block.layout, dtype=curvature.dtype, device=curvature.device
            ),
            "adam": AdamPreconditioner(
                diagonal,
                layout=block.layout,
                damping=config.geometry.adam_damping,
            ),
        }
        for alpha in config.geometry.alpha_values:
            preconditioners[f"shampoo_{alpha:g}"] = ShampooPreconditioner(
                left,
                right,
                layout=block.layout,
                alpha=float(alpha),
                damping_left=shampoo_damping_left,
                damping_right=shampoo_damping_right,
                name=f"shampoo-alpha-{alpha:g}",
            )
        optimizer_state = extract_frozen_preconditioner(
            checkpoint_payload, model, block
        )
        if optimizer_state is not None:
            preconditioners["optimizer_state"] = optimizer_state.preconditioner

        geometries: dict[str, GeometryEstimate] = {}
        for name, preconditioner in preconditioners.items():
            geometries[name] = measure_frozen_geometry(
                curvature,
                preconditioner,
                exact_max_dim=config.geometry.exact_condition_max_dim,
                lanczos_steps=config.curvature.lanczos_steps,
                lanczos_starts=config.curvature.lanczos_starts,
                seed=config.run.seed + 1009 * block_index,
                positive_threshold=config.curvature.positive_threshold,
                residual_tolerance=config.curvature.residual_tolerance,
                subspace_policy="strict_spd",
            )

        adam_statistic = adam_statistic_operator(
            diagonal, damping=config.geometry.adam_damping
        )
        shampoo_statistic = shampoo_statistic_operator(
            left,
            right,
            block.layout,
            damping_left=shampoo_damping_left,
            damping_right=shampoo_damping_right,
        )
        response_adam, response_adam_rank = _safe_response(
            curvature, adam_statistic, response_vectors
        )
        response_shampoo, response_shampoo_rank = _safe_response(
            curvature, shampoo_statistic, response_vectors
        )
        comm_adam = projected_commutator(curvature, adam_statistic, response_vectors)
        comm_shampoo = projected_commutator(curvature, shampoo_statistic, response_vectors)
        overlap_rank = max(1, min(3, curvature.dimension))
        overlap = eigenspace_overlap(
            curvature,
            shampoo_statistic,
            rank=overlap_rank,
            exact_max_dim=config.geometry.exact_condition_max_dim,
            seed=config.run.seed + block_index,
        )

        identity_geometry = geometries["identity"]
        adam_geometry = geometries["adam"]
        k_curvature = identity_geometry.effective_condition.condition_number
        row: dict[str, Any] = {
            "run_name": config.run.name,
            "checkpoint": str(checkpoint),
            "block_name": block.name,
            "block_shape": "x".join(map(str, block.shape)),
            "block_numel": block.numel,
            "curvature_kind": config.curvature.kind,
            "curvature_shift": config.curvature.shift,
            "moment_centered": centered,
            "num_examples": moments.num_examples,
            "shampoo_damping_left": shampoo_damping_left,
            "shampoo_damping_right": shampoo_damping_right,
            "shampoo_damping_ratio": config.geometry.shampoo_damping_ratio,
            "shampoo_damping_censored": damping_censor_reason is not None,
            "shampoo_damping_censor_reason": damping_censor_reason,
            "left_factor_active_rank": left_summary.active_rank,
            "left_factor_numerical_null_rank": left_summary.numerical_null_rank,
            "left_factor_lambda_min_active": left_summary.minimum_active,
            "left_factor_lambda_max_active": left_summary.maximum_active,
            "left_rho_over_min": (
                shampoo_damping_left / left_summary.minimum_active
                if left_summary.minimum_active
                else math.nan
            ),
            "left_rho_over_max": (
                shampoo_damping_left / left_summary.maximum_active
                if left_summary.maximum_active
                else math.nan
            ),
            "right_factor_active_rank": right_summary.active_rank,
            "right_factor_numerical_null_rank": right_summary.numerical_null_rank,
            "right_factor_lambda_min_active": right_summary.minimum_active,
            "right_factor_lambda_max_active": right_summary.maximum_active,
            "right_rho_over_min": (
                shampoo_damping_right / right_summary.minimum_active
                if right_summary.minimum_active
                else math.nan
            ),
            "right_rho_over_max": (
                shampoo_damping_right / right_summary.maximum_active
                if right_summary.maximum_active
                else math.nan
            ),
            "K_curvature": k_curvature,
            "K_adam": adam_geometry.effective_condition.condition_number,
            "G_adam": adam_geometry.gain,
            "response_adam": response_adam,
            "response_adam_spearman": response_adam_rank,
            "response_shampoo": response_shampoo,
            "response_shampoo_spearman": response_shampoo_rank,
            "projected_commutator_adam": comm_adam,
            "projected_commutator_shampoo": comm_shampoo,
            "leading_overlap_affinity": overlap.affinity,
            "curvature_censored": identity_geometry.effective_condition.censored,
            "adam_censored": adam_geometry.effective_condition.censored,
            "optimizer_state_kind": (
                optimizer_state.optimizer_name if optimizer_state is not None else None
            ),
            "optimizer_state_step": (
                optimizer_state.optimizer_step if optimizer_state is not None else math.nan
            ),
            "optimizer_state_scope": (
                optimizer_state.metadata.get("scope", "frozen_checkpoint_operator")
                if optimizer_state is not None
                else None
            ),
            "K_optimizer_state": (
                geometries["optimizer_state"].effective_condition.condition_number
                if optimizer_state is not None
                else math.nan
            ),
            "G_optimizer_state": (
                geometries["optimizer_state"].gain
                if optimizer_state is not None
                else math.nan
            ),
            "optimizer_state_censored": (
                geometries["optimizer_state"].effective_condition.censored
                if optimizer_state is not None
                else True
            ),
        }
        for alpha in config.geometry.alpha_values:
            key = f"shampoo_{alpha:g}"
            geometry = geometries[key]
            row[f"K_shampoo_{alpha:g}"] = geometry.effective_condition.condition_number
            row[f"G_shampoo_{alpha:g}"] = geometry.gain
            row[f"delta_G_{alpha:g}"] = (
                geometry.gain - adam_geometry.gain
                if geometry.gain is not None and adam_geometry.gain is not None
                else math.nan
            )
            row[f"shampoo_{alpha:g}_censored"] = geometry.effective_condition.censored

        if curvature_matrix is not None:
            covariance = _covariance_matrix(gradients, centered=centered)
            try:
                row["NCI_adam"] = nci(preconditioners["adam"], curvature_matrix, covariance)
            except ValueError:
                row["NCI_adam"] = math.nan
            for alpha in config.geometry.alpha_values:
                try:
                    row[f"NCI_shampoo_{alpha:g}"] = nci(
                        preconditioners[f"shampoo_{alpha:g}"], curvature_matrix, covariance
                    )
                except ValueError:
                    row[f"NCI_shampoo_{alpha:g}"] = math.nan
        else:
            row["NCI_adam"] = math.nan

        try:
            slq = slq_spectrum(
                curvature,
                probes=config.curvature.slq_probes,
                steps=config.curvature.slq_steps,
                seed=config.run.seed + block_index,
            )
            q01, q99 = slq.quantile(0.01), slq.quantile(0.99)
            row["K_curvature_q99_q01"] = q99 / q01 if q01 > 0 else math.nan
        except (ValueError, RuntimeError):
            row["K_curvature_q99_q01"] = math.nan

        rows.append(row)
        analysis = BlockAnalysis(
            block=block,
            gradients=gradients,
            moments=moments,
            curvature=curvature,
            curvature_matrix=curvature_matrix,
            eigenvalues=eigenvalues,
            eigenvectors=eigenvectors,
            centered=centered,
            shampoo_damping_left=shampoo_damping_left,
            shampoo_damping_right=shampoo_damping_right,
            preconditioners=preconditioners,
            geometries=geometries,
            functional=functional,
            batch=batch,
            optimizer_state=optimizer_state,
        )
        analyses.append(analysis)

        safe = safe_block_name(block.name)
        save_tensor_bundle(
            {
                "block": asdict(block),
                "gradients": gradients.cpu(),
                "moments": {
                    "mean_matrix": moments.mean_matrix.cpu(),
                    "centered_left": moments.centered_left.cpu(),
                    "centered_right": moments.centered_right.cpu(),
                    "centered_diag": moments.centered_diag.cpu(),
                    "uncentered_left": moments.uncentered_left.cpu(),
                    "uncentered_right": moments.uncentered_right.cpu(),
                    "uncentered_diag": moments.uncentered_diag.cpu(),
                },
                "curvature_matrix": curvature_matrix.cpu() if curvature_matrix is not None else None,
                "curvature_eigenvalues": eigenvalues.cpu(),
                "curvature_eigenvectors": eigenvectors.cpu(),
                "centered": centered,
                "shampoo_damping_left": shampoo_damping_left,
                "shampoo_damping_right": shampoo_damping_right,
                "shampoo_damping_ratio": config.geometry.shampoo_damping_ratio,
                "shampoo_damping_censor_reason": damping_censor_reason,
                "checkpoint": str(checkpoint),
                "optimizer_state": (
                    {
                        "optimizer_name": optimizer_state.optimizer_name,
                        "optimizer_step": optimizer_state.optimizer_step,
                        "metadata": optimizer_state.metadata,
                    }
                    if optimizer_state is not None
                    else None
                ),
            },
            artifact_dir / f"{safe}.pt",
        )
        np.savez_compressed(
            spectra_dir / f"{safe}.npz",
            curvature_eigenvalues=eigenvalues.detach().cpu().numpy(),
            left_factor_eigenvalues=torch.linalg.eigvalsh(left).detach().cpu().numpy(),
            right_factor_eigenvalues=torch.linalg.eigvalsh(right).detach().cpu().numpy(),
        )

    frame = pd.DataFrame(rows)
    write_dataframe(frame, geometry_dir / "geometry.csv")
    plot_geometry_delta_gain(frame, directory / "figures" / "geometry_delta_gain.pdf")
    return frame, analyses


def run_geometry(config: ExperimentConfig, *, directory: Path | None = None) -> dict[str, Any]:
    directory = directory or _prepare_run(config)
    frame, _ = _analyze_blocks(config, directory=directory)
    return {
        "run_dir": str(directory),
        "geometry_csv": str(directory / "geometry" / "geometry.csv"),
        "blocks": len(frame),
    }


def _intervention_rows(
    config: ExperimentConfig,
    analyses: Iterable[BlockAnalysis],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for block_index, analysis in enumerate(analyses):
        left, right = analysis.moments.factors(centered=analysis.centered)
        positive = analysis.eigenvalues > config.curvature.positive_threshold
        proxy_values = analysis.eigenvalues[positive]
        proxy_vectors = analysis.eigenvectors[:, positive]
        if proxy_values.numel() == 0:
            continue
        left_proxy, right_proxy = curvature_factor_proxies_from_ritz(
            proxy_values,
            proxy_vectors,
            analysis.block.layout,
            positive_only=True,
        )
        identity_geometry = analysis.geometries["identity"]
        adam_geometry = analysis.geometries["adam"]
        adam_gain = adam_geometry.gain
        common = {
            "run_name": config.run.name,
            "model_family": config.model.family,
            "seed": config.run.seed,
            "checkpoint": config.model.checkpoint,
            "block_name": analysis.block.name,
            "K_curvature": identity_geometry.effective_condition.condition_number,
            "K_adam": adam_geometry.effective_condition.condition_number,
            "G_adam": adam_gain,
            "curvature_censored": identity_geometry.effective_condition.censored,
            "adam_censored": adam_geometry.effective_condition.censored,
        }

        def append_measurement(
            *,
            intervention: str,
            branch: str,
            alpha: float | None,
            condition_number: float | None,
            gain: float | None,
            censored: bool,
            censor_reason: str | None,
            rho_over_min: float | None = None,
            rho_over_max: float | None = None,
            rho_left_over_min: float | None = None,
            rho_left_over_max: float | None = None,
            rho_right_over_min: float | None = None,
            rho_right_over_max: float | None = None,
            damping_left: float | None = None,
            damping_right: float | None = None,
            scale: float | None = None,
            sample_size: int | None = None,
        ) -> None:
            valid_gain = (
                gain is not None
                and adam_gain is not None
                and math.isfinite(float(gain))
                and math.isfinite(float(adam_gain))
                and not censored
            )
            rows.append(
                {
                    **common,
                    "intervention": intervention,
                    "branch": branch,
                    "alpha": alpha,
                    "rho_over_min": rho_over_min,
                    "rho_over_max": rho_over_max,
                    "rho_left_over_min": rho_left_over_min,
                    "rho_left_over_max": rho_left_over_max,
                    "rho_right_over_min": rho_right_over_min,
                    "rho_right_over_max": rho_right_over_max,
                    "damping_left": damping_left,
                    "damping_right": damping_right,
                    "scale": scale,
                    "sample_size": sample_size,
                    "condition_number": condition_number,
                    "K_shampoo": condition_number,
                    "gain": gain if not censored else None,
                    "G_shampoo": gain if not censored else None,
                    "delta_G": float(gain) - float(adam_gain) if valid_gain else None,
                    "censored": bool(censored),
                    "censor_reason": censor_reason,
                }
            )

        assignment_results = assignment_sweep(
            analysis.curvature,
            left,
            right,
            left_proxy,
            right_proxy,
            layout=analysis.block.layout,
            alpha=0.25,
            damping_left=analysis.shampoo_damping_left,
            damping_right=analysis.shampoo_damping_right,
            random_repeats=config.geometry.random_assignment_repeats,
            exact_max_dim=config.geometry.exact_condition_max_dim,
            lanczos_steps=config.curvature.lanczos_steps,
            lanczos_starts=config.curvature.lanczos_starts,
            seed=config.run.seed + block_index * 100,
            positive_threshold=config.curvature.positive_threshold,
            residual_tolerance=config.curvature.residual_tolerance,
        )
        assigned_factors: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        for branch in ("aligned", "reversed"):
            assigned_factors[branch] = (
                reassign_factor_spectrum(left, left_proxy, mode=branch),
                reassign_factor_spectrum(right, right_proxy, mode=branch),
            )
        for result in assignment_results:
            reason = (
                result.geometry.effective_condition.censor_reason
                if result.geometry is not None
                else None
            )
            append_measurement(
                intervention="assignment",
                branch=result.label,
                alpha=0.25,
                condition_number=result.condition_number,
                gain=result.gain,
                censored=result.censored,
                censor_reason=reason,
                damping_left=result.damping_left,
                damping_right=result.damping_right,
            )

        factor_sets = {"natural": (left, right), **assigned_factors}
        for branch, (branch_left, branch_right) in factor_sets.items():
            for result in alpha_sweep(
                analysis.curvature,
                branch_left,
                branch_right,
                layout=analysis.block.layout,
                alphas=config.geometry.alpha_values,
                damping_left=analysis.shampoo_damping_left,
                damping_right=analysis.shampoo_damping_right,
                exact_max_dim=config.geometry.exact_condition_max_dim,
                lanczos_steps=config.curvature.lanczos_steps,
                lanczos_starts=config.curvature.lanczos_starts,
                seed=config.run.seed + block_index * 1000,
                positive_threshold=config.curvature.positive_threshold,
                residual_tolerance=config.curvature.residual_tolerance,
            ):
                reason = (
                    result.geometry.effective_condition.censor_reason
                    if result.geometry is not None
                    else None
                )
                append_measurement(
                    intervention="alpha",
                    branch=branch,
                    alpha=result.alpha,
                    condition_number=result.condition_number,
                    gain=result.gain,
                    censored=result.censored,
                    censor_reason=reason,
                    damping_left=result.damping_left,
                    damping_right=result.damping_right,
                )
            for result in damping_sweep(
                analysis.curvature,
                branch_left,
                branch_right,
                layout=analysis.block.layout,
                alpha=0.25,
                normalized_ratios=config.geometry.damping_ratios,
                exact_max_dim=config.geometry.exact_condition_max_dim,
                lanczos_steps=config.curvature.lanczos_steps,
                lanczos_starts=config.curvature.lanczos_starts,
                seed=config.run.seed + block_index * 1000 + 17,
                positive_threshold=config.curvature.positive_threshold,
                residual_tolerance=config.curvature.residual_tolerance,
            ):
                reason = (
                    result.geometry.effective_condition.censor_reason
                    if result.geometry is not None
                    else None
                )
                append_measurement(
                    intervention="damping",
                    branch=branch,
                    alpha=0.25,
                    condition_number=result.condition_number,
                    gain=result.gain,
                    censored=result.censored,
                    censor_reason=reason,
                    rho_over_min=result.rho_over_min,
                    rho_over_max=result.rho_over_max,
                    rho_left_over_min=result.rho_left_over_min,
                    rho_left_over_max=result.rho_left_over_max,
                    rho_right_over_min=result.rho_right_over_min,
                    rho_right_over_max=result.rho_right_over_max,
                    damping_left=result.damping_left,
                    damping_right=result.damping_right,
                )

        natural = analysis.preconditioners.get("shampoo_0.25")
        if natural is not None:
            for scale in config.geometry.grafting_scales:
                geometry = measure_frozen_geometry(
                    analysis.curvature,
                    ScaledPreconditioner(natural, float(scale)),
                    exact_max_dim=config.geometry.exact_condition_max_dim,
                    lanczos_steps=config.curvature.lanczos_steps,
                    lanczos_starts=config.curvature.lanczos_starts,
                    seed=config.run.seed + block_index,
                    positive_threshold=config.curvature.positive_threshold,
                    residual_tolerance=config.curvature.residual_tolerance,
                    subspace_policy="strict_spd",
                )
                append_measurement(
                    intervention="grafting",
                    branch="natural",
                    alpha=0.25,
                    condition_number=geometry.effective_condition.condition_number,
                    gain=geometry.gain,
                    censored=geometry.censored,
                    censor_reason=geometry.effective_condition.censor_reason,
                    damping_left=analysis.shampoo_damping_left,
                    damping_right=analysis.shampoo_damping_right,
                    scale=float(scale),
                )

        sample_sizes = sorted(
            {max(2, analysis.gradients.shape[0] // 2), analysis.gradients.shape[0]}
        )
        for point in finite_sample_sweep(
            analysis.gradients,
            layout=analysis.block.layout,
            sample_sizes=sample_sizes,
            seed=config.run.seed + block_index,
        ):
            sample_left, sample_right = point.moments.factors(centered=analysis.centered)
            sample_damping_left, sample_damping_right, sample_damping_error = (
                _factor_damping_or_zero(
                    sample_left,
                    sample_right,
                    normalized_ratio=config.geometry.shampoo_damping_ratio,
                    relative_threshold=config.curvature.positive_threshold,
                )
            )
            sample_preconditioner = ShampooPreconditioner(
                sample_left,
                sample_right,
                layout=analysis.block.layout,
                alpha=0.25,
                damping_left=sample_damping_left,
                damping_right=sample_damping_right,
            )
            geometry = measure_frozen_geometry(
                analysis.curvature,
                sample_preconditioner,
                exact_max_dim=config.geometry.exact_condition_max_dim,
                lanczos_steps=config.curvature.lanczos_steps,
                lanczos_starts=config.curvature.lanczos_starts,
                seed=config.run.seed + block_index,
                positive_threshold=config.curvature.positive_threshold,
                residual_tolerance=config.curvature.residual_tolerance,
                subspace_policy="strict_spd",
            )
            censored = geometry.censored or sample_damping_error is not None
            append_measurement(
                intervention="finite_sample",
                branch="natural",
                alpha=0.25,
                condition_number=geometry.effective_condition.condition_number,
                gain=geometry.gain,
                censored=censored,
                censor_reason=sample_damping_error
                or geometry.effective_condition.censor_reason,
                rho_over_min=config.geometry.shampoo_damping_ratio,
                damping_left=sample_damping_left,
                damping_right=sample_damping_right,
                sample_size=point.sample_size,
            )

        if config.blocks.tile_rows and config.blocks.tile_cols:
            try:
                tiled = TiledShampooPreconditioner.from_per_example_gradients(
                    analysis.gradients,
                    layout=analysis.block.layout,
                    alpha=0.25,
                    damping=0.0,
                    damping_ratio=config.geometry.shampoo_damping_ratio,
                    relative_eigenvalue_floor=config.curvature.positive_threshold,
                    tile_rows=config.blocks.tile_rows,
                    tile_cols=config.blocks.tile_cols,
                    centered=analysis.centered,
                )
                geometry = measure_frozen_geometry(
                    analysis.curvature,
                    tiled,
                    exact_max_dim=config.geometry.exact_condition_max_dim,
                    lanczos_steps=config.curvature.lanczos_steps,
                    lanczos_starts=config.curvature.lanczos_starts,
                    seed=config.run.seed + block_index,
                    positive_threshold=config.curvature.positive_threshold,
                    residual_tolerance=config.curvature.residual_tolerance,
                    subspace_policy="strict_spd",
                )
                tile_error = None
            except ValueError as error:
                geometry = None
                tile_error = str(error)
            append_measurement(
                intervention="tiling",
                branch=f"{config.blocks.tile_rows}x{config.blocks.tile_cols}",
                alpha=0.25,
                condition_number=(
                    geometry.effective_condition.condition_number if geometry is not None else None
                ),
                gain=geometry.gain if geometry is not None else None,
                censored=tile_error is not None or (geometry.censored if geometry is not None else True),
                censor_reason=tile_error
                or (geometry.effective_condition.censor_reason if geometry is not None else None),
                rho_over_min=config.geometry.shampoo_damping_ratio,
            )
    return pd.DataFrame(rows)

def run_interventions(
    config: ExperimentConfig,
    *,
    directory: Path | None = None,
    analyses: list[BlockAnalysis] | None = None,
) -> dict[str, Any]:
    directory = directory or _prepare_run(config)
    if analyses is None:
        _, analyses = _analyze_blocks(config, directory=directory)
    frame = _intervention_rows(config, analyses)
    path = write_dataframe(frame, directory / "interventions" / "interventions.csv")
    plot_intervention_conditions(frame, directory / "figures" / "intervention_conditions.pdf")
    return {"run_dir": str(directory), "interventions_csv": str(path), "rows": len(frame)}


def _dynamics_rows(config: ExperimentConfig, analyses: Iterable[BlockAnalysis]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for block_index, analysis in enumerate(analyses):
        generator = torch.Generator(device=analysis.curvature.device.type).manual_seed(
            config.run.seed + 1543 * block_index
        )
        original_initial = torch.randn(
            analysis.curvature.dimension,
            generator=generator,
            dtype=analysis.curvature.dtype,
            device=analysis.curvature.device,
        )
        initial_fingerprint = original_error_fingerprint(original_initial)
        initial_objective = float(quadratic_energy(analysis.curvature, original_initial).item())
        steps = min(config.geometry.dynamics_steps, max(1, analysis.curvature.dimension))
        selected_names = ["identity", "adam"]
        if "optimizer_state" in analysis.preconditioners:
            selected_names.append("optimizer_state")
        for alpha in (0.25, 0.5):
            key = f"shampoo_{alpha:g}"
            if key in analysis.preconditioners:
                selected_names.append(key)
        for name in selected_names:
            preconditioner = analysis.preconditioners[name]
            effective = CongruenceOperator(
                analysis.curvature,
                preconditioner.apply_sqrt,
                name=f"effective-{name}",
            )
            geometry = analysis.geometries[name]
            if geometry.censored:
                continue
            minimum = geometry.effective_condition.min_eigenvalue
            maximum = geometry.effective_condition.max_eigenvalue
            if minimum is None or maximum is None or minimum <= 0:
                continue
            try:
                initial = transformed_initial_for_same_error(
                    preconditioner, original_initial
                )
            except (TypeError, ValueError, RuntimeError):
                continue
            transformed_objective = float(quadratic_energy(effective, initial).item())
            tolerance = 1e-8 * max(1.0, abs(initial_objective))
            if not math.isfinite(transformed_objective) or abs(
                transformed_objective - initial_objective
            ) > tolerance:
                continue
            trajectories = [
                run_gradient_descent(
                    effective,
                    initial,
                    steps=steps,
                    eigen_min=minimum,
                    eigen_max=maximum,
                ),
                run_chebyshev(
                    effective,
                    initial,
                    steps=steps,
                    eigen_min=minimum,
                    eigen_max=maximum,
                ),
                run_conjugate_gradient(effective, initial, steps=steps),
            ]
            for trajectory in trajectories:
                relative_values = trajectory.relative_objective.detach().cpu().tolist()
                gradient_norms = trajectory.gradient_norms.detach().cpu().tolist()
                step_sizes = trajectory.step_sizes.detach().cpu().tolist()
                for iteration, value in enumerate(relative_values):
                    step_size = (
                        float(step_sizes[iteration - 1])
                        if iteration > 0 and iteration - 1 < len(step_sizes)
                        else math.nan
                    )
                    rows.append(
                        {
                            "run_name": config.run.name,
                            "block_name": analysis.block.name,
                            "preconditioner": name,
                            "method": trajectory.method,
                            "iteration": iteration,
                            "relative_objective": value,
                            "gradient_norm": float(gradient_norms[iteration]),
                            "step_size": step_size,
                            "condition_number": geometry.effective_condition.condition_number,
                            "initial_error_sha256": initial_fingerprint,
                            "initial_objective": initial_objective,
                        }
                    )
    return pd.DataFrame(rows)


def run_dynamics(
    config: ExperimentConfig,
    *,
    directory: Path | None = None,
    analyses: list[BlockAnalysis] | None = None,
) -> dict[str, Any]:
    directory = directory or _prepare_run(config)
    if analyses is None:
        _, analyses = _analyze_blocks(config, directory=directory)
    frame = _dynamics_rows(config, analyses)
    path = write_dataframe(frame, directory / "dynamics" / "dynamics.csv")
    plot_dynamics_curves(frame, directory / "figures" / "dynamics_curves.pdf")
    return {"run_dir": str(directory), "dynamics_csv": str(path), "rows": len(frame)}



def _continuation_rows(
    config: ExperimentConfig,
    analyses: Iterable[BlockAnalysis],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for analysis in analyses:
        for name in config.continuation.preconditioners:
            preconditioner = analysis.preconditioners.get(name)
            geometry = analysis.geometries.get(name)
            if preconditioner is None or geometry is None:
                continue
            maximum = geometry.effective_condition.max_eigenvalue
            if maximum is None or maximum <= 0 or not math.isfinite(maximum):
                continue
            step_size = config.continuation.step_fraction / maximum
            result = run_frozen_block_continuation(
                analysis.functional,
                analysis.batch,
                preconditioner,
                steps=config.continuation.steps,
                step_size=step_size,
            )
            initial_loss = result.losses[0]
            for iteration, (loss, gradient_norm) in enumerate(
                zip(result.losses, result.gradient_norms)
            ):
                rows.append(
                    {
                        "run_name": config.run.name,
                        "block_name": analysis.block.name,
                        "preconditioner": name,
                        "iteration": iteration,
                        "loss": loss,
                        "relative_loss": loss / initial_loss if initial_loss != 0 else math.nan,
                        "gradient_norm": gradient_norm,
                        "step_size": step_size,
                        "effective_max_eigenvalue": maximum,
                        "condition_number": geometry.effective_condition.condition_number,
                    }
                )
    return pd.DataFrame(rows)


def run_continuations(
    config: ExperimentConfig,
    *,
    directory: Path | None = None,
    analyses: list[BlockAnalysis] | None = None,
) -> dict[str, Any]:
    directory = directory or _prepare_run(config)
    if analyses is None:
        _, analyses = _analyze_blocks(config, directory=directory)
    frame = _continuation_rows(config, analyses)
    path = write_dataframe(frame, directory / "continuations" / "continuations.csv")
    figure = plot_continuation_curves(
        frame, directory / "figures" / "continuation_curves.pdf"
    )
    return {
        "run_dir": str(directory),
        "continuations_csv": str(path),
        "figure": str(figure),
        "rows": len(frame),
    }


def run_synthetic(config: ExperimentConfig) -> dict[str, Any]:
    directory = _prepare_run(config)
    synthetic_dir = directory / "synthetic"
    rows: list[dict[str, Any]] = []
    kappas = [10.0, 100.0]
    alphas = [value for value in config.geometry.alpha_values if value in {0.25, 0.5}]
    if not alphas:
        alphas = [0.25, 0.5]
    for kappa in kappas:
        b = torch.diag(torch.tensor([1.0, kappa], dtype=torch.float64))
        hessian_matrix = torch.kron(b, b)
        hessian = DenseOperator(hessian_matrix)
        layout = MatrixLayout.from_shape((2, 2))
        k_h = kappa**2
        for alpha in alphas:
            r_values = [1.0, 2.0, 4.0] if alpha <= 0.25 else [1.0, 2.0]
            for r in r_values:
                plus_factor = torch.diag(torch.tensor([1.0, kappa**r], dtype=torch.float64))
                minus_factor = torch.diag(torch.tensor([kappa**r, 1.0], dtype=torch.float64))
                plus = ShampooPreconditioner(
                    plus_factor,
                    plus_factor,
                    layout=layout,
                    alpha=alpha,
                    damping_left=0.0,
                    damping_right=0.0,
                )
                minus = ShampooPreconditioner(
                    minus_factor,
                    minus_factor,
                    layout=layout,
                    alpha=alpha,
                    damping_left=0.0,
                    damping_right=0.0,
                )
                k_plus = measure_frozen_geometry(
                    hessian, plus, exact_max_dim=16
                ).effective_condition.condition_number
                k_minus = measure_frozen_geometry(
                    hessian, minus, exact_max_dim=16
                ).effective_condition.condition_number
                rows.append(
                    {
                        "experiment": "flat_kron_pair",
                        "kappa": kappa,
                        "K_H": k_h,
                        "alpha": alpha,
                        "r": r,
                        "rho": 0.0,
                        "K_plus": k_plus,
                        "K_minus": k_minus,
                        "K_plus_theory": k_h / (kappa ** (2 * alpha * r)),
                        "K_minus_theory": k_h * (kappa ** (2 * alpha * r)),
                    }
                )

        chi = kappa
        sigma = torch.diag(torch.tensor([chi**2, 1.0], dtype=torch.float64))
        information_layout = MatrixLayout.from_shape((1, 2))
        adam = AdamPreconditioner(
            torch.diag(sigma), layout=information_layout, damping=0.0
        )
        h_parallel = DenseOperator(torch.diag(torch.tensor([chi, 1.0], dtype=torch.float64)))
        h_perp = DenseOperator(torch.diag(torch.tensor([1.0, chi], dtype=torch.float64)))
        rows.append(
            {
                "experiment": "information_pair",
                "kappa": kappa,
                "K_H": chi,
                "alpha": 0.5,
                "r": 2.0,
                "rho": 0.0,
                "K_plus": measure_frozen_geometry(h_parallel, adam, exact_max_dim=16).effective_condition.condition_number,
                "K_minus": measure_frozen_geometry(h_perp, adam, exact_max_dim=16).effective_condition.condition_number,
                "K_plus_theory": 1.0,
                "K_minus_theory": chi**2,
            }
        )
    frame = pd.DataFrame(rows)
    results_path = write_dataframe(frame, synthetic_dir / "synthetic_results.csv")
    plot_synthetic_fan(frame, directory / "figures" / "synthetic_fan.pdf")
    return {
        "run_dir": str(directory),
        "results_csv": str(results_path),
        "rows": len(frame),
    }


def run_smoke(config: ExperimentConfig) -> dict[str, Any]:
    directory = _prepare_run(config)
    training = run_training(config, directory=directory)
    config.model.checkpoint = training["final_checkpoint"]
    save_config(config, directory / "resolved_config.yaml")
    geometry_frame, analyses = _analyze_blocks(config, directory=directory)
    intervention_frame = _intervention_rows(config, analyses)
    write_dataframe(intervention_frame, directory / "interventions" / "interventions.csv")
    plot_intervention_conditions(
        intervention_frame, directory / "figures" / "intervention_conditions.pdf"
    )
    dynamics_frame = _dynamics_rows(config, analyses)
    write_dataframe(dynamics_frame, directory / "dynamics" / "dynamics.csv")
    plot_dynamics_curves(dynamics_frame, directory / "figures" / "dynamics_curves.pdf")
    continuation_frame = _continuation_rows(config, analyses)
    write_dataframe(
        continuation_frame, directory / "continuations" / "continuations.csv"
    )
    plot_continuation_curves(
        continuation_frame, directory / "figures" / "continuation_curves.pdf"
    )
    summary = summarize_hypotheses(
        geometry_frame,
        intervention_frame,
        sign_threshold=config.geometry.sign_threshold,
    )
    atomic_write_json(summary, directory / "summary" / "hypotheses.json")
    return {
        "run_dir": str(directory),
        "final_checkpoint": training["final_checkpoint"],
        "geometry_rows": len(geometry_frame),
        "intervention_rows": len(intervention_frame),
        "dynamics_rows": len(dynamics_frame),
        "continuation_rows": len(continuation_frame),
    }


def run_aggregate(
    geometry_paths: Iterable[str | Path],
    *,
    output_dir: str | Path,
    intervention_paths: Iterable[str | Path] | None = None,
    sign_threshold: float = math.log(1.25),
    bootstrap_replicates: int = 2000,
    seed: int = 0,
) -> dict[str, Any]:
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    frame = aggregate_geometry_files(geometry_paths)
    csv_path = write_dataframe(frame, output / "geometry_aggregate.csv")
    figure_path = plot_geometry_delta_gain(frame, output / "geometry_delta_gain.pdf")

    try:
        delta_column = delta_gain_column(frame)
    except ValueError:
        delta_column = ""
    statistics_payload: dict[str, Any] = {}
    cluster_columns = (
        ["run_name"]
        if "run_name" in frame.columns and frame["run_name"].nunique() >= 2
        else ["checkpoint_step"]
        if "checkpoint_step" in frame.columns and frame["checkpoint_step"].nunique() >= 2
        else []
    )
    if delta_column in frame.columns and cluster_columns:
        statistics_payload["sign_fractions"] = cluster_bootstrap_sign_fractions(
            frame,
            delta_column=delta_column,
            cluster_columns=cluster_columns,
            threshold=sign_threshold,
            replicates=bootstrap_replicates,
            seed=seed,
        )
        predictor_columns = [
            column
            for column in ("response_shampoo", "projected_commutator_shampoo")
            if column in frame.columns
        ]
        if predictor_columns:
            try:
                prediction = leave_one_cluster_out_prediction(
                    frame,
                    target_column=delta_column,
                    predictor_columns=predictor_columns,
                    cluster_columns=cluster_columns,
                    sign_threshold=sign_threshold,
                )
                statistics_payload["mechanism_prediction"] = prediction.summary
                write_dataframe(
                    prediction.predictions, output / "mechanism_predictions.csv"
                )
            except ValueError as error:
                statistics_payload["mechanism_prediction"] = {
                    "status": "insufficient_data",
                    "reason": str(error),
                }
    else:
        statistics_payload["sign_fractions"] = {
            "status": "insufficient_data",
            "reason": "Need a delta_G column and at least two clusters",
        }
    interventions_csv: Path | None = None
    assignment_effects_csv: Path | None = None
    if intervention_paths is not None:
        intervention_frames: list[pd.DataFrame] = []
        for value in intervention_paths:
            source = Path(value).expanduser().resolve()
            current = pd.read_csv(source)
            if "run_name" not in current.columns:
                if source.parent.name in {"interventions", "checkpoint_sweep"}:
                    inferred_run = source.parent.parent.name
                else:
                    inferred_run = source.stem
                current.insert(0, "run_name", inferred_run)
            current["source_file"] = str(source)
            intervention_frames.append(current)
        if intervention_frames:
            intervention_frame = pd.concat(
                intervention_frames, ignore_index=True, sort=False
            )
            interventions_csv = write_dataframe(
                intervention_frame, output / "interventions_aggregate.csv"
            )
            group_columns = ["run_name"]
            if "checkpoint_step" in intervention_frame.columns:
                group_columns.append("checkpoint_step")
            group_columns.append("block_name")
            assignment_effects = paired_intervention_effects(
                intervention_frame,
                intervention="assignment",
                reference_branch="aligned",
                treatment_branch="reversed",
                value_column="condition_number",
                group_columns=group_columns,
            )
            assignment_effects_csv = write_dataframe(
                assignment_effects, output / "assignment_paired_effects.csv"
            )
            if assignment_effects.empty:
                statistics_payload["assignment_paired"] = {
                    "status": "insufficient_data",
                    "pairs": 0,
                }
            else:
                differences = assignment_effects["paired_difference"].to_numpy(
                    dtype=float
                )
                log_ratios = assignment_effects["paired_log_ratio"].dropna().to_numpy(
                    dtype=float
                )
                statistics_payload["assignment_paired"] = {
                    "status": "available",
                    "pairs": int(len(assignment_effects)),
                    "success_fraction": float(np.mean(differences > 0)),
                    "median_difference": float(np.median(differences)),
                    "median_log_ratio": (
                        float(np.median(log_ratios))
                        if len(log_ratios)
                        else math.nan
                    ),
                }

    statistics_payload["cluster_columns"] = cluster_columns
    statistics_payload["sign_threshold"] = sign_threshold
    statistics_path = atomic_write_json(statistics_payload, output / "statistics.json")
    return {
        "geometry_csv": str(csv_path),
        "figure": str(figure_path),
        "statistics_json": str(statistics_path),
        "interventions_csv": (
            str(interventions_csv) if interventions_csv is not None else None
        ),
        "assignment_effects_csv": (
            str(assignment_effects_csv)
            if assignment_effects_csv is not None
            else None
        ),
        "rows": len(frame),
    }



def _checkpoint_step(path: str | Path) -> int:
    match = re.search(r"checkpoint_step_(\d+)\.pt$", Path(path).name)
    if match is not None:
        return int(match.group(1))
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if "step" not in payload:
        raise ValueError(f"Checkpoint does not contain a step: {path}")
    return int(payload["step"])


def _staleness_rows(
    config: ExperimentConfig,
    checkpoint_analyses: list[tuple[int, list[BlockAnalysis]]],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    lags = sorted(set(int(value) for value in config.sweep.staleness_checkpoint_lags))
    if any(value < 0 for value in lags):
        raise ValueError("staleness checkpoint lags must be nonnegative")
    indexed = [
        (step, {analysis.block.name: analysis for analysis in analyses})
        for step, analyses in checkpoint_analyses
    ]
    for target_index, (target_step, target_map) in enumerate(indexed):
        for lag in lags:
            source_index = target_index - lag
            if source_index < 0:
                continue
            source_step, source_map = indexed[source_index]
            for block_index, block_name in enumerate(sorted(set(target_map) & set(source_map))):
                target = target_map[block_name]
                source = source_map[block_name]
                left, right = source.moments.factors(centered=source.centered)
                damping_left, damping_right, damping_error = _factor_damping_or_zero(
                    left,
                    right,
                    normalized_ratio=config.geometry.shampoo_damping_ratio,
                    relative_threshold=config.curvature.positive_threshold,
                )
                point = staleness_sweep(
                    target.curvature,
                    [(f"source_step={source_step}", left, right)],
                    layout=target.block.layout,
                    alpha=config.sweep.staleness_alpha,
                    damping_left=damping_left,
                    damping_right=damping_right,
                    exact_max_dim=config.geometry.exact_condition_max_dim,
                    lanczos_steps=config.curvature.lanczos_steps,
                    lanczos_starts=config.curvature.lanczos_starts,
                    seed=config.run.seed + 10007 * target_index + block_index,
                    positive_threshold=config.curvature.positive_threshold,
                    residual_tolerance=config.curvature.residual_tolerance,
                )[0]
                fresh_name = f"shampoo_{config.sweep.staleness_alpha:g}"
                fresh_geometry = target.geometries.get(fresh_name)
                fresh_condition = (
                    fresh_geometry.effective_condition.condition_number
                    if fresh_geometry is not None
                    else None
                )
                ratio = (
                    point.condition_number / fresh_condition
                    if fresh_condition is not None
                    and fresh_condition > 0
                    and math.isfinite(point.condition_number)
                    else math.nan
                )
                rows.append(
                    {
                        "block_name": block_name,
                        "source_step": source_step,
                        "target_step": target_step,
                        "checkpoint_lag": lag,
                        "alpha": config.sweep.staleness_alpha,
                        "damping_left": damping_left,
                        "damping_right": damping_right,
                        "condition_number": point.condition_number,
                        "fresh_condition_number": fresh_condition,
                        "condition_ratio_to_fresh": ratio,
                        "gain": point.gain,
                        "censored": point.censored or damping_error is not None,
                        "censor_reason": damping_error
                        or (
                            point.geometry.effective_condition.censor_reason
                            if point.geometry is not None
                            else None
                        ),
                    }
                )
    return pd.DataFrame(rows)


def run_checkpoint_sweep(config: ExperimentConfig) -> dict[str, Any]:
    """Train once, analyze every requested checkpoint, and quantify factor staleness."""

    directory = _prepare_run(config)
    training = run_training(config, directory=directory)
    checkpoint_paths = [Path(value) for value in training["checkpoints"]]
    sweep_root = directory / "checkpoint_sweep"
    geometry_frames: list[pd.DataFrame] = []
    intervention_frames: list[pd.DataFrame] = []
    dynamics_frames: list[pd.DataFrame] = []
    continuation_frames: list[pd.DataFrame] = []
    checkpoint_analyses: list[tuple[int, list[BlockAnalysis]]] = []

    for checkpoint_index, checkpoint in enumerate(checkpoint_paths):
        step = _checkpoint_step(checkpoint)
        local_config = deepcopy(config)
        local_config.model.checkpoint = str(checkpoint)
        local_directory = sweep_root / f"step_{step:06d}"
        local_directory.mkdir(parents=True, exist_ok=True)
        save_config(local_config, local_directory / "resolved_config.yaml")
        geometry, analyses = _analyze_blocks(local_config, directory=local_directory)
        geometry.insert(0, "checkpoint_step", step)
        geometry.insert(1, "checkpoint_index", checkpoint_index)
        write_dataframe(geometry, local_directory / "geometry" / "geometry.csv")
        geometry_frames.append(geometry)
        checkpoint_analyses.append((step, analyses))

        if config.sweep.run_interventions:
            interventions = _intervention_rows(local_config, analyses)
            interventions.insert(0, "checkpoint_step", step)
            interventions.insert(1, "checkpoint_index", checkpoint_index)
            write_dataframe(
                interventions, local_directory / "interventions" / "interventions.csv"
            )
            intervention_frames.append(interventions)
        if config.sweep.run_dynamics:
            dynamics = _dynamics_rows(local_config, analyses)
            dynamics.insert(0, "checkpoint_step", step)
            dynamics.insert(1, "checkpoint_index", checkpoint_index)
            write_dataframe(dynamics, local_directory / "dynamics" / "dynamics.csv")
            dynamics_frames.append(dynamics)
        if config.sweep.run_continuations:
            continuations = _continuation_rows(local_config, analyses)
            continuations.insert(0, "checkpoint_step", step)
            continuations.insert(1, "checkpoint_index", checkpoint_index)
            write_dataframe(
                continuations, local_directory / "continuations" / "continuations.csv"
            )
            continuation_frames.append(continuations)

    combined_geometry = pd.concat(geometry_frames, ignore_index=True, sort=False)
    geometry_path = write_dataframe(combined_geometry, sweep_root / "geometry.csv")
    checkpoint_figure = plot_checkpoint_heatmap(
        combined_geometry, directory / "figures" / "checkpoint_delta_gain.pdf"
    )

    interventions_path: Path | None = None
    if intervention_frames:
        combined_interventions = pd.concat(
            intervention_frames, ignore_index=True, sort=False
        )
        interventions_path = write_dataframe(
            combined_interventions, sweep_root / "interventions.csv"
        )
    else:
        combined_interventions = pd.DataFrame(
            columns=[
                "run_name",
                "checkpoint_step",
                "block_name",
                "K_curvature",
                "intervention",
                "branch",
                "alpha",
                "condition_number",
                "gain",
            ]
        )
    dynamics_path: Path | None = None
    if dynamics_frames:
        combined_dynamics = pd.concat(dynamics_frames, ignore_index=True, sort=False)
        dynamics_path = write_dataframe(combined_dynamics, sweep_root / "dynamics.csv")
    continuations_path: Path | None = None
    if continuation_frames:
        combined_continuations = pd.concat(
            continuation_frames, ignore_index=True, sort=False
        )
        continuations_path = write_dataframe(
            combined_continuations, sweep_root / "continuations.csv"
        )

    staleness = _staleness_rows(config, checkpoint_analyses)
    staleness_path = write_dataframe(staleness, sweep_root / "staleness.csv")
    staleness_figure = plot_staleness(
        staleness, directory / "figures" / "staleness.pdf"
    )
    hypotheses = summarize_hypotheses(
        combined_geometry,
        combined_interventions,
        sign_threshold=config.geometry.sign_threshold,
    )
    hypotheses_path = atomic_write_json(
        hypotheses, directory / "summary" / "hypotheses.json"
    )
    return {
        "run_dir": str(directory),
        "geometry_csv": str(geometry_path),
        "interventions_csv": str(interventions_path) if interventions_path else None,
        "dynamics_csv": str(dynamics_path) if dynamics_path else None,
        "continuations_csv": str(continuations_path) if continuations_path else None,
        "staleness_csv": str(staleness_path),
        "checkpoint_figure": str(checkpoint_figure),
        "staleness_figure": str(staleness_figure),
        "hypotheses_json": str(hypotheses_path),
        "checkpoints": len(checkpoint_paths),
        "geometry_rows": len(combined_geometry),
        "staleness_rows": len(staleness),
    }
