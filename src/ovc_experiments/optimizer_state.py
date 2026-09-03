from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from .blocks import BlockSpec
from .preconditioners import (
    ExplicitDiagonalPreconditioner,
    ExplicitFactorPreconditioner,
    FrozenPreconditioner,
)


@dataclass(frozen=True)
class OptimizerStateSnapshot:
    optimizer_name: str
    preconditioner: FrozenPreconditioner
    parameter_name: str
    optimizer_step: int
    metadata: dict[str, Any]


def _step_as_int(value: Any) -> int:
    if isinstance(value, torch.Tensor):
        return int(value.detach().cpu().item())
    return int(value)


def _state_and_group_by_name(
    checkpoint: dict[str, Any],
    model: torch.nn.Module,
) -> dict[str, tuple[dict[str, Any], dict[str, Any]]]:
    optimizer_state = checkpoint.get("optimizer_state")
    if not isinstance(optimizer_state, dict):
        return {}
    groups = optimizer_state.get("param_groups", [])
    state = optimizer_state.get("state", {})
    parameter_ids: list[int] = []
    group_by_id: dict[int, dict[str, Any]] = {}
    for group in groups:
        for parameter_id in group.get("params", []):
            normalized = int(parameter_id)
            parameter_ids.append(normalized)
            group_by_id[normalized] = group

    names = checkpoint.get("parameter_names")
    if not isinstance(names, list):
        names = [name for name, _parameter in model.named_parameters()]
    if len(parameter_ids) != len(names):
        raise ValueError(
            "Optimizer parameter order cannot be matched to model parameters: "
            f"{len(parameter_ids)} ids versus {len(names)} names"
        )
    result: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
    for parameter_id, name in zip(parameter_ids, names):
        parameter_state = state.get(parameter_id, {})
        if parameter_state:
            result[str(name)] = (parameter_state, group_by_id[parameter_id])
    return result


def extract_frozen_preconditioner(
    checkpoint: dict[str, Any],
    model: torch.nn.Module,
    block: BlockSpec,
) -> OptimizerStateSnapshot | None:
    """Recover the frozen anisotropic operator stored in an optimizer checkpoint.

    For AdamW this is the bias-corrected outside-root diagonal operator actually
    used by PyTorch AdamW. For the package's MatrixShampoo optimizer this is the
    saved pair of possibly stale root factors. Momentum, weight decay, and
    gradient-dependent grafting scales are intentionally excluded.
    """

    mapped = _state_and_group_by_name(checkpoint, model)
    entry = mapped.get(block.name)
    if entry is None:
        return None
    parameter_state, group = entry
    parameter = dict(model.named_parameters())[block.name]
    optimizer_name = str(checkpoint.get("optimizer_name", "unknown")).lower()
    step = _step_as_int(parameter_state.get("step", checkpoint.get("step", 0)))

    if optimizer_name == "adamw":
        statistic = parameter_state.get(
            "max_exp_avg_sq" if bool(group.get("amsgrad", False)) else "exp_avg_sq"
        )
        if not isinstance(statistic, torch.Tensor):
            return None
        beta2 = float(group.get("betas", (0.9, 0.999))[1])
        correction = max(1.0 - beta2**step, torch.finfo(parameter.dtype).eps)
        v_hat = statistic.to(dtype=parameter.dtype, device=parameter.device) / correction
        epsilon = float(group.get("eps", 1e-8))
        weights = 1.0 / (v_hat.sqrt() + epsilon)
        preconditioner = ExplicitDiagonalPreconditioner(
            weights.reshape(-1),
            layout=block.layout,
            name="adamw-checkpoint-state",
        )
        return OptimizerStateSnapshot(
            optimizer_name=optimizer_name,
            preconditioner=preconditioner,
            parameter_name=block.name,
            optimizer_step=step,
            metadata={
                "beta2": beta2,
                "epsilon": epsilon,
                "bias_corrected": True,
                "damping_convention": "outside_root",
            },
        )

    if optimizer_name == "shampoo":
        left_root = parameter_state.get("left_root")
        right_root = parameter_state.get("right_root")
        if not isinstance(left_root, torch.Tensor) or not isinstance(right_root, torch.Tensor):
            return None
        preconditioner = ExplicitFactorPreconditioner(
            left_root.to(dtype=parameter.dtype, device=parameter.device),
            right_root.to(dtype=parameter.dtype, device=parameter.device),
            layout=block.layout,
            name="shampoo-checkpoint-root",
        )
        return OptimizerStateSnapshot(
            optimizer_name=optimizer_name,
            preconditioner=preconditioner,
            parameter_name=block.name,
            optimizer_step=step,
            metadata={
                "alpha": float(group.get("alpha", 0.25)),
                "epsilon": float(group.get("epsilon", 1e-8)),
                "root_frequency": int(group.get("root_frequency", 1)),
                "grafting": str(group.get("grafting", "none")),
                "scope": "root_direction_before_gradient_dependent_grafting_scale",
            },
        )

    return None
