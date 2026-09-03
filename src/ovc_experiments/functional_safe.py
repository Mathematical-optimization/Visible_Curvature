from __future__ import annotations
from collections import defaultdict
from typing import Any, Mapping

import torch


def tied_parameter_groups(module: torch.nn.Module) -> list[tuple[str, ...]]:
    groups: dict[int, list[str]] = defaultdict(list)
    for name, parameter in module.named_parameters(remove_duplicate=False):
        groups[id(parameter)].append(name)
    return [tuple(names) for names in groups.values() if len(names) > 1]


def functional_call_tied(
    module: torch.nn.Module,
    parameters: Mapping[str, torch.Tensor],
    buffers: Mapping[str, torch.Tensor] | None,
    args: tuple[Any, ...],
    kwargs: Mapping[str, Any] | None = None,
    *,
    strict: bool = False,
) -> Any:
    """Alias-safe torch.func.functional_call for tied embedding/head models."""
    kwargs = {} if kwargs is None else dict(kwargs)
    state = (dict(parameters), {} if buffers is None else dict(buffers))
    return torch.func.functional_call(
        module,
        state,
        args,
        kwargs,
        tie_weights=True,
        strict=strict,
    )
