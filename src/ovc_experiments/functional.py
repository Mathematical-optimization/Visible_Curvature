from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Callable, Iterable, Iterator
from typing import Any

import torch
from torch.func import functional_call

from .blocks import BlockSpec
from .tasks import Batch, TaskAdapter


@dataclass
class FunctionalBlockModel:
    model: torch.nn.Module
    block: BlockSpec
    task: TaskAdapter

    def __post_init__(self) -> None:
        parameters = dict(self.model.named_parameters())
        if self.block.name not in parameters:
            raise KeyError(f"Parameter {self.block.name!r} not found")
        self.base_parameters = {name: value.detach() for name, value in parameters.items()}
        self.base_buffers = {name: value.detach() for name, value in self.model.named_buffers()}
        self.block_parameter = parameters[self.block.name].detach().clone()
        self.model.eval()

    def _parameter_mapping(self, block_parameter: torch.Tensor) -> dict[str, torch.Tensor]:
        mapping = dict(self.base_parameters)
        mapping[self.block.name] = block_parameter
        return mapping

    def logits(self, block_parameter: torch.Tensor, batch: Batch) -> torch.Tensor:
        args, kwargs = self.task.call_spec(batch)
        output = functional_call(
            self.model,
            (self._parameter_mapping(block_parameter), self.base_buffers),
            args,
            kwargs,
            strict=False,
            tie_weights=True,
        )
        return self.task.extract_logits(output)

    def loss_per_example(self, block_parameter: torch.Tensor, batch: Batch) -> torch.Tensor:
        return self.task.loss_per_example_from_logits(self.logits(block_parameter, batch), batch)

    def mean_loss(self, block_parameter: torch.Tensor, batch: Batch) -> torch.Tensor:
        return self.loss_per_example(block_parameter, batch).mean()


def collect_per_example_gradients(
    functional: FunctionalBlockModel,
    batch: Batch,
    *,
    backend: str = "loop",
) -> torch.Tensor:
    def single_loss(block_parameter: torch.Tensor, sample: Batch) -> torch.Tensor:
        batched = functional.task.ensure_batched(sample)
        return functional.loss_per_example(block_parameter, batched).squeeze(0)

    gradient_function = torch.func.grad(single_loss)
    if backend == "vmap":
        try:
            return torch.func.vmap(gradient_function, in_dims=(None, 0))(
                functional.block_parameter, batch
            ).detach()
        except (RuntimeError, NotImplementedError, ValueError):
            # Some third-party modules do not provide vmap rules. The loop is
            # mathematically identical and remains the portable fallback.
            backend = "loop"
    if backend != "loop":
        raise ValueError(f"Unknown per-example gradient backend: {backend}")

    gradients: list[torch.Tensor] = []
    for index in range(functional.task.batch_size(batch)):
        sample = functional.task.slice_batch(batch, index, keepdim=False)
        gradients.append(gradient_function(functional.block_parameter, sample).detach())
    return torch.stack(gradients, dim=0)


def iter_per_example_gradients(
    functional: FunctionalBlockModel,
    batch_factory: Callable[[], Iterable[Batch]],
    *,
    backend: str = "loop",
) -> Iterator[torch.Tensor]:
    """Yield one block gradient at a time from a replayable batch factory.

    The largest temporary gradient tensor is bounded by one configured data
    batch. Calling this function again re-invokes ``batch_factory`` and
    therefore reproduces the same deterministic stream when the factory is
    deterministic.
    """

    for batch in batch_factory():
        gradients = collect_per_example_gradients(functional, batch, backend=backend)
        for gradient in gradients.unbind(dim=0):
            yield gradient.detach()
