from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import torch
import torch.nn.functional as F


Batch = Any


def _map_tensors(tree: Any, function) -> Any:
    if isinstance(tree, torch.Tensor):
        return function(tree)
    if isinstance(tree, dict):
        return {key: _map_tensors(value, function) for key, value in tree.items()}
    if isinstance(tree, tuple):
        return tuple(_map_tensors(value, function) for value in tree)
    if isinstance(tree, list):
        return [_map_tensors(value, function) for value in tree]
    return tree


def _first_tensor(tree: Any) -> torch.Tensor:
    if isinstance(tree, torch.Tensor):
        return tree
    if isinstance(tree, dict):
        for value in tree.values():
            try:
                return _first_tensor(value)
            except ValueError:
                continue
    if isinstance(tree, (tuple, list)):
        for value in tree:
            try:
                return _first_tensor(value)
            except ValueError:
                continue
    raise ValueError("Batch contains no tensor")


def _extract_logits(output: Any) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, dict) and "logits" in output:
        return output["logits"]
    if hasattr(output, "logits"):
        return output.logits
    if isinstance(output, (tuple, list)) and output and isinstance(output[0], torch.Tensor):
        return output[0]
    raise TypeError(f"Cannot extract logits from output type {type(output)!r}")


class TaskAdapter(Protocol):
    def call_spec(self, batch: Batch) -> tuple[tuple[Any, ...], dict[str, Any]]: ...

    def extract_logits(self, output: Any) -> torch.Tensor: ...

    def loss_per_example_from_logits(self, logits: torch.Tensor, batch: Batch) -> torch.Tensor: ...

    def output_hessian_action(
        self, fixed_logits: torch.Tensor, direction: torch.Tensor, batch: Batch
    ) -> torch.Tensor: ...

    def batch_size(self, batch: Batch) -> int: ...

    def slice_batch(self, batch: Batch, index: int, *, keepdim: bool) -> Batch: ...

    def ensure_batched(self, sample: Batch) -> Batch: ...


@dataclass(frozen=True)
class ClassificationTask:
    input_key: str = "inputs"
    target_key: str = "targets"

    def call_spec(self, batch: Batch) -> tuple[tuple[Any, ...], dict[str, Any]]:
        if isinstance(batch, dict):
            return (batch[self.input_key],), {}
        if isinstance(batch, (tuple, list)) and len(batch) >= 2:
            return (batch[0],), {}
        raise TypeError("Classification batch must be a mapping or (inputs, targets)")

    def targets(self, batch: Batch) -> torch.Tensor:
        if isinstance(batch, dict):
            return batch[self.target_key]
        return batch[1]

    def extract_logits(self, output: Any) -> torch.Tensor:
        return _extract_logits(output)

    def loss_per_example_from_logits(self, logits: torch.Tensor, batch: Batch) -> torch.Tensor:
        targets = self.targets(batch)
        return F.cross_entropy(logits, targets, reduction="none")

    def output_hessian_action(
        self, fixed_logits: torch.Tensor, direction: torch.Tensor, batch: Batch
    ) -> torch.Tensor:
        probabilities = torch.softmax(fixed_logits, dim=-1)
        centered = direction - (probabilities * direction).sum(dim=-1, keepdim=True)
        return probabilities * centered / fixed_logits.shape[0]

    def batch_size(self, batch: Batch) -> int:
        return int(_first_tensor(batch).shape[0])

    def slice_batch(self, batch: Batch, index: int, *, keepdim: bool = True) -> Batch:
        def select(tensor: torch.Tensor) -> torch.Tensor:
            return tensor[index : index + 1] if keepdim else tensor[index]

        return _map_tensors(batch, select)

    def ensure_batched(self, sample: Batch) -> Batch:
        return _map_tensors(sample, lambda tensor: tensor.unsqueeze(0))


@dataclass(frozen=True)
class CausalLMTask:
    input_key: str = "input_ids"
    target_key: str = "labels"
    ignore_index: int = -100

    def call_spec(self, batch: Batch) -> tuple[tuple[Any, ...], dict[str, Any]]:
        if not isinstance(batch, dict):
            raise TypeError("CausalLM batch must be a mapping")
        kwargs: dict[str, Any] = {}
        for key in ("attention_mask", "position_ids"):
            if key in batch:
                kwargs[key] = batch[key]
        return (batch[self.input_key],), kwargs

    def targets(self, batch: Batch) -> torch.Tensor:
        return batch[self.target_key]

    def extract_logits(self, output: Any) -> torch.Tensor:
        return _extract_logits(output)

    def loss_per_example_from_logits(self, logits: torch.Tensor, batch: Batch) -> torch.Tensor:
        targets = self.targets(batch)
        if logits.shape[:-1] != targets.shape:
            raise ValueError(
                f"Logit prefix {tuple(logits.shape[:-1])} must match targets {tuple(targets.shape)}"
            )
        token_losses = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            targets.reshape(-1),
            reduction="none",
            ignore_index=self.ignore_index,
        ).reshape(targets.shape)
        mask = targets.ne(self.ignore_index)
        counts = mask.sum(dim=-1).clamp_min(1)
        return (token_losses * mask).sum(dim=-1) / counts

    def output_hessian_action(
        self, fixed_logits: torch.Tensor, direction: torch.Tensor, batch: Batch
    ) -> torch.Tensor:
        targets = self.targets(batch)
        probabilities = torch.softmax(fixed_logits, dim=-1)
        centered = direction - (probabilities * direction).sum(dim=-1, keepdim=True)
        result = probabilities * centered
        mask = targets.ne(self.ignore_index)
        counts = mask.sum(dim=-1).clamp_min(1)
        scale = mask.to(result.dtype) / (fixed_logits.shape[0] * counts[:, None])
        return result * scale.unsqueeze(-1)

    def batch_size(self, batch: Batch) -> int:
        return int(_first_tensor(batch).shape[0])

    def slice_batch(self, batch: Batch, index: int, *, keepdim: bool = True) -> Batch:
        def select(tensor: torch.Tensor) -> torch.Tensor:
            return tensor[index : index + 1] if keepdim else tensor[index]

        return _map_tensors(batch, select)

    def ensure_batched(self, sample: Batch) -> Batch:
        return _map_tensors(sample, lambda tensor: tensor.unsqueeze(0))
