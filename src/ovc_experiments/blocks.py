from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable, Sequence

import torch


@dataclass(frozen=True)
class MatrixLayout:
    """View an arbitrary tensor with at least two axes as a matrix.

    The first tensor axis is the matrix row axis; all remaining axes are
    flattened into the matrix column axis. This matches the common Shampoo
    convention for linear, embedding, and convolutional weights.
    """

    original_shape: tuple[int, ...]
    matrix_shape: tuple[int, int]

    @classmethod
    def from_shape(cls, shape: Sequence[int]) -> "MatrixLayout":
        normalized = tuple(int(value) for value in shape)
        if len(normalized) < 2:
            raise ValueError(f"Matrix-shaped block requires ndim >= 2, got {normalized}")
        rows = normalized[0]
        cols = 1
        for size in normalized[1:]:
            cols *= size
        return cls(original_shape=normalized, matrix_shape=(rows, cols))

    @property
    def numel(self) -> int:
        return self.matrix_shape[0] * self.matrix_shape[1]

    def to_matrix(self, tensor: torch.Tensor) -> torch.Tensor:
        if tuple(tensor.shape) != self.original_shape:
            raise ValueError(
                f"Expected tensor shape {self.original_shape}, got {tuple(tensor.shape)}"
            )
        return tensor.reshape(self.matrix_shape)

    def from_matrix(self, matrix: torch.Tensor) -> torch.Tensor:
        if tuple(matrix.shape) != self.matrix_shape:
            raise ValueError(
                f"Expected matrix shape {self.matrix_shape}, got {tuple(matrix.shape)}"
            )
        return matrix.reshape(self.original_shape)

    def flatten(self, tensor: torch.Tensor) -> torch.Tensor:
        return self.to_matrix(tensor).reshape(-1)

    def unflatten(self, vector: torch.Tensor) -> torch.Tensor:
        if vector.numel() != self.numel:
            raise ValueError(f"Expected {self.numel} elements, got {vector.numel()}")
        return self.from_matrix(vector.reshape(self.matrix_shape))


@dataclass(frozen=True)
class BlockSpec:
    name: str
    shape: tuple[int, ...]
    layout: MatrixLayout
    numel: int


def _matches_any(name: str, patterns: Iterable[str]) -> bool:
    return any(re.search(pattern, name) is not None for pattern in patterns)


def discover_matrix_blocks(
    model: torch.nn.Module,
    *,
    include: Sequence[str] | None = None,
    exclude: Sequence[str] | None = None,
    min_numel: int = 2,
    max_numel: int | None = None,
) -> list[BlockSpec]:
    include_patterns = list(include or [r".*"])
    exclude_patterns = list(exclude or [])
    result: list[BlockSpec] = []
    for name, parameter in model.named_parameters():
        if parameter.ndim < 2:
            continue
        if not _matches_any(name, include_patterns):
            continue
        if exclude_patterns and _matches_any(name, exclude_patterns):
            continue
        if parameter.numel() < min_numel:
            continue
        if max_numel is not None and parameter.numel() > max_numel:
            continue
        shape = tuple(int(value) for value in parameter.shape)
        layout = MatrixLayout.from_shape(shape)
        result.append(BlockSpec(name=name, shape=shape, layout=layout, numel=parameter.numel()))
    return result
