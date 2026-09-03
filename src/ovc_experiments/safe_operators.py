from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import torch

Tensor = torch.Tensor


def _as_vector(x: Tensor) -> Tensor:
    return x.reshape(-1)


@dataclass(frozen=True)
class DiagonalOperator:
    """Symmetric matrix-free diagonal operator with O(d) storage."""

    diagonal: Tensor

    def __post_init__(self) -> None:
        d = _as_vector(self.diagonal)
        if d.numel() == 0:
            raise ValueError('diagonal must be non-empty')
        if not torch.isfinite(d).all():
            raise ValueError('diagonal contains non-finite values')
        object.__setattr__(self, 'diagonal', d)

    @property
    def dim(self) -> int:
        return int(self.diagonal.numel())

    @property
    def dimension(self) -> int:
        return self.dim

    @property
    def shape(self) -> tuple[int, int]:
        return (self.dim, self.dim)

    @property
    def dtype(self) -> torch.dtype:
        return self.diagonal.dtype

    @property
    def device(self) -> torch.device:
        return self.diagonal.device

    def matvec(self, vector: Tensor) -> Tensor:
        v = _as_vector(vector).to(device=self.device, dtype=self.dtype)
        if v.numel() != self.dim:
            raise ValueError(f'expected vector of length {self.dim}, got {v.numel()}')
        return self.diagonal * v

    def __matmul__(self, vector: Tensor) -> Tensor:
        return self.matvec(vector)

    def to_dense(self, *, max_dim: Optional[int] = None) -> Tensor:
        if max_dim is not None and self.dim > max_dim:
            raise MemoryError(f'refusing to materialize {self.dim}x{self.dim} diagonal operator')
        return torch.diag(self.diagonal)

    def quadratic_form(self, vector: Tensor) -> Tensor:
        v = _as_vector(vector).to(device=self.device, dtype=self.dtype)
        return torch.dot(v, self.diagonal * v)


@dataclass(frozen=True)
class FunctionOperator:
    """Small duck-typed symmetric operator used by the hardened utilities."""

    dimension: int
    function: Callable[[Tensor], Tensor]
    dtype: torch.dtype = torch.float64
    device: torch.device | str = torch.device('cpu')
    diagonal: Optional[Tensor] = None
    name: str = 'function_operator'

    def __post_init__(self) -> None:
        if self.dimension <= 0:
            raise ValueError('dimension must be positive')
        object.__setattr__(self, 'device', torch.device(self.device))
        if self.diagonal is not None:
            d = _as_vector(self.diagonal)
            if d.numel() != self.dimension:
                raise ValueError('diagonal length does not match dimension')
            object.__setattr__(self, 'diagonal', d.to(device=self.device, dtype=self.dtype))

    @property
    def dim(self) -> int:
        return int(self.dimension)

    @property
    def shape(self) -> tuple[int, int]:
        return (self.dimension, self.dimension)

    def matvec(self, vector: Tensor) -> Tensor:
        v = _as_vector(vector).to(device=self.device, dtype=self.dtype)
        if v.numel() != self.dimension:
            raise ValueError(f'expected vector of length {self.dimension}, got {v.numel()}')
        out = _as_vector(self.function(v))
        if out.numel() != self.dimension:
            raise ValueError('operator returned a vector with the wrong length')
        return out.to(device=self.device, dtype=self.dtype)

    def __matmul__(self, vector: Tensor) -> Tensor:
        return self.matvec(vector)
