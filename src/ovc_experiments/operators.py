from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch


class SymmetricLinearOperator:
    """Minimal symmetric linear-operator protocol used throughout the package."""

    dimension: int
    dtype: torch.dtype
    device: torch.device
    name: str

    def matvec(self, vector: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def __call__(self, vector: torch.Tensor) -> torch.Tensor:
        return self.matvec(vector)

    def _validate_vector(self, vector: torch.Tensor) -> None:
        if vector.ndim != 1 or vector.numel() != self.dimension:
            raise ValueError(
                f"{self.name} expects a vector of shape ({self.dimension},), "
                f"got {tuple(vector.shape)}"
            )


@dataclass
class FunctionOperator(SymmetricLinearOperator):
    dimension: int
    matvec_fn: Callable[[torch.Tensor], torch.Tensor]
    dtype: torch.dtype
    device: torch.device
    name: str = "function-operator"

    def matvec(self, vector: torch.Tensor) -> torch.Tensor:
        self._validate_vector(vector)
        output = self.matvec_fn(vector)
        if output.ndim != 1 or output.numel() != self.dimension:
            raise ValueError(
                f"{self.name} returned shape {tuple(output.shape)}; "
                f"expected ({self.dimension},)"
            )
        return output


class DenseOperator(SymmetricLinearOperator):
    def __init__(
        self,
        matrix: torch.Tensor,
        *,
        name: str = "dense-operator",
        check_symmetric: bool = True,
        symmetry_tolerance: float = 1e-10,
    ) -> None:
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError(f"Expected square matrix, got {tuple(matrix.shape)}")
        if check_symmetric and not torch.allclose(
            matrix,
            matrix.T,
            atol=symmetry_tolerance,
            rtol=symmetry_tolerance,
        ):
            raise ValueError("DenseOperator requires a symmetric matrix")
        self.matrix = matrix
        self.dimension = int(matrix.shape[0])
        self.dtype = matrix.dtype
        self.device = matrix.device
        self.name = name

    def matvec(self, vector: torch.Tensor) -> torch.Tensor:
        self._validate_vector(vector)
        return self.matrix @ vector


class ShiftedOperator(SymmetricLinearOperator):
    def __init__(self, base: SymmetricLinearOperator, shift: float, *, name: str | None = None) -> None:
        self.base = base
        self.shift = float(shift)
        self.dimension = base.dimension
        self.dtype = base.dtype
        self.device = base.device
        self.name = name or f"{base.name}+{self.shift:g}I"

    def matvec(self, vector: torch.Tensor) -> torch.Tensor:
        self._validate_vector(vector)
        return self.base.matvec(vector) + self.shift * vector


class ScaledOperator(SymmetricLinearOperator):
    def __init__(self, base: SymmetricLinearOperator, scale: float, *, name: str | None = None) -> None:
        self.base = base
        self.scale = float(scale)
        self.dimension = base.dimension
        self.dtype = base.dtype
        self.device = base.device
        self.name = name or f"{self.scale:g}*{base.name}"

    def matvec(self, vector: torch.Tensor) -> torch.Tensor:
        self._validate_vector(vector)
        return self.scale * self.base.matvec(vector)


class CongruenceOperator(SymmetricLinearOperator):
    """Operator v -> L A L v for a symmetric application L."""

    def __init__(
        self,
        base: SymmetricLinearOperator,
        apply_left: Callable[[torch.Tensor], torch.Tensor],
        *,
        name: str | None = None,
    ) -> None:
        self.base = base
        self.apply_left = apply_left
        self.dimension = base.dimension
        self.dtype = base.dtype
        self.device = base.device
        self.name = name or f"congruence({base.name})"

    def matvec(self, vector: torch.Tensor) -> torch.Tensor:
        self._validate_vector(vector)
        transformed = self.apply_left(vector)
        return self.apply_left(self.base.matvec(transformed))


def materialize(
    operator: SymmetricLinearOperator,
    *,
    max_dimension: int | None = None,
    symmetrize: bool = True,
) -> torch.Tensor:
    if max_dimension is not None and operator.dimension > max_dimension:
        raise ValueError(
            f"Refusing to materialize dimension {operator.dimension}; "
            f"limit is {max_dimension}"
        )
    eye = torch.eye(operator.dimension, dtype=operator.dtype, device=operator.device)
    columns = [operator.matvec(eye[:, index]) for index in range(operator.dimension)]
    matrix = torch.stack(columns, dim=1)
    if symmetrize:
        matrix = 0.5 * (matrix + matrix.T)
    return matrix
