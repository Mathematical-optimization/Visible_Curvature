from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import torch

from .blocks import MatrixLayout
from .moments import estimate_moments_from_gradients
from .spectral_power import SubspacePolicy, symmetric_matrix_power
from .spectrum_utils import factorwise_damping


class FrozenPreconditioner:
    layout: MatrixLayout
    dimension: int
    dtype: torch.dtype
    device: torch.device
    name: str

    def apply_power(self, vector: torch.Tensor, power: float) -> torch.Tensor:
        raise NotImplementedError

    def apply(self, vector: torch.Tensor) -> torch.Tensor:
        return self.apply_power(vector, 1.0)

    def apply_sqrt(self, vector: torch.Tensor) -> torch.Tensor:
        return self.apply_power(vector, 0.5)

    def _validate(self, vector: torch.Tensor) -> None:
        if vector.ndim != 1 or vector.numel() != self.dimension:
            raise ValueError(
                f"{self.name} expects shape ({self.dimension},), got {tuple(vector.shape)}"
            )


class IdentityPreconditioner(FrozenPreconditioner):
    def __init__(self, layout: MatrixLayout, *, dtype: torch.dtype, device: torch.device | str) -> None:
        self.layout = layout
        self.dimension = layout.numel
        self.dtype = dtype
        self.device = torch.device(device)
        self.name = "identity"

    def apply_power(self, vector: torch.Tensor, power: float) -> torch.Tensor:
        self._validate(vector)
        return vector


class AdamPreconditioner(FrozenPreconditioner):
    def __init__(
        self,
        statistic: torch.Tensor,
        *,
        layout: MatrixLayout,
        damping: float = 0.0,
        eigenvalue_floor: float = 0.0,
        relative_eigenvalue_floor: float = 1e-12,
        subspace_policy: SubspacePolicy = "strict_spd",
        name: str = "adam-form",
    ) -> None:
        if statistic.ndim != 1 or statistic.numel() != layout.numel:
            raise ValueError(
                f"Expected diagonal statistic of length {layout.numel}, got {tuple(statistic.shape)}"
            )
        if damping < 0:
            raise ValueError("damping must be nonnegative")
        self.statistic = statistic
        self.layout = layout
        self.dimension = layout.numel
        self.dtype = statistic.dtype
        self.device = statistic.device
        self.damping = float(damping)
        self.eigenvalue_floor = float(eigenvalue_floor)
        self.relative_eigenvalue_floor = float(relative_eigenvalue_floor)
        self.subspace_policy = subspace_policy
        self.name = name

    def diagonal_for_power(self, power: float) -> torch.Tensor:
        if power == 0.0:
            return torch.ones_like(self.statistic)
        base = self.statistic + self.damping
        exponent = -0.5 * power
        maximum = float(base.abs().max().item()) if base.numel() else 0.0
        eps = torch.finfo(base.dtype).eps
        numerical_floor = max(
            self.eigenvalue_floor,
            self.relative_eigenvalue_floor * maximum,
            64.0 * eps * base.numel() * max(maximum, torch.finfo(base.dtype).tiny),
        )
        if torch.any(base < -numerical_floor):
            raise ValueError("Adam statistic plus damping must be positive semidefinite")
        if self.subspace_policy not in {"strict_spd", "positive_active", "pseudoinverse"}:
            raise ValueError(f"unknown subspace policy: {self.subspace_policy}")
        if exponent < 0:
            if self.subspace_policy == "strict_spd" and torch.any(base <= 0):
                raise ValueError(
                    "negative powers require positive Adam statistics in strict_spd mode; "
                    "add damping or choose an explicit active-subspace policy"
                )
            powered = base.clamp_min(numerical_floor).pow(exponent)
            if self.subspace_policy in {"positive_active", "pseudoinverse"}:
                powered = torch.where(base > numerical_floor, powered, torch.zeros_like(powered))
            return powered
        return base.clamp_min(0.0).pow(exponent)

    def apply_power(self, vector: torch.Tensor, power: float) -> torch.Tensor:
        self._validate(vector)
        return self.diagonal_for_power(power) * vector


class ShampooPreconditioner(FrozenPreconditioner):
    def __init__(
        self,
        left_factor: torch.Tensor,
        right_factor: torch.Tensor,
        *,
        layout: MatrixLayout,
        alpha: float = 0.25,
        damping_left: float = 0.0,
        damping_right: float | None = None,
        eigenvalue_floor: float = 0.0,
        relative_eigenvalue_floor: float = 1e-12,
        subspace_policy: SubspacePolicy = "strict_spd",
        name: str = "shampoo-form",
    ) -> None:
        rows, cols = layout.matrix_shape
        if left_factor.shape != (rows, rows):
            raise ValueError(f"Expected left factor {(rows, rows)}, got {tuple(left_factor.shape)}")
        if right_factor.shape != (cols, cols):
            raise ValueError(f"Expected right factor {(cols, cols)}, got {tuple(right_factor.shape)}")
        if alpha < 0:
            raise ValueError("alpha must be nonnegative")
        if damping_left < 0 or (damping_right is not None and damping_right < 0):
            raise ValueError("damping must be nonnegative")
        self.left_factor = 0.5 * (left_factor + left_factor.T)
        self.right_factor = 0.5 * (right_factor + right_factor.T)
        self.layout = layout
        self.dimension = layout.numel
        self.dtype = left_factor.dtype
        self.device = left_factor.device
        self.alpha = float(alpha)
        self.damping_left = float(damping_left)
        self.damping_right = float(damping_left if damping_right is None else damping_right)
        self.eigenvalue_floor = float(eigenvalue_floor)
        self.relative_eigenvalue_floor = float(relative_eigenvalue_floor)
        self.subspace_policy = subspace_policy
        self.name = name
        self._cache: dict[tuple[str, float], torch.Tensor] = {}

    def _factor_power(self, side: str, power: float) -> torch.Tensor:
        key = (side, float(power))
        if key in self._cache:
            return self._cache[key]
        if side == "left":
            factor = self.left_factor
            damping = self.damping_left
        elif side == "right":
            factor = self.right_factor
            damping = self.damping_right
        else:
            raise ValueError(f"Unknown factor side: {side}")
        result = symmetric_matrix_power(
            factor,
            -self.alpha * power,
            damping=damping,
            absolute_eigenvalue_floor=self.eigenvalue_floor,
            relative_eigenvalue_floor=self.relative_eigenvalue_floor,
            subspace_policy=self.subspace_policy,
        )
        self._cache[key] = result
        return result

    def apply_matrix_power(self, matrix: torch.Tensor, power: float) -> torch.Tensor:
        if tuple(matrix.shape) != self.layout.matrix_shape:
            raise ValueError(
                f"Expected matrix shape {self.layout.matrix_shape}, got {tuple(matrix.shape)}"
            )
        if self.alpha == 0.0 or power == 0.0:
            return matrix
        left = self._factor_power("left", power)
        right = self._factor_power("right", power)
        return left @ matrix @ right

    def apply_power(self, vector: torch.Tensor, power: float) -> torch.Tensor:
        self._validate(vector)
        matrix = vector.reshape(self.layout.matrix_shape)
        return self.apply_matrix_power(matrix, power).reshape(-1)

    def factor_spectra(self) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.linalg.eigvalsh(self.left_factor), torch.linalg.eigvalsh(self.right_factor)


@dataclass
class _Tile:
    row_slice: slice
    col_slice: slice
    preconditioner: ShampooPreconditioner


class TiledShampooPreconditioner(FrozenPreconditioner):
    def __init__(self, layout: MatrixLayout, tiles: Iterable[_Tile], *, name: str = "tiled-shampoo") -> None:
        self.layout = layout
        self.dimension = layout.numel
        self.tiles = list(tiles)
        if not self.tiles:
            raise ValueError("At least one tile is required")
        self.dtype = self.tiles[0].preconditioner.dtype
        self.device = self.tiles[0].preconditioner.device
        self.name = name

    @classmethod
    def from_per_example_gradients(
        cls,
        gradients: torch.Tensor,
        *,
        layout: MatrixLayout,
        alpha: float,
        damping: float,
        damping_ratio: float | None = None,
        relative_eigenvalue_floor: float = 1e-12,
        tile_rows: int,
        tile_cols: int,
        centered: bool,
        accumulation_dtype: torch.dtype = torch.float64,
    ) -> "TiledShampooPreconditioner":
        if tile_rows < 1 or tile_cols < 1:
            raise ValueError("tile sizes must be positive")
        matrices = gradients.reshape(gradients.shape[0], *layout.matrix_shape)
        rows, cols = layout.matrix_shape
        tiles: list[_Tile] = []
        for row_start in range(0, rows, tile_rows):
            row_stop = min(rows, row_start + tile_rows)
            for col_start in range(0, cols, tile_cols):
                col_stop = min(cols, col_start + tile_cols)
                tile_gradients = matrices[:, row_start:row_stop, col_start:col_stop]
                tile_layout = MatrixLayout.from_shape((row_stop - row_start, col_stop - col_start))
                moments = estimate_moments_from_gradients(
                    tile_gradients,
                    tile_layout,
                    accumulation_dtype=accumulation_dtype,
                )
                left, right = moments.factors(centered=centered)
                if damping_ratio is None:
                    damping_left = float(damping)
                    damping_right = float(damping)
                else:
                    normalized = factorwise_damping(
                        left,
                        right,
                        normalized_ratio=float(damping_ratio),
                        relative_threshold=float(relative_eigenvalue_floor),
                    )
                    damping_left = normalized.left
                    damping_right = normalized.right
                preconditioner = ShampooPreconditioner(
                    left,
                    right,
                    layout=tile_layout,
                    alpha=alpha,
                    damping_left=damping_left,
                    damping_right=damping_right,
                    relative_eigenvalue_floor=relative_eigenvalue_floor,
                    name=f"tile[{row_start}:{row_stop},{col_start}:{col_stop}]",
                )
                tiles.append(
                    _Tile(
                        row_slice=slice(row_start, row_stop),
                        col_slice=slice(col_start, col_stop),
                        preconditioner=preconditioner,
                    )
                )
        return cls(layout, tiles)

    def apply_power(self, vector: torch.Tensor, power: float) -> torch.Tensor:
        self._validate(vector)
        matrix = vector.reshape(self.layout.matrix_shape)
        output = torch.zeros_like(matrix)
        for tile in self.tiles:
            block = matrix[tile.row_slice, tile.col_slice]
            transformed = tile.preconditioner.apply_matrix_power(block, power)
            output[tile.row_slice, tile.col_slice] = transformed
        return output.reshape(-1)


class ScaledPreconditioner(FrozenPreconditioner):
    def __init__(self, base: FrozenPreconditioner, scale: float, *, name: str | None = None) -> None:
        if scale <= 0 or not math.isfinite(scale):
            raise ValueError("scale must be finite and positive")
        self.base = base
        self.scale = float(scale)
        self.layout = base.layout
        self.dimension = base.dimension
        self.dtype = base.dtype
        self.device = base.device
        self.name = name or f"{scale:g}*{base.name}"

    def apply_power(self, vector: torch.Tensor, power: float) -> torch.Tensor:
        self._validate(vector)
        return (self.scale ** power) * self.base.apply_power(vector, power)


class ExplicitDiagonalPreconditioner(FrozenPreconditioner):
    """Frozen positive diagonal operator specified by its applied weights."""

    def __init__(
        self,
        weights: torch.Tensor,
        *,
        layout: MatrixLayout,
        eigenvalue_floor: float = 1e-30,
        name: str = "explicit-diagonal",
    ) -> None:
        if weights.ndim != 1 or weights.numel() != layout.numel:
            raise ValueError(
                f"Expected {layout.numel} diagonal weights, got {tuple(weights.shape)}"
            )
        if torch.any(weights < 0):
            raise ValueError("Diagonal preconditioner weights must be nonnegative")
        self.weights = weights
        self.layout = layout
        self.dimension = layout.numel
        self.dtype = weights.dtype
        self.device = weights.device
        self.eigenvalue_floor = float(eigenvalue_floor)
        self.name = name

    def apply_power(self, vector: torch.Tensor, power: float) -> torch.Tensor:
        self._validate(vector)
        if power == 0.0:
            return vector
        powered = torch.where(
            self.weights > self.eigenvalue_floor,
            self.weights.clamp_min(self.eigenvalue_floor).pow(power),
            torch.zeros_like(self.weights),
        )
        return powered * vector


class ExplicitFactorPreconditioner(FrozenPreconditioner):
    """Frozen two-sided operator specified by its already-computed factors."""

    def __init__(
        self,
        left_operator: torch.Tensor,
        right_operator: torch.Tensor,
        *,
        layout: MatrixLayout,
        eigenvalue_floor: float = 1e-30,
        subspace_policy: SubspacePolicy = "strict_spd",
        name: str = "explicit-factor",
    ) -> None:
        rows, cols = layout.matrix_shape
        if left_operator.shape != (rows, rows):
            raise ValueError(
                f"Expected left operator {(rows, rows)}, got {tuple(left_operator.shape)}"
            )
        if right_operator.shape != (cols, cols):
            raise ValueError(
                f"Expected right operator {(cols, cols)}, got {tuple(right_operator.shape)}"
            )
        self.left_operator = 0.5 * (left_operator + left_operator.T)
        self.right_operator = 0.5 * (right_operator + right_operator.T)
        self.layout = layout
        self.dimension = layout.numel
        self.dtype = left_operator.dtype
        self.device = left_operator.device
        self.eigenvalue_floor = float(eigenvalue_floor)
        self.subspace_policy = subspace_policy
        self.name = name
        self._cache: dict[tuple[str, float], torch.Tensor] = {}

    def _power(self, side: str, power: float) -> torch.Tensor:
        key = (side, float(power))
        if key in self._cache:
            return self._cache[key]
        matrix = self.left_operator if side == "left" else self.right_operator
        result = symmetric_matrix_power(
            matrix,
            power,
            absolute_eigenvalue_floor=self.eigenvalue_floor,
            subspace_policy=self.subspace_policy,
        )
        self._cache[key] = result
        return result

    def apply_power(self, vector: torch.Tensor, power: float) -> torch.Tensor:
        self._validate(vector)
        if power == 0.0:
            return vector
        matrix = vector.reshape(self.layout.matrix_shape)
        return (self._power("left", power) @ matrix @ self._power("right", power)).reshape(-1)

