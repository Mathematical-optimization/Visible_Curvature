from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Iterator, Optional

import torch

Tensor = torch.Tensor


@dataclass(frozen=True)
class MatrixMomentStatistics:
    count: int
    rows: int
    cols: int
    mean: Tensor
    adam_diagonal_centered: Tensor
    adam_diagonal_uncentered: Tensor
    left_centered: Tensor
    right_centered: Tensor
    left_uncentered: Tensor
    right_uncentered: Tensor

    @property
    def dimension(self) -> int:
        return self.rows * self.cols

    def effective_rank(self, matrix: Tensor, *, relative_threshold: float = 1e-10) -> float:
        eig = torch.linalg.eigvalsh((matrix + matrix.mT) * 0.5).clamp_min(0)
        top = float(eig.max().item()) if eig.numel() else 0.0
        if top <= 0:
            return 0.0
        active = eig[eig > relative_threshold * top]
        if active.numel() == 0:
            return 0.0
        total = active.sum()
        return float((total * total / active.square().sum()).item())


class StreamingMatrixMoments:
    """Online matrix-gradient moments using a mergeable Welford update.

    Memory is O(rc + r^2 + c^2), independent of the number of examples.
    No per-example gradient is retained unless the caller does so separately.
    """

    def __init__(
        self,
        rows: int,
        cols: int,
        *,
        dtype: torch.dtype = torch.float64,
        device: torch.device | str = 'cpu',
    ) -> None:
        if rows <= 0 or cols <= 0:
            raise ValueError('rows and cols must be positive')
        self.rows = int(rows)
        self.cols = int(cols)
        self.dtype = dtype
        self.device = torch.device(device)
        self.count = 0
        self.mean = torch.zeros((rows, cols), dtype=dtype, device=self.device)
        self.coordinate_m2 = torch.zeros((rows, cols), dtype=dtype, device=self.device)
        self.left_uncentered_sum = torch.zeros((rows, rows), dtype=dtype, device=self.device)
        self.right_uncentered_sum = torch.zeros((cols, cols), dtype=dtype, device=self.device)

    def update(self, gradient: Tensor) -> None:
        x = gradient.detach().to(device=self.device, dtype=self.dtype).reshape(self.rows, self.cols)
        if not torch.isfinite(x).all():
            raise ValueError('gradient contains non-finite values')
        self.count += 1
        delta = x - self.mean
        self.mean.add_(delta / self.count)
        delta2 = x - self.mean
        self.coordinate_m2.add_(delta * delta2)
        self.left_uncentered_sum.add_(x @ x.mT)
        self.right_uncentered_sum.add_(x.mT @ x)

    def update_many(self, gradients: Iterable[Tensor]) -> None:
        for gradient in gradients:
            self.update(gradient)

    def merge(self, other: 'StreamingMatrixMoments') -> None:
        if (self.rows, self.cols, self.dtype, self.device) != (
            other.rows, other.cols, other.dtype, other.device
        ):
            raise ValueError('incompatible accumulators')
        if other.count == 0:
            return
        if self.count == 0:
            self.count = other.count
            self.mean.copy_(other.mean)
            self.coordinate_m2.copy_(other.coordinate_m2)
            self.left_uncentered_sum.copy_(other.left_uncentered_sum)
            self.right_uncentered_sum.copy_(other.right_uncentered_sum)
            return
        n_a, n_b = self.count, other.count
        n = n_a + n_b
        delta = other.mean - self.mean
        self.coordinate_m2.add_(other.coordinate_m2)
        self.coordinate_m2.add_(delta.square() * (n_a * n_b / n))
        self.mean.add_(delta * (n_b / n))
        self.left_uncentered_sum.add_(other.left_uncentered_sum)
        self.right_uncentered_sum.add_(other.right_uncentered_sum)
        self.count = n

    def finalize(self, *, population: bool = True) -> MatrixMomentStatistics:
        if self.count == 0:
            raise ValueError('cannot finalize an empty accumulator')
        divisor = self.count if population else self.count - 1
        if divisor <= 0:
            raise ValueError('at least two examples are required for sample moments')
        mean = self.mean.clone()
        adam_centered = (self.coordinate_m2 / divisor).reshape(-1)
        adam_uncentered = (self.coordinate_m2 / self.count + mean.square()).reshape(-1)
        left_uncentered = self.left_uncentered_sum / self.count
        right_uncentered = self.right_uncentered_sum / self.count
        left_centered_population = left_uncentered - mean @ mean.mT
        right_centered_population = right_uncentered - mean.mT @ mean
        if population:
            left_centered = left_centered_population
            right_centered = right_centered_population
        else:
            correction = self.count / (self.count - 1)
            left_centered = left_centered_population * correction
            right_centered = right_centered_population * correction
        # Symmetrize to suppress round-off asymmetry.
        left_centered = (left_centered + left_centered.mT) * 0.5
        right_centered = (right_centered + right_centered.mT) * 0.5
        return MatrixMomentStatistics(
            count=self.count,
            rows=self.rows,
            cols=self.cols,
            mean=mean,
            adam_diagonal_centered=adam_centered,
            adam_diagonal_uncentered=adam_uncentered,
            left_centered=left_centered,
            right_centered=right_centered,
            left_uncentered=(left_uncentered + left_uncentered.mT) * 0.5,
            right_uncentered=(right_uncentered + right_uncentered.mT) * 0.5,
        )


def accumulate_matrix_moments(
    gradients: Iterable[Tensor],
    rows: int,
    cols: int,
    *,
    population: bool = True,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str = 'cpu',
) -> MatrixMomentStatistics:
    accumulator = StreamingMatrixMoments(rows, cols, dtype=dtype, device=device)
    accumulator.update_many(gradients)
    return accumulator.finalize(population=population)
