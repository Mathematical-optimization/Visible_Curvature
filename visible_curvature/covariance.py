from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence

import numpy as np
import torch


@dataclass
class CovarianceState:
    count: int
    mean: torch.Tensor
    m2_diag: torch.Tensor
    m2_left: torch.Tensor
    m2_right: torch.Tensor

    @classmethod
    def zeros(cls, shape: tuple[int, int], device: torch.device | str = "cpu", dtype: torch.dtype = torch.float64) -> "CovarianceState":
        m, n = shape
        return cls(
            count=0,
            mean=torch.zeros(m, n, device=device, dtype=dtype),
            m2_diag=torch.zeros(m, n, device=device, dtype=dtype),
            m2_left=torch.zeros(m, m, device=device, dtype=dtype),
            m2_right=torch.zeros(n, n, device=device, dtype=dtype),
        )

    def clone(self) -> "CovarianceState":
        return CovarianceState(
            self.count,
            self.mean.clone(),
            self.m2_diag.clone(),
            self.m2_left.clone(),
            self.m2_right.clone(),
        )

    def update(self, x: torch.Tensor) -> None:
        x = x.to(device=self.mean.device, dtype=self.mean.dtype)
        n1 = self.count
        n2 = n1 + 1
        if n1 == 0:
            self.mean.copy_(x)
            self.count = 1
            return
        delta = x - self.mean
        self.mean.add_(delta / n2)
        delta2 = x - self.mean
        self.m2_diag.add_(delta * delta2)
        self.m2_left.add_(delta @ delta2.T)
        self.m2_right.add_(delta.T @ delta2)
        self.count = n2

    def merge(self, other: "CovarianceState") -> "CovarianceState":
        if other.count == 0:
            return self.clone()
        if self.count == 0:
            return other.clone()
        a = self
        b = other
        n = a.count + b.count
        delta = b.mean - a.mean
        weight = (a.count * b.count) / n
        out = CovarianceState.zeros(tuple(a.mean.shape), device=a.mean.device, dtype=a.mean.dtype)
        out.count = n
        out.mean = a.mean + delta * (b.count / n)
        out.m2_diag = a.m2_diag + b.m2_diag + delta.square() * weight
        out.m2_left = a.m2_left + b.m2_left + (delta @ delta.T) * weight
        out.m2_right = a.m2_right + b.m2_right + (delta.T @ delta) * weight
        return out

    def finalize(self, ddof: int = 0, dtype: torch.dtype = torch.float32) -> "CovarianceEstimate":
        denom = self.count - ddof
        if denom <= 0:
            raise ValueError(f"Need count > ddof; count={self.count}, ddof={ddof}")
        diag = (self.m2_diag / denom).to(dtype)
        left = (self.m2_left / denom).to(dtype)
        right = (self.m2_right / denom).to(dtype)
        left = 0.5 * (left + left.T)
        right = 0.5 * (right + right.T)
        return CovarianceEstimate(
            count=self.count,
            mean=self.mean.to(dtype),
            diag=diag,
            left=left,
            right=right,
        )


@dataclass
class CovarianceEstimate:
    count: int
    mean: torch.Tensor
    diag: torch.Tensor
    left: torch.Tensor
    right: torch.Tensor

    def to(self, device: torch.device, dtype: torch.dtype = torch.float32) -> "CovarianceEstimate":
        return CovarianceEstimate(
            count=self.count,
            mean=self.mean.to(device=device, dtype=dtype),
            diag=self.diag.to(device=device, dtype=dtype),
            left=self.left.to(device=device, dtype=dtype),
            right=self.right.to(device=device, dtype=dtype),
        )

    def uncentered(self) -> "CovarianceEstimate":
        """Return the second moment including the mean-gradient outer product."""
        return CovarianceEstimate(
            count=self.count,
            mean=self.mean.clone(),
            diag=self.diag + self.mean.square(),
            left=self.left + self.mean @ self.mean.T,
            right=self.right + self.mean.T @ self.mean,
        )



def merge_states(states: Sequence[CovarianceState]) -> CovarianceState:
    if not states:
        raise ValueError("No states to merge")
    out = CovarianceState.zeros(tuple(states[0].mean.shape), device=states[0].mean.device, dtype=states[0].mean.dtype)
    for s in states:
        out = out.merge(s)
    return out


def bootstrap_state(groups: Sequence[CovarianceState], rng: np.random.Generator) -> CovarianceState:
    if not groups:
        raise ValueError("No covariance groups for bootstrap")
    idx = rng.integers(0, len(groups), size=len(groups))
    return merge_states([groups[int(i)] for i in idx])
