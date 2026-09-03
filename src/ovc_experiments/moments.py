from __future__ import annotations

from dataclasses import dataclass

import torch

from .blocks import MatrixLayout


@dataclass
class MomentEstimate:
    layout: MatrixLayout
    num_examples: int
    mean_matrix: torch.Tensor
    centered_left: torch.Tensor
    centered_right: torch.Tensor
    centered_diag: torch.Tensor
    uncentered_left: torch.Tensor
    uncentered_right: torch.Tensor
    uncentered_diag: torch.Tensor

    def factors(self, *, centered: bool) -> tuple[torch.Tensor, torch.Tensor]:
        if centered:
            return self.centered_left, self.centered_right
        return self.uncentered_left, self.uncentered_right

    def diagonal(self, *, centered: bool) -> torch.Tensor:
        return self.centered_diag if centered else self.uncentered_diag

    def to(self, *, device: torch.device | str | None = None, dtype: torch.dtype | None = None) -> "MomentEstimate":
        kwargs = {"device": device, "dtype": dtype}
        return MomentEstimate(
            layout=self.layout,
            num_examples=self.num_examples,
            mean_matrix=self.mean_matrix.to(**kwargs),
            centered_left=self.centered_left.to(**kwargs),
            centered_right=self.centered_right.to(**kwargs),
            centered_diag=self.centered_diag.to(**kwargs),
            uncentered_left=self.uncentered_left.to(**kwargs),
            uncentered_right=self.uncentered_right.to(**kwargs),
            uncentered_diag=self.uncentered_diag.to(**kwargs),
        )


def _as_matrix_batch(gradients: torch.Tensor, layout: MatrixLayout) -> torch.Tensor:
    if gradients.ndim < 2:
        raise ValueError("Per-example gradients must include a sample dimension")
    expected_tail = layout.original_shape
    if tuple(gradients.shape[1:]) == expected_tail:
        return gradients.reshape(gradients.shape[0], *layout.matrix_shape)
    if tuple(gradients.shape[1:]) == layout.matrix_shape:
        return gradients
    raise ValueError(
        f"Expected gradient tail {expected_tail} or {layout.matrix_shape}, "
        f"got {tuple(gradients.shape[1:])}"
    )


def estimate_moments_from_gradients(
    gradients: torch.Tensor,
    layout: MatrixLayout,
    *,
    accumulation_dtype: torch.dtype = torch.float64,
) -> MomentEstimate:
    matrices = _as_matrix_batch(gradients, layout).to(dtype=accumulation_dtype)
    if matrices.shape[0] < 1:
        raise ValueError("At least one per-example gradient is required")
    mean_matrix = matrices.mean(dim=0)
    centered = matrices - mean_matrix

    uncentered_left = torch.einsum("nrc,nsc->rs", matrices, matrices) / matrices.shape[0]
    uncentered_right = torch.einsum("nrc,nrd->cd", matrices, matrices) / matrices.shape[0]
    centered_left = torch.einsum("nrc,nsc->rs", centered, centered) / matrices.shape[0]
    centered_right = torch.einsum("nrc,nrd->cd", centered, centered) / matrices.shape[0]

    flat = matrices.reshape(matrices.shape[0], -1)
    flat_centered = centered.reshape(centered.shape[0], -1)
    uncentered_diag = (flat * flat).mean(dim=0)
    centered_diag = (flat_centered * flat_centered).mean(dim=0)

    return MomentEstimate(
        layout=layout,
        num_examples=int(matrices.shape[0]),
        mean_matrix=mean_matrix,
        centered_left=0.5 * (centered_left + centered_left.T),
        centered_right=0.5 * (centered_right + centered_right.T),
        centered_diag=centered_diag,
        uncentered_left=0.5 * (uncentered_left + uncentered_left.T),
        uncentered_right=0.5 * (uncentered_right + uncentered_right.T),
        uncentered_diag=uncentered_diag,
    )
