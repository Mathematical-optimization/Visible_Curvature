from __future__ import annotations
from .safe_operators import DiagonalOperator as _OVCDiagonalOperator
from dataclasses import dataclass
import math
import torch
from scipy.stats import spearmanr
from .blocks import MatrixLayout
from .operators import DenseOperator, FunctionOperator, SymmetricLinearOperator, materialize
from .preconditioners import FrozenPreconditioner
from .spectral import lanczos

@dataclass
class ResponseEstimate:
    slope: float
    intercept: float
    spearman: float
    valid_directions: int
    curvatures: torch.Tensor
    statistics: torch.Tensor

    def to_dict(self) -> dict[str, object]:
        return {'slope': self.slope, 'intercept': self.intercept, 'spearman': self.spearman, 'valid_directions': self.valid_directions, 'curvatures': self.curvatures.detach().cpu().tolist(), 'statistics': self.statistics.detach().cpu().tolist()}

@dataclass
class OverlapEstimate:
    squared_overlap: torch.Tensor
    affinity: float
    rank: int
    method: str

def project_operator(operator: SymmetricLinearOperator, basis: torch.Tensor) -> torch.Tensor:
    if basis.ndim != 2 or basis.shape[0] != operator.dimension:
        raise ValueError(f'Basis must have shape ({operator.dimension}, k), got {tuple(basis.shape)}')
    images = torch.stack([operator.matvec(basis[:, index]) for index in range(basis.shape[1])], dim=1)
    projected = basis.T @ images
    return 0.5 * (projected + projected.T)

def matched_response(curvature: SymmetricLinearOperator, statistic: SymmetricLinearOperator, vectors: torch.Tensor, *, positive_threshold: float=1e-12) -> ResponseEstimate:
    if curvature.dimension != statistic.dimension:
        raise ValueError('Curvature and statistic dimensions differ')
    if vectors.ndim != 2 or vectors.shape[0] != curvature.dimension:
        raise ValueError('vectors must contain directions as columns')
    h_values: list[torch.Tensor] = []
    q_values: list[torch.Tensor] = []
    for index in range(vectors.shape[1]):
        direction = vectors[:, index]
        norm = torch.linalg.vector_norm(direction)
        if norm <= positive_threshold:
            continue
        direction = direction / norm
        h = torch.dot(direction, curvature.matvec(direction))
        q = torch.dot(direction, statistic.matvec(direction))
        if h > positive_threshold and q > positive_threshold:
            h_values.append(h)
            q_values.append(q)
    if len(h_values) < 2:
        raise ValueError('At least two positive matched directions are required')
    h_tensor = torch.stack(h_values)
    q_tensor = torch.stack(q_values)
    x = torch.log(h_tensor)
    y = torch.log(q_tensor)
    design = torch.stack([x, torch.ones_like(x)], dim=1)
    coefficients = torch.linalg.lstsq(design, y).solution
    slope = float(coefficients[0].item())
    intercept = float(coefficients[1].item())
    correlation = spearmanr(h_tensor.detach().cpu().numpy(), q_tensor.detach().cpu().numpy()).statistic
    spearman = float(correlation) if math.isfinite(float(correlation)) else 0.0
    return ResponseEstimate(slope=slope, intercept=intercept, spearman=spearman, valid_directions=len(h_values), curvatures=h_tensor, statistics=q_tensor)

def projected_commutator(first: SymmetricLinearOperator, second: SymmetricLinearOperator, basis: torch.Tensor) -> float:
    first_projected = project_operator(first, basis)
    second_projected = project_operator(second, basis)
    commutator = first_projected @ second_projected - second_projected @ first_projected
    denominator = torch.linalg.matrix_norm(first_projected) * torch.linalg.matrix_norm(second_projected)
    if denominator == 0:
        return 0.0
    return float((torch.linalg.matrix_norm(commutator) / denominator).item())

def _leading_eigenspace(operator: SymmetricLinearOperator, rank: int, *, exact_max_dim: int, seed: int) -> tuple[torch.Tensor, str]:
    if rank < 1 or rank > operator.dimension:
        raise ValueError('rank must be between 1 and operator dimension')
    if operator.dimension <= exact_max_dim:
        eigenvalues, eigenvectors = torch.linalg.eigh(materialize(operator))
        del eigenvalues
        return (eigenvectors[:, -rank:], 'exact')
    result = lanczos(operator, steps=min(operator.dimension, max(4 * rank, 32)), seed=seed)
    return (result.ritz_vectors[:, -rank:], 'lanczos')

def eigenspace_overlap(first: SymmetricLinearOperator, second: SymmetricLinearOperator, *, rank: int, exact_max_dim: int=512, seed: int=0) -> OverlapEstimate:
    if first.dimension != second.dimension:
        raise ValueError('Operator dimensions differ')
    first_space, first_method = _leading_eigenspace(first, rank, exact_max_dim=exact_max_dim, seed=seed)
    second_space, second_method = _leading_eigenspace(second, rank, exact_max_dim=exact_max_dim, seed=seed + 17)
    squared = torch.abs(first_space.T @ second_space).square()
    affinity = float((squared.sum() / rank).item())
    method = 'exact' if first_method == second_method == 'exact' else 'lanczos'
    return OverlapEstimate(squared_overlap=squared, affinity=affinity, rank=rank, method=method)

def preconditioner_matrix(preconditioner: FrozenPreconditioner) -> torch.Tensor:
    eye = torch.eye(preconditioner.dimension, dtype=preconditioner.dtype, device=preconditioner.device)
    columns = [preconditioner.apply(eye[:, index]) for index in range(preconditioner.dimension)]
    matrix = torch.stack(columns, dim=1)
    return 0.5 * (matrix + matrix.T)

def nci(preconditioner: FrozenPreconditioner, hessian: torch.Tensor, covariance: torch.Tensor) -> float:
    if hessian.shape != covariance.shape or hessian.shape != (preconditioner.dimension, preconditioner.dimension):
        raise ValueError('NCI inputs must have matching square dimensions')
    p_matrix = preconditioner_matrix(preconditioner)
    numerator = torch.trace(p_matrix @ hessian @ p_matrix @ covariance)
    denominator = torch.linalg.matrix_norm(p_matrix).square()
    if denominator == 0:
        raise ValueError('Preconditioner Frobenius norm is zero')
    return float((numerator / denominator).item())

def curvature_factor_proxies_from_ritz(eigenvalues: torch.Tensor, eigenvectors: torch.Tensor, layout: MatrixLayout, *, positive_only: bool=False) -> tuple[torch.Tensor, torch.Tensor]:
    if eigenvectors.ndim != 2 or eigenvectors.shape[0] != layout.numel:
        raise ValueError('Ritz vectors must have shape (block_dimension, rank)')
    if eigenvectors.shape[1] != eigenvalues.numel():
        raise ValueError('Ritz value/vector counts differ')
    rows, cols = layout.matrix_shape
    left = torch.zeros((rows, rows), dtype=eigenvectors.dtype, device=eigenvectors.device)
    right = torch.zeros((cols, cols), dtype=eigenvectors.dtype, device=eigenvectors.device)
    for value, vector in zip(eigenvalues, eigenvectors.T):
        if positive_only and value <= 0:
            continue
        matrix = vector.reshape(layout.matrix_shape)
        left = left + value * (matrix @ matrix.T)
        right = right + value * (matrix.T @ matrix)
    return (0.5 * (left + left.T), 0.5 * (right + right.T))

def adam_statistic_operator(diagonal: torch.Tensor, *, damping: float=0.0, name: str='adam-retained-statistic') -> SymmetricLinearOperator:
    matrix = diagonal + damping
    return _OVCDiagonalOperator(matrix)

def shampoo_statistic_operator(left_factor: torch.Tensor, right_factor: torch.Tensor, layout: MatrixLayout, *, damping_left: float=0.0, damping_right: float | None=None, name: str='shampoo-retained-statistic') -> SymmetricLinearOperator:
    damping_right = damping_left if damping_right is None else damping_right
    left = left_factor + damping_left * torch.eye(left_factor.shape[0], dtype=left_factor.dtype, device=left_factor.device)
    right = right_factor + damping_right * torch.eye(right_factor.shape[0], dtype=right_factor.dtype, device=right_factor.device)

    def matvec(vector: torch.Tensor) -> torch.Tensor:
        matrix = vector.reshape(layout.matrix_shape)
        return (left @ matrix @ right).reshape(-1)
    return FunctionOperator(dimension=layout.numel, matvec_fn=matvec, dtype=left.dtype, device=left.device, name=name)
import torch
