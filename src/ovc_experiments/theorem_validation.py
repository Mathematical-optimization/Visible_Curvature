from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any

import torch


def normalized_hadamard(order: int, *, dtype: torch.dtype = torch.float64) -> torch.Tensor:
    if order <= 0 or order & (order - 1):
        raise ValueError('Hadamard order must be a positive power of two')
    H = torch.ones((1, 1), dtype=dtype)
    while H.shape[0] < order:
        H = torch.cat((torch.cat((H, H), dim=1), torch.cat((H, -H), dim=1)), dim=0)
    return H / math.sqrt(order)


def reciprocal_closed_spectrum(kappa: float, half_size: int, *, dtype: torch.dtype = torch.float64) -> torch.Tensor:
    if kappa <= 1 or half_size <= 0:
        raise ValueError('kappa must exceed one and half_size must be positive')
    first = torch.logspace(0, math.log10(kappa), half_size, dtype=dtype)
    return torch.cat((first, kappa / first.flip(0)))


@dataclass(frozen=True)
class FlatKronValidation:
    covariance_spectrum_error: float
    covariance_diagonal_error: float
    adam_scalar_relative_error: float
    aligned_condition: float
    reversed_condition: float
    aligned_theory: float
    reversed_theory: float
    product_relative_error: float
    passed: bool


def validate_flat_kron_pair(kappa: float = 8.0, r: float = 2.0, rho: float = 0.5, half_size: int = 2) -> FlatKronValidation:
    b = reciprocal_closed_spectrum(kappa, half_size)
    Q = normalized_hadamard(b.numel())
    B = Q @ torch.diag(b) @ Q.mT
    c_plus = Q @ torch.diag(b.pow(r)) @ Q.mT
    c_minus = Q @ torch.diag((kappa / b).pow(r)) @ Q.mT
    sigma_plus = torch.kron(c_plus, c_plus)
    sigma_minus = torch.kron(c_minus, c_minus)
    spec_error = float((torch.linalg.eigvalsh(sigma_plus) - torch.linalg.eigvalsh(sigma_minus)).abs().max().item())
    diag_error = float((torch.diagonal(sigma_plus) - torch.diagonal(sigma_minus)).abs().max().item())
    adam_diag = torch.diagonal(sigma_plus)
    adam_scalar_error = float((adam_diag - adam_diag.mean()).abs().max().item() / max(abs(float(adam_diag.mean().item())), 1e-300))
    H = torch.kron(B, B)
    def inv_quarter(C: torch.Tensor) -> torch.Tensor:
        values, vectors = torch.linalg.eigh(C + rho * torch.eye(C.shape[0], dtype=C.dtype))
        return (vectors * values.pow(-0.25).unsqueeze(0)) @ vectors.mT
    P_plus = torch.kron(inv_quarter(c_plus), inv_quarter(c_plus))
    P_minus = torch.kron(inv_quarter(c_minus), inv_quarter(c_minus))
    def condition(P: torch.Tensor) -> float:
        values, vectors = torch.linalg.eigh(P)
        root = (vectors * values.sqrt().unsqueeze(0)) @ vectors.mT
        eig = torch.linalg.eigvalsh(root @ H @ root)
        return float((eig.max() / eig.min()).item())
    k_plus = condition(P_plus)
    k_minus = condition(P_minus)
    k_h = kappa * kappa
    S = ((kappa**r + rho) / (1.0 + rho)) ** 0.25
    theory_plus = k_h / (S * S)
    theory_minus = k_h * (S * S)
    product_error = abs(k_plus * k_minus - k_h * k_h) / (k_h * k_h)
    tol = 1e-8
    passed = spec_error <= tol and diag_error <= tol and adam_scalar_error <= tol and abs(k_plus-theory_plus)/theory_plus <= tol and abs(k_minus-theory_minus)/theory_minus <= tol and product_error <= tol
    return FlatKronValidation(spec_error, diag_error, adam_scalar_error, k_plus, k_minus, theory_plus, theory_minus, product_error, passed)


@dataclass(frozen=True)
class WeightedChebyshevValidation:
    condition_number: float
    degree: int
    theoretical_barrier: float
    least_squares_minimum: float
    kkt_residual: float
    relative_error: float
    passed: bool


def validate_weighted_chebyshev(condition_number: float = 100.0, degree: int = 8) -> WeightedChebyshevValidation:
    if condition_number <= 1 or degree < 1:
        raise ValueError('condition_number must exceed one and degree must be positive')
    K = float(condition_number)
    T = int(degree)
    j = torch.arange(T + 1, dtype=torch.float64)
    t = torch.cos(math.pi * j / T)
    center = (K + 1.0) / 2.0
    radius = (K - 1.0) / 2.0
    nodes = center + radius * t
    delta = torch.ones(T + 1, dtype=torch.float64)
    delta[0] = delta[-1] = 0.5
    weights = delta / nodes
    weights = weights / weights.sum()
    # Residual polynomial p(0)=1. Free coefficients multiply x^1,...,x^T.
    scaled_nodes = nodes / K
    V = torch.stack([scaled_nodes.pow(k) for k in range(1, T + 1)], dim=1)
    W = torch.diag(weights)
    normal = V.mT @ W @ V
    rhs = -(V.mT @ weights)
    coefficients = torch.linalg.solve(normal, rhs)
    values = 1.0 + V @ coefficients
    minimum = float((weights * values.square()).sum().item())
    q = (math.sqrt(K) + 1.0) / (math.sqrt(K) - 1.0)
    barrier = 1.0 / math.cosh(T * math.log(q)) ** 2
    # At Chebyshev extrema, optimal values alternate and weighted moments vanish.
    kkt = float(torch.linalg.vector_norm(V.mT @ (weights * values)).item())
    rel = abs(minimum - barrier) / barrier
    passed = rel <= 1e-6 and kkt <= 1e-7
    return WeightedChebyshevValidation(K, T, barrier, minimum, kkt, rel, passed)
