import math

import torch

from visible_curvature.curvature import LinearMatrixOperator
from visible_curvature.preconditioners import AdamFormPreconditioner, ShampooFormPreconditioner


def _dense_effective(op, preconditioner):
    basis = torch.eye(op.dim, dtype=op.dtype)
    columns = []
    for index in range(op.dim):
        vector = basis[:, index].reshape(op.shape)
        half = preconditioner.apply_half(vector)
        columns.append(preconditioner.apply_half(op.matvec_matrix(half)).reshape(-1))
    matrix = torch.stack(columns, dim=1)
    return 0.5 * (matrix + matrix.T)


def _condition(matrix):
    values = torch.linalg.eigvalsh(matrix)
    return float(values[-1] / values[0])


def _hadamard(n: int) -> torch.Tensor:
    matrix = torch.ones(1, 1, dtype=torch.float64)
    while matrix.shape[0] < n:
        matrix = torch.cat(
            [torch.cat([matrix, matrix], dim=1), torch.cat([matrix, -matrix], dim=1)],
            dim=0,
        )
    return matrix / math.sqrt(n)


def test_theorem3_adam_aligned_and_reversed_condition_numbers():
    kappa = 10.0
    covariance = torch.tensor([[kappa**2], [1.0]], dtype=torch.float64)
    preconditioner = AdamFormPreconditioner(covariance, damping=0.0)
    aligned_h = torch.tensor([kappa, 1.0], dtype=torch.float64)
    reversed_h = torch.tensor([1.0, kappa], dtype=torch.float64)

    def make_operator(diagonal):
        return LinearMatrixOperator(
            (2, 1),
            lambda vector: diagonal.reshape(2, 1) * vector,
            device=torch.device("cpu"),
            dtype=torch.float64,
        )

    aligned = _condition(_dense_effective(make_operator(aligned_h), preconditioner))
    reversed_value = _condition(_dense_effective(make_operator(reversed_h), preconditioner))
    assert math.isclose(aligned, 1.0, rel_tol=1e-12)
    assert math.isclose(reversed_value, kappa**2, rel_tol=1e-12)


def test_flat_kronecker_pair_reproduces_adam_shampoo_order_reversal():
    n = 4
    kappa = 16.0
    r = 2.0
    Q = _hadamard(n)
    eigenvalues = torch.tensor([1.0, kappa, 1.0, kappa], dtype=torch.float64)
    B = Q @ torch.diag(eigenvalues) @ Q.T
    H = LinearMatrixOperator(
        (n, n),
        lambda vector: B @ vector @ B,
        device=torch.device("cpu"),
        dtype=torch.float64,
    )
    C_plus = torch.linalg.matrix_power(B, int(r))
    C_minus = (kappa**r) * torch.linalg.inv(C_plus)
    sigma_diag_plus = torch.outer(torch.diag(C_plus), torch.diag(C_plus))
    sigma_diag_minus = torch.outer(torch.diag(C_minus), torch.diag(C_minus))
    assert torch.allclose(sigma_diag_plus, sigma_diag_minus, atol=1e-10, rtol=1e-10)

    adam_plus = AdamFormPreconditioner(sigma_diag_plus, damping=0.0)
    adam_minus = AdamFormPreconditioner(sigma_diag_minus, damping=0.0)
    shampoo_plus = ShampooFormPreconditioner(C_plus, C_plus, damping=0.0)
    shampoo_minus = ShampooFormPreconditioner(C_minus, C_minus, damping=0.0)

    K_ad_plus = _condition(_dense_effective(H, adam_plus))
    K_ad_minus = _condition(_dense_effective(H, adam_minus))
    K_sh_plus = _condition(_dense_effective(H, shampoo_plus))
    K_sh_minus = _condition(_dense_effective(H, shampoo_minus))

    S = kappa ** (r / 4.0)
    assert math.isclose(K_ad_plus, kappa**2, rel_tol=1e-9)
    assert math.isclose(K_ad_minus, kappa**2, rel_tol=1e-9)
    assert math.isclose(K_sh_plus, kappa**2 / S**2, rel_tol=1e-9)
    assert math.isclose(K_sh_minus, kappa**2 * S**2, rel_tol=1e-9)
    assert K_sh_plus < K_ad_plus < K_sh_minus


def test_coordinate_power_coupling_has_predicted_condition_exponent():
    kappa = 32.0
    r = 1.25
    h = torch.tensor([[1.0], [kappa**2]], dtype=torch.float64)
    covariance = h.pow(r)
    preconditioner = AdamFormPreconditioner(covariance, damping=0.0)
    operator = LinearMatrixOperator(
        (2, 1),
        lambda vector: h * vector,
        device=torch.device("cpu"),
        dtype=torch.float64,
    )
    value = _condition(_dense_effective(operator, preconditioner))
    expected = kappa ** (2.0 - r)
    assert math.isclose(value, expected, rel_tol=1e-10)
