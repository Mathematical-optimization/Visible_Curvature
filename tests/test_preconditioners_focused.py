import math

import pytest
import torch

from visible_curvature.preconditioners import AdamFormPreconditioner, ShampooFormPreconditioner


def _expected_action(left_diag, right_diag, exponent, matrix):
    return left_diag.pow(-exponent).unsqueeze(1) * matrix * right_diag.pow(-exponent).unsqueeze(0)


def test_shampoo_factor_exponent_controls_full_and_half_actions():
    left_diag = torch.tensor([1.0, 16.0], dtype=torch.float64)
    right_diag = torch.tensor([4.0, 25.0, 36.0], dtype=torch.float64)
    left = torch.diag(left_diag)
    right = torch.diag(right_diag)
    matrix = torch.arange(1.0, 7.0, dtype=torch.float64).reshape(2, 3)

    for alpha in (0.25, 0.5):
        preconditioner = ShampooFormPreconditioner(
            left,
            right,
            damping=0.0,
            factor_exponent=alpha,
        )
        assert preconditioner.factor_exponent == alpha
        assert torch.allclose(
            preconditioner.apply(matrix),
            _expected_action(left_diag, right_diag, alpha, matrix),
            atol=1e-12,
            rtol=1e-12,
        )
        assert torch.allclose(
            preconditioner.apply_half(matrix),
            _expected_action(left_diag, right_diag, alpha / 2.0, matrix),
            atol=1e-12,
            rtol=1e-12,
        )


def test_shampoo_rejects_exponents_outside_the_paper_control_range():
    factor = torch.eye(2, dtype=torch.float64)
    with pytest.raises(ValueError, match="factor_exponent"):
        ShampooFormPreconditioner(factor, factor, damping=0.0, factor_exponent=0.0)
    with pytest.raises(ValueError, match="factor_exponent"):
        ShampooFormPreconditioner(factor, factor, damping=0.0, factor_exponent=0.75)


def test_factor_root_metadata_reports_adaptive_floor_and_floored_fraction():
    left = torch.diag(torch.tensor([0.0, 1.0], dtype=torch.float32))
    right = torch.diag(torch.tensor([1.0, 4.0], dtype=torch.float32))
    preconditioner = ShampooFormPreconditioner(
        left,
        right,
        damping=0.0,
        factor_exponent=0.25,
        eig_floor=0.0,
        relative_eig_floor=64.0,
    )

    assert preconditioner.left_effective_eig_floor > 0.0
    assert math.isclose(preconditioner.left_floored_fraction, 0.5)
    assert math.isclose(preconditioner.right_floored_fraction, 0.0)
    assert torch.isfinite(preconditioner.apply(torch.ones(2, 2))).all()


def test_adam_uses_scale_relative_floor_instead_of_dtype_tiny():
    diag = torch.tensor([[0.0, 1.0]], dtype=torch.float32)
    preconditioner = AdamFormPreconditioner(diag, damping=0.0, relative_floor=64.0)

    assert preconditioner.effective_floor > torch.finfo(torch.float32).tiny
    assert math.isclose(preconditioner.floored_fraction, 0.5)
    assert math.isfinite(preconditioner.max_entry)
    assert torch.isfinite(preconditioner.apply(torch.ones_like(diag))).all()


def test_shampoo_can_reuse_precomputed_factor_spectra():
    from visible_curvature.preconditioners import SymmetricSpectrum

    left = torch.tensor([[2.0, 0.5], [0.5, 3.0]], dtype=torch.float32)
    right = torch.tensor([[4.0, 0.25], [0.25, 5.0]], dtype=torch.float32)
    matrix = torch.arange(1.0, 5.0, dtype=torch.float32).reshape(2, 2)
    direct = ShampooFormPreconditioner(left, right, damping=0.1, factor_exponent=0.5)
    cached = ShampooFormPreconditioner(
        left,
        right,
        damping=0.1,
        factor_exponent=0.5,
        left_spectrum=SymmetricSpectrum.from_matrix(left),
        right_spectrum=SymmetricSpectrum.from_matrix(right),
    )
    assert torch.allclose(direct.apply(matrix), cached.apply(matrix), atol=1e-6, rtol=1e-6)
    assert torch.allclose(direct.apply_half(matrix), cached.apply_half(matrix), atol=1e-6, rtol=1e-6)
