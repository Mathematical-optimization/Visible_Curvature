import pytest
import torch
from ovc_experiments.spectral_power import symmetric_matrix_power


def test_negative_power_does_not_silently_delete_spd_direction():
    matrix = torch.diag(torch.tensor([1e-14, 1.0], dtype=torch.float64))
    powered = symmetric_matrix_power(matrix, -0.25, subspace_policy='strict_spd')
    assert powered[0, 0] > 0


def test_singular_negative_power_requires_explicit_policy():
    matrix = torch.diag(torch.tensor([0.0, 1.0], dtype=torch.float64))
    with pytest.raises(ValueError):
        symmetric_matrix_power(matrix, -0.25, subspace_policy='strict_spd')
    pseudo = symmetric_matrix_power(matrix, -0.25, subspace_policy='pseudoinverse')
    assert pseudo[0, 0] == 0
