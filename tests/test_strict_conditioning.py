from __future__ import annotations

import math

import pytest
import torch

from ovc_experiments.blocks import MatrixLayout
from ovc_experiments.operators import DenseOperator
from ovc_experiments.preconditioners import AdamPreconditioner, ShampooPreconditioner
from ovc_experiments.spectral import estimate_condition


def test_default_condition_policy_censors_a_null_direction() -> None:
    operator = DenseOperator(torch.diag(torch.tensor([1.0, 0.0], dtype=torch.float64)))

    estimate = estimate_condition(operator, exact_max_dim=8, positive_threshold=1e-12)

    assert estimate.censored
    assert estimate.condition_number is None
    assert estimate.null_eigenvalues == 1
    assert estimate.censor_reason == "null_eigenvalues_in_strict_spd_mode"


def test_positive_active_condition_number_requires_explicit_opt_in() -> None:
    operator = DenseOperator(torch.diag(torch.tensor([1.0, 0.0], dtype=torch.float64)))

    estimate = estimate_condition(
        operator,
        exact_max_dim=8,
        positive_threshold=1e-12,
        subspace_policy="positive_active",
    )

    assert not estimate.censored
    assert estimate.condition_number is not None
    assert math.isclose(estimate.condition_number, 1.0, rel_tol=1e-12)
    assert estimate.null_eigenvalues == 1


def test_adam_strict_mode_does_not_silently_delete_zero_statistic() -> None:
    layout = MatrixLayout.from_shape((1, 2))
    preconditioner = AdamPreconditioner(
        torch.tensor([1.0, 0.0], dtype=torch.float64),
        layout=layout,
        damping=0.0,
    )

    with pytest.raises(ValueError, match="strict_spd"):
        preconditioner.apply_sqrt(torch.ones(2, dtype=torch.float64))


def test_shampoo_positive_active_mode_is_explicit_and_singular() -> None:
    layout = MatrixLayout.from_shape((2, 1))
    preconditioner = ShampooPreconditioner(
        torch.diag(torch.tensor([1.0, 0.0], dtype=torch.float64)),
        torch.ones((1, 1), dtype=torch.float64),
        layout=layout,
        alpha=0.25,
        subspace_policy="positive_active",
    )

    output = preconditioner.apply_sqrt(torch.ones(2, dtype=torch.float64))

    assert output[0] > 0
    assert output[1] == 0
