from __future__ import annotations

import math

import torch

from ovc_experiments.blocks import BlockSpec, MatrixLayout
from ovc_experiments.config import ExperimentConfig
from ovc_experiments.dynamics import QuadraticTrajectory
from ovc_experiments.geometry import measure_frozen_geometry
from ovc_experiments.moments import estimate_moments_from_gradients
from ovc_experiments.operators import DenseOperator
from ovc_experiments.preconditioners import AdamPreconditioner, IdentityPreconditioner
from ovc_experiments.runners import BlockAnalysis, _dynamics_rows


def test_runner_transforms_one_original_error_for_every_preconditioner(monkeypatch) -> None:
    layout = MatrixLayout.from_shape((2, 1))
    block = BlockSpec("weight", (2, 1), layout, 2)
    curvature = DenseOperator(torch.diag(torch.tensor([1.0, 4.0], dtype=torch.float64)))
    identity = IdentityPreconditioner(layout, dtype=torch.float64, device="cpu")
    adam = AdamPreconditioner(
        torch.tensor([1.0, 16.0], dtype=torch.float64),
        layout=layout,
        damping=1e-6,
    )
    preconditioners = {"identity": identity, "adam": adam}
    geometries = {
        name: measure_frozen_geometry(curvature, preconditioner, exact_max_dim=8)
        for name, preconditioner in preconditioners.items()
    }
    gradients = torch.tensor(
        [[[1.0], [2.0]], [[2.0], [1.0]], [[3.0], [4.0]]], dtype=torch.float64
    )
    moments = estimate_moments_from_gradients(gradients, layout)
    eigenvalues, eigenvectors = torch.linalg.eigh(curvature.matrix)
    analysis = BlockAnalysis(
        block=block,
        gradients=gradients,
        moments=moments,
        curvature=curvature,
        curvature_matrix=curvature.matrix,
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
        centered=False,
        shampoo_damping_left=0.0,
        shampoo_damping_right=0.0,
        preconditioners=preconditioners,
        geometries=geometries,
        functional=None,  # not used by the frozen quadratic runner
        batch=None,
    )
    config = ExperimentConfig()
    config.run.name = "same-error"
    config.run.seed = 19
    config.geometry.dynamics_steps = 1

    captured: dict[str, torch.Tensor] = {}

    def fake_trajectory(operator, initial, **kwargs):
        name = operator.name.removeprefix("effective-")
        captured.setdefault(name, initial.detach().clone())
        return QuadraticTrajectory(
            method="fake",
            relative_objective=torch.tensor([1.0, 0.5], dtype=torch.float64),
            gradient_norms=torch.tensor([1.0, 0.5], dtype=torch.float64),
            step_sizes=torch.tensor([0.1], dtype=torch.float64),
            final_vector=initial.detach().clone(),
        )

    monkeypatch.setattr("ovc_experiments.runners.run_gradient_descent", fake_trajectory)
    monkeypatch.setattr("ovc_experiments.runners.run_chebyshev", fake_trajectory)
    monkeypatch.setattr("ovc_experiments.runners.run_conjugate_gradient", fake_trajectory)

    frame = _dynamics_rows(config, [analysis])

    reconstructed = {
        name: preconditioners[name].apply_sqrt(initial)
        for name, initial in captured.items()
    }
    assert set(reconstructed) == {"identity", "adam"}
    assert torch.allclose(reconstructed["identity"], reconstructed["adam"], atol=1e-12)
    assert frame["initial_error_sha256"].nunique() == 1
    assert frame["initial_objective"].nunique() == 1
    assert math.isfinite(float(frame["initial_objective"].iloc[0]))
