from __future__ import annotations

import pandas as pd

from ovc_experiments.hardened_reporting import evaluate_hypotheses
from ovc_experiments.reporting import summarize_hypotheses


def _status(summary: dict[str, object], hypothesis: str) -> str:
    return str(summary[hypothesis]["status"])


def test_public_summary_rejects_negative_h2_with_wide_geometry_schema() -> None:
    geometry = pd.DataFrame(
        {
            "delta_G_0.25": [3.0, 2.0, 1.0, 0.0],
            "response_shampoo": [0.0, 1.0, 2.0, 3.0],
            "curvature_censored": [False] * 4,
            "adam_censored": [False] * 4,
            "shampoo_0.25_censored": [False] * 4,
            # This unrelated optional operator must not censor the primary panel.
            "optimizer_state_censored": [True] * 4,
        }
    )

    summary = summarize_hypotheses(
        geometry,
        pd.DataFrame(columns=["intervention", "branch"]),
        sign_threshold=1e-8,
    )

    assert _status(summary, "H2") == "not_supported"


def test_h3_requires_signed_gain_reversal_not_condition_ordering() -> None:
    geometry = pd.DataFrame({"delta_G_0.25": [1.0, -1.0]})
    interventions = pd.DataFrame(
        {
            "run_name": ["r", "r"],
            "checkpoint_step": [1, 1],
            "block_name": ["b", "b"],
            "intervention": ["assignment", "assignment"],
            "branch": ["aligned", "reversed"],
            "condition_number": [29.24, 29.25],
            "gain": [-3.37, -3.38],
            "G_adam": [0.0, 0.0],
            "censored": [False, False],
        }
    )

    summary = summarize_hypotheses(geometry, interventions, sign_threshold=1e-8)

    assert _status(summary, "H3") == "not_supported"


def test_h5_requires_favorable_and_unfavorable_regimes() -> None:
    geometry = pd.DataFrame({"delta_G_0.25": [1.0, -1.0]})
    interventions = pd.DataFrame(
        {
            "run_name": ["r", "r", "r", "r"],
            "checkpoint_step": [1, 1, 1, 1],
            "block_name": ["bad-1", "bad-1", "bad-2", "bad-2"],
            "intervention": ["alpha"] * 4,
            "branch": ["natural"] * 4,
            "alpha": [0.25, 0.5, 0.25, 0.5],
            "gain": [-1.0, -2.0, -0.5, -1.0],
            "censored": [False] * 4,
        }
    )

    summary = summarize_hypotheses(geometry, interventions, sign_threshold=1e-8)

    assert _status(summary, "H5") == "not_supported"


def test_h6_groups_grafting_by_checkpoint_not_only_block_name() -> None:
    geometry = pd.DataFrame({"delta_G_0.25": [1.0, -1.0]})
    interventions = pd.DataFrame(
        {
            "run_name": ["r"] * 4,
            "checkpoint_step": [1, 1, 2, 2],
            "block_name": ["b"] * 4,
            "intervention": ["grafting"] * 4,
            "branch": ["natural"] * 4,
            "scale": [0.5, 2.0, 0.5, 2.0],
            # Invariant within each checkpoint; different baselines across checkpoints.
            "condition_number": [10.0, 10.0, 100.0, 100.0],
            "censored": [False] * 4,
        }
    )

    summary = summarize_hypotheses(geometry, interventions, sign_threshold=1e-8)

    assert _status(summary, "H6") == "supported"


def test_h4_uses_shampoo_gain_contraction_on_rho_over_min_schema() -> None:
    geometry = pd.DataFrame({"delta_G_0.25": [1.0, -1.0]})
    interventions = pd.DataFrame(
        {
            "run_name": ["r"] * 3,
            "checkpoint_step": [1] * 3,
            "block_name": ["b"] * 3,
            "intervention": ["damping"] * 3,
            "branch": ["natural"] * 3,
            "rho_over_min": [0.0, 1.0, 100.0],
            "gain": [-2.0, -1.0, -0.1],
            "censored": [False] * 3,
        }
    )

    summary = evaluate_hypotheses(geometry, interventions)
    row = summary.query("hypothesis == 'H4'").iloc[0]

    assert row.status == "supported"
