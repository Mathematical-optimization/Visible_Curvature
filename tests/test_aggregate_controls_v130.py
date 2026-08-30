from __future__ import annotations

import pandas as pd

from visible_curvature.aggregate import _paired_control_contrasts


def _common() -> dict:
    return {
        "protocol_hash": "p",
        "runtime_identity_sha256": "r",
        "block_name": "b",
        "block_type": "attn_q",
        "seed": 0,
        "covariance_moment": "centered",
        "balanced_reliable_for_inference": True,
    }


def test_control_contrasts_are_paired_within_block_and_seed():
    interventions = pd.DataFrame(
        [
            {**_common(), "assignment": "aligned", "alpha": 0.25, "delta_g": 1.2},
            {**_common(), "assignment": "reversed", "alpha": 0.25, "delta_g": -0.8},
        ]
    )
    alpha = pd.DataFrame(
        [
            {**_common(), "assignment": "observed", "alpha": 0.25, "delta_g": 0.4},
            {**_common(), "assignment": "observed", "alpha": 0.5, "delta_g": 0.9},
        ]
    )
    damping = pd.DataFrame(
        [
            {
                **_common(),
                "assignment": "observed",
                "alpha": 0.25,
                "sweep_mode": "joint",
                "damping_coefficient": 0.0,
                "control_estimand": "abs_delta_g",
                "control_value": 2.0,
            },
            {
                **_common(),
                "assignment": "observed",
                "alpha": 0.25,
                "sweep_mode": "joint",
                "damping_coefficient": 1.0,
                "control_estimand": "abs_delta_g",
                "control_value": 0.5,
            },
        ]
    )

    contrasts = _paired_control_contrasts(interventions, alpha, damping)
    assignment = contrasts.query(
        "contrast_type == 'assignment_aligned_minus_reversed'"
    ).iloc[0]
    assert assignment["contrast_value"] == 2.0
    assert bool(assignment["sign_reversal"])
    assert bool(assignment["paired_reliable"])

    alpha_response = contrasts.query(
        "contrast_type == 'alpha_signed_change_from_practical'"
    ).iloc[0]
    assert alpha_response["reference_label"] == "0.25"
    assert alpha_response["comparison_label"] == "0.5"
    assert alpha_response["contrast_value"] == 0.5

    attenuation = contrasts.query(
        "contrast_type == 'damping_change_from_minimum'"
    ).iloc[0]
    assert attenuation["contrast_value"] == -1.5
    assert attenuation["expected_direction"] == "nonpositive"
    assert bool(attenuation["expectation_satisfied"])
