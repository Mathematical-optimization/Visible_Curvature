from __future__ import annotations

import math

import pandas as pd

from ovc_experiments.statistics import (
    cluster_bootstrap_sign_fractions,
    leave_one_cluster_out_prediction,
    paired_intervention_effects,
)


def test_cluster_bootstrap_sign_fractions_reports_empirical_estimates() -> None:
    frame = pd.DataFrame(
        {
            "run_name": ["a", "a", "b", "b"],
            "checkpoint_step": [0, 1, 0, 1],
            "delta_G_0.25": [1.0, -1.0, 0.1, 2.0],
        }
    )
    result = cluster_bootstrap_sign_fractions(
        frame,
        delta_column="delta_G_0.25",
        cluster_columns=["run_name"],
        threshold=0.2,
        replicates=100,
        seed=3,
    )
    assert math.isclose(result["positive"]["estimate"], 0.5)
    assert math.isclose(result["negative"]["estimate"], 0.25)
    assert math.isclose(result["neutral"]["estimate"], 0.25)
    for label in ("positive", "negative", "neutral"):
        assert 0.0 <= result[label]["lower"] <= result[label]["upper"] <= 1.0


def test_paired_intervention_effects_preserve_within_block_pairing() -> None:
    frame = pd.DataFrame(
        {
            "checkpoint_step": [0, 0, 1, 1],
            "block_name": ["w", "w", "w", "w"],
            "intervention": ["assignment"] * 4,
            "branch": ["aligned", "reversed", "aligned", "reversed"],
            "condition_number": [2.0, 8.0, 3.0, 12.0],
        }
    )
    effects = paired_intervention_effects(
        frame,
        intervention="assignment",
        reference_branch="aligned",
        treatment_branch="reversed",
        value_column="condition_number",
        group_columns=["checkpoint_step", "block_name"],
    )
    assert effects["paired_difference"].tolist() == [6.0, 9.0]
    assert effects["paired_log_ratio"].round(8).tolist() == [
        round(math.log(4.0), 8),
        round(math.log(4.0), 8),
    ]


def test_leave_one_cluster_out_prediction_recovers_simple_mechanism() -> None:
    rows = []
    for cluster in range(4):
        for index in range(5):
            response = -1.0 + 0.5 * index + 0.1 * cluster
            commutator = 0.05 * index
            target = 2.0 * response - 0.5 * commutator
            rows.append(
                {
                    "cluster": cluster,
                    "response": response,
                    "commutator": commutator,
                    "target": target,
                }
            )
    frame = pd.DataFrame(rows)
    result = leave_one_cluster_out_prediction(
        frame,
        target_column="target",
        predictor_columns=["response", "commutator"],
        cluster_columns=["cluster"],
        sign_threshold=0.1,
    )
    assert result.summary["rows"] == len(frame)
    assert result.summary["spearman"] > 0.99
    assert result.summary["sign_accuracy"] > 0.95
