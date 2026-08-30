from __future__ import annotations

import math
import warnings

import pandas as pd
import pytest

from visible_curvature.mechanism import (
    build_elasticity_prediction_rows,
    summarize_elasticity_predictions,
)


def _metrics() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "block_name": "positive",
                "seed": 0,
                "covariance_moment": "centered",
                "assignment": "observed",
                "alpha": 0.25,
                "delta_g": 0.8,
                "delta_g_predicted_consumption": 0.1,
                "delta_g_predicted_full_proxy": 0.6,
                "factor_elasticity_reliable": True,
                "balanced_primary_reliable": True,
            },
            {
                "block_name": "negative",
                "seed": 0,
                "covariance_moment": "centered",
                "assignment": "observed",
                "alpha": 0.25,
                "delta_g": -0.9,
                "delta_g_predicted_consumption": 0.2,
                "delta_g_predicted_full_proxy": -0.5,
                "factor_elasticity_reliable": True,
                "balanced_primary_reliable": True,
            },
            {
                "block_name": "rejected",
                "seed": 0,
                "covariance_moment": "centered",
                "assignment": "observed",
                "alpha": 0.25,
                "delta_g": 1.0,
                "delta_g_predicted_consumption": 1.0,
                "delta_g_predicted_full_proxy": 1.0,
                "factor_elasticity_reliable": False,
                "balanced_primary_reliable": True,
            },
            {
                "block_name": "nonprimary",
                "seed": 0,
                "covariance_moment": "uncentered",
                "assignment": "observed",
                "alpha": 0.25,
                "delta_g": 1.0,
                "delta_g_predicted_consumption": 1.0,
                "delta_g_predicted_full_proxy": 1.0,
                "factor_elasticity_reliable": True,
                "balanced_primary_reliable": True,
            },
        ]
    )


def test_elasticity_prediction_rows_record_eligibility_and_reason_codes():
    rows = build_elasticity_prediction_rows(_metrics())
    assert set(rows["predictor_name"]) == {"consumption", "full_proxy"}
    eligible = rows[rows["prediction_eligible"]]
    assert len(eligible) == 4
    rejected = rows[rows["block_name"] == "rejected"]
    assert not rejected["prediction_eligible"].any()
    assert rejected["prediction_eligibility_reasons"].str.contains(
        "factor_elasticity"
    ).all()
    nonprimary = rows[rows["block_name"] == "nonprimary"]
    assert not nonprimary["prediction_eligible"].any()
    assert nonprimary["prediction_eligibility_reasons"].str.contains(
        "nonprimary"
    ).all()


def test_elasticity_prediction_summary_uses_only_eligible_rows():
    summary = summarize_elasticity_predictions(
        build_elasticity_prediction_rows(_metrics())
    ).set_index("predictor_name")
    full = summary.loc["full_proxy"]
    consumption = summary.loc["consumption"]
    assert full["n_pairs"] == 2
    assert full["sign_accuracy"] == 1.0
    assert full["sign_balanced_accuracy"] == 1.0
    assert full["spearman"] == pytest.approx(1.0)
    assert consumption["sign_accuracy"] == 0.5
    assert consumption["sign_balanced_accuracy"] == 0.5
    assert consumption["spearman"] == pytest.approx(-1.0)
    assert not math.isnan(float(consumption["spearman"]))


def test_constant_predictor_returns_nan_without_scipy_warning():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        summary = summarize_elasticity_predictions(
            build_elasticity_prediction_rows(
                pd.DataFrame(
                    [
                        {
                            "block_name": name,
                            "covariance_moment": "centered",
                            "assignment": "observed",
                            "alpha": 0.25,
                            "delta_g": value,
                            "delta_g_predicted_consumption": 0.1,
                            "delta_g_predicted_full_proxy": 0.1,
                            "factor_elasticity_reliable": True,
                            "balanced_primary_reliable": True,
                        }
                        for name, value in (("a", -1.0), ("b", 1.0))
                    ]
                )
            )
        )
    assert not caught
    assert summary["spearman"].isna().all()
