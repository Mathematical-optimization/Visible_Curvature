import math

import pandas as pd

from visible_curvature.figures import _alpha_plot_data, _damping_plot_data
from visible_curvature.paper_export import _summarize_control_table


def test_alpha_plot_data_uses_signed_paired_response_column():
    frame = pd.DataFrame(
        [
            {"assignment": "observed", "alpha": 0.25, "delta_g": 10.0, "alpha_delta_from_practical": 0.0},
            {"assignment": "observed", "alpha": 0.5, "delta_g": 20.0, "alpha_delta_from_practical": -0.75},
        ]
    )
    grouped, value = _alpha_plot_data(frame)
    assert value == "alpha_delta_from_practical"
    assert math.isclose(grouped.loc[grouped["alpha"] == 0.5, value].iloc[0], -0.75)


def test_damping_plot_data_uses_declared_control_value_not_abs_delta_g():
    frame = pd.DataFrame(
        [
            {
                "sweep_mode": "shampoo_only",
                "assignment": "observed",
                "damping_coefficient": 1.0,
                "delta_g": -3.0,
                "control_estimand": "abs_g_shampoo",
                "control_value": 0.25,
            }
        ]
    )
    grouped = _damping_plot_data(frame)
    assert grouped.iloc[0]["control_estimand"] == "abs_g_shampoo"
    assert math.isclose(grouped.iloc[0]["control_value"], 0.25)


def test_paper_control_summary_reports_control_value_and_estimand():
    frame = pd.DataFrame(
        [
            {
                "sweep_mode": "shampoo_only",
                "assignment": "observed",
                "damping_coefficient": 1.0,
                "delta_g": -9.0,
                "control_estimand": "abs_g_shampoo",
                "control_value": 0.125,
            }
        ]
    )
    grouped, value_column = _summarize_control_table(
        frame,
        source="damping_sweep.csv",
        group_keys=["assignment", "damping_coefficient"],
    )
    assert value_column == "control_value"
    assert grouped.iloc[0]["control_estimand"] == "abs_g_shampoo"
    assert math.isclose(grouped.iloc[0][value_column], 0.125)
