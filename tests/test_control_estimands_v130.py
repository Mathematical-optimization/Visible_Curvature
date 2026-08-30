import math

from visible_curvature.analysis_runner import (
    _annotate_alpha_control_rows,
    _control_estimand_record,
    _gain_record,
)


def test_gain_record_uses_common_h_condition():
    record = _gain_record(K_H=100.0, K_adam=25.0, K_shampoo=4.0)
    assert math.isclose(record["G_adam"], math.log(100.0) - math.log(25.0))
    assert math.isclose(record["G_shampoo"], math.log(100.0) - math.log(4.0))
    assert math.isclose(record["delta_g_from_gains"], record["G_shampoo"] - record["G_adam"])


def test_joint_damping_targets_absolute_delta_g():
    record = _control_estimand_record(
        sweep_mode="joint",
        delta_g=-0.7,
        G_adam=0.2,
        G_shampoo=-0.5,
    )
    assert record["control_estimand"] == "abs_delta_g"
    assert math.isclose(record["control_value"], 0.7)


def test_shampoo_only_damping_targets_distance_to_scalar_limit():
    record = _control_estimand_record(
        sweep_mode="shampoo_only",
        delta_g=-0.7,
        G_adam=0.2,
        G_shampoo=-0.5,
    )
    assert record["control_estimand"] == "abs_g_shampoo"
    assert math.isclose(record["control_value"], 0.5)
    assert math.isclose(record["delta_g_scalar_limit"], -0.2)
    assert math.isclose(record["delta_g_distance_to_scalar_limit"], 0.5)


def test_alpha_rows_use_within_block_signed_change_from_practical():
    rows = [
        {"block_name": "b0", "seed": 0, "assignment": "aligned", "alpha": 0.25, "delta_g": -0.8},
        {"block_name": "b0", "seed": 0, "assignment": "aligned", "alpha": 0.5, "delta_g": 0.4},
        {"block_name": "b0", "seed": 0, "assignment": "reversed", "alpha": 0.25, "delta_g": -1.3},
        {"block_name": "b0", "seed": 0, "assignment": "reversed", "alpha": 0.5, "delta_g": -0.4},
    ]
    annotated = _annotate_alpha_control_rows(rows, practical_alpha=0.25)
    by_key = {(row["assignment"], row["alpha"]): row for row in annotated}
    assert math.isclose(by_key[("aligned", 0.25)]["alpha_delta_from_practical"], 0.0)
    assert math.isclose(by_key[("aligned", 0.5)]["alpha_delta_from_practical"], 1.2)
    assert math.isclose(by_key[("reversed", 0.5)]["alpha_delta_from_practical"], 0.9)
    assert by_key[("aligned", 0.5)]["control_estimand"] == "signed_delta_g_change_from_alpha_0.25"


def test_pair_condition_uses_truncated_metric_when_reference_curvature_is_unresolved():
    from visible_curvature.analysis_runner import _pair_conditions

    reference = {"min_ritz": -1.0e-9, "max_ritz": 10.0}
    adam = {"min_ritz": 1.0, "max_ritz": 10.0}
    shampoo = {"min_ritz": 2.0, "max_ritz": 8.0}
    record = _pair_conditions(
        adam,
        shampoo,
        reference_spec=reference,
        relative_floor=1.0e-8,
        fallback_tau=1.0e-4,
    )
    assert record["condition_metric"] == "truncated"
    assert math.isfinite(record["K_H"])
    assert math.isfinite(record["G_adam"])
    assert math.isfinite(record["G_shampoo"])
    assert math.isclose(record["delta_g"], record["delta_g_from_gains"])


def test_control_and_bootstrap_empty_schemas_are_separated():
    from visible_curvature.analysis_runner import BOOTSTRAP_COLUMNS, CONTROL_COLUMNS

    for column in (
        "K_H",
        "G_adam",
        "G_shampoo",
        "control_estimand",
        "control_value",
        "alpha_delta_from_practical",
        "fallback_tau",
    ):
        assert column in CONTROL_COLUMNS
    for column in (
        "control_estimand",
        "control_value",
        "alpha_reference",
        "alpha_delta_from_practical",
    ):
        assert column not in BOOTSTRAP_COLUMNS
