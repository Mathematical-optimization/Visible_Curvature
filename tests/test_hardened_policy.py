import pytest
from ovc_experiments.curvature_policy import normalize_curvature_kind, validate_curvature_policy


def test_empirical_fisher_is_control_only_by_default():
    assert normalize_curvature_kind('fisher') == 'empirical_fisher'
    with pytest.raises(ValueError):
        validate_curvature_policy('fisher', primary_analysis=True)
    policy = validate_curvature_policy('empirical_fisher', primary_analysis=False)
    assert policy.kind == 'empirical_fisher'
