from visible_curvature.reliability import classify_reliable_ordering, tau_sign_stability


def test_tau_sign_stability_requires_all_resolved_tau_signs_to_agree():
    stable = tau_sign_stability({1.0e-2: 0.4, 1.0e-3: 0.2, 1.0e-4: 0.1})
    assert stable["stable"] is True
    assert stable["sign"] == 1
    unstable = tau_sign_stability({1.0e-2: 0.4, 1.0e-3: -0.2})
    assert unstable["stable"] is False
    assert "sign_flip" in unstable["reason"]


def test_reliable_ordering_demotes_significant_ci_when_numerics_fail():
    result = classify_reliable_ordering(
        delta=0.5,
        ci_low=0.2,
        ci_high=0.8,
        checks={"tau_sign_stable": True, "ritz_residual_ok": False},
    )
    assert result["point_label"] == "shampoo_favorable"
    assert result["ci_label"] == "positive"
    assert result["reliable_label"] == "inconclusive"
    assert result["reliable"] is False
    assert result["reliability_reasons"] == "ritz_residual_ok"


def test_reliable_ordering_accepts_negative_ci_when_all_checks_pass():
    result = classify_reliable_ordering(
        delta=-0.5,
        ci_low=-0.8,
        ci_high=-0.2,
        checks={"tau_sign_stable": True, "ritz_residual_ok": True},
    )
    assert result["reliable_label"] == "negative"
    assert result["reliable"] is True
    assert result["reliability_reasons"] == ""


def test_reliable_ordering_rejects_point_and_ci_sign_mismatch():
    result = classify_reliable_ordering(
        delta=-0.1,
        ci_low=0.2,
        ci_high=0.8,
        checks={"ritz": True, "tau": True},
    )
    assert result["point_label"] == "shampoo_unfavorable"
    assert result["ci_label"] == "positive"
    assert result["reliable_label"] == "inconclusive"
    assert result["reliable"] is False
    assert "point_bootstrap_sign_mismatch" in result["reliability_reasons"]
