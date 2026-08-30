import math

from visible_curvature.diagnostics import predicted_delta_g


def test_predicted_delta_g_scales_factor_response_with_alpha():
    kwargs = dict(
        r_left=2.0,
        width_left=1.5,
        r_right=1.0,
        width_right=2.0,
        r_adam=0.0,
        width_adam=3.5,
    )
    practical = predicted_delta_g(**kwargs, factor_exponent=0.25)
    control = predicted_delta_g(**kwargs, factor_exponent=0.5)

    assert math.isclose(control, 2.0 * practical, rel_tol=1e-12)


def test_predicted_delta_g_default_is_practical_shampoo_exponent():
    kwargs = dict(
        r_left=1.0,
        width_left=2.0,
        r_right=0.5,
        width_right=4.0,
        r_adam=0.25,
        width_adam=3.0,
    )
    assert math.isclose(
        predicted_delta_g(**kwargs),
        predicted_delta_g(**kwargs, factor_exponent=0.25),
        rel_tol=1e-12,
    )
