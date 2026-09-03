import math
import torch
from ovc_experiments.safe_operators import DiagonalOperator
from ovc_experiments.hardened_spectral import estimate_condition


def _get(obj, *names):
    for name in names:
        if hasattr(obj, name):
            return getattr(obj, name)
    raise AttributeError(names)


def test_large_diagonal_condition_is_exact_and_not_false_uncensored():
    diagonal = torch.logspace(0, 6, 3000, dtype=torch.float64)
    result = estimate_condition(DiagonalOperator(diagonal), exact_condition_max_dim=512, lanczos_steps=32)
    assert not bool(_get(result, 'censored', 'is_censored'))
    condition = float(_get(result, 'condition_number', 'condition'))
    assert math.isclose(condition, 1e6, rel_tol=1e-12)


def test_condition_is_scale_invariant():
    base = torch.logspace(-3, 3, 700, dtype=torch.float64)
    values = []
    for scale in (1e-12, 1.0, 1e12):
        result = estimate_condition(DiagonalOperator(base * scale), exact_condition_max_dim=128)
        values.append(float(_get(result, 'condition_number', 'condition')))
    assert max(values) / min(values) < 1.000000001
