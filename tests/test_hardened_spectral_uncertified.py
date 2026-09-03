import math
import torch
from ovc_experiments.safe_operators import FunctionOperator
from ovc_experiments.hardened_spectral import estimate_condition


def test_plain_lanczos_never_labels_a_converged_large_ritz_value_as_the_minimum():
    diagonal = torch.logspace(0, 6, 3000, dtype=torch.float64)
    operator = FunctionOperator(3000, lambda x: diagonal * x, dtype=torch.float64)
    result = estimate_condition(operator, exact_condition_max_dim=512, lanczos_steps=24, starts=1, minimum_method='lanczos', residual_tolerance=1e-5)
    censored = bool(getattr(result, 'censored', getattr(result, 'is_censored', True)))
    condition = float(getattr(result, 'condition_number', getattr(result, 'condition', math.inf)))
    assert censored or math.isclose(condition, 1e6, rel_tol=0.05)
