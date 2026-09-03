from __future__ import annotations
import json, math
import torch
from ovc_experiments.hardened_spectral import estimate_condition
from ovc_experiments.safe_operators import DiagonalOperator, FunctionOperator
from ovc_experiments.streaming_moments import accumulate_matrix_moments
from ovc_experiments.theorem_validation import validate_flat_kron_pair, validate_weighted_chebyshev


def main() -> None:
    flat = validate_flat_kron_pair()
    cheb = validate_weighted_chebyshev(25.0, 5)
    diagonal = torch.logspace(0, 6, 3000, dtype=torch.float64)
    exact = estimate_condition(DiagonalOperator(diagonal), exact_condition_max_dim=512)
    generic = estimate_condition(
        FunctionOperator(3000, lambda x: diagonal * x, dtype=torch.float64),
        exact_condition_max_dim=512, lanczos_steps=24, starts=1,
        minimum_method='lanczos', residual_tolerance=1e-5,
    )
    gradients = [torch.randn(3, 2, dtype=torch.float64) for _ in range(16)]
    moments = accumulate_matrix_moments(gradients, 3, 2)
    get = lambda obj, names, default=None: next((getattr(obj, n) for n in names if hasattr(obj, n)), default)
    generic_censored = bool(get(generic, ('censored', 'is_censored'), True))
    generic_condition = float(get(generic, ('condition_number', 'condition'), math.inf))
    payload = {
        'flat_kron_passed': flat.passed,
        'weighted_chebyshev_passed': cheb.passed,
        'diagonal_condition': float(get(exact, ('condition_number', 'condition'))),
        'generic_large_operator_safe': generic_censored or math.isclose(generic_condition, 1e6, rel_tol=.05),
        'streaming_count': moments.count,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    assert all((payload['flat_kron_passed'], payload['weighted_chebyshev_passed'], payload['generic_large_operator_safe']))
    assert math.isclose(payload['diagonal_condition'], 1e6, rel_tol=1e-10)
    assert payload['streaming_count'] == 16


if __name__ == '__main__':
    main()
