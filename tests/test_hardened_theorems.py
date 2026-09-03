from ovc_experiments.theorem_validation import validate_flat_kron_pair, validate_weighted_chebyshev


def test_flat_kron_pair_matches_theorem_three_invariants():
    result = validate_flat_kron_pair()
    assert result.passed


def test_weighted_chebyshev_witness_matches_closed_form_barrier():
    result = validate_weighted_chebyshev(25.0, 5)
    assert result.passed
