# Historical verification notice

This v1.1.0 report is retained for provenance and is superseded by
`VERIFICATION_REPORT_v1.2.0.md`. Its pass status does not cover the v1.2.0
submission-hardening changes.

# Optimizer-Visible Curvature v1.1.0 — Verification Report

- Overall: **PASS**
- Python: `3.13.5`
- Import package: `ovc_experiments`
- `compileall`: exit `0`
- full `pytest -q`: exit `0`
- `scripts/validate_hardened.py`: exit `0`

## Full pytest output

```text
..................................................................       [100%]
=============================== warnings summary ===============================
tests/test_curvature.py: 18 warnings
  /opt/pyvenv/lib/python3.13/site-packages/torch/jit/_script.py:1480: DeprecationWarning: `torch.jit.script` is deprecated. Please switch to `torch.compile` or `torch.export`.
    warnings.warn(

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
66 passed, 18 warnings in 5.58s


```

## Hardened validator output

```text
{
  "diagonal_condition": 1000000.0,
  "flat_kron_passed": true,
  "generic_large_operator_safe": true,
  "streaming_count": 16,
  "weighted_chebyshev_passed": true
}


```
