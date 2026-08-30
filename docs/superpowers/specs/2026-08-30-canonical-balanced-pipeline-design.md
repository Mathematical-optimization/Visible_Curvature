# Canonical Balanced Reliability Pipeline v1.2.0 Design

## Goal

Produce a submission-ready v1.2.0 experiment package whose scientific outputs are generated only from the balanced reliability path, whose curvature operator is fixed across numerical-budget stages, whose factor-assignment controls are admitted only when their partial-trace geometry is stable, and whose synthetic tests cover both the flat-Kronecker conditioning algebra and the weighted-Chebyshev minimax barrier.

## Scientific scope

The package continues to test frozen Adam- and Shampoo-form operators at fixed checkpoints. It does not claim online optimizer dominance. The primary estimand remains

\[
\Delta G = \log K_{\mathrm{Adam}}-\log K_{\mathrm{Shampoo}},
\qquad
K_\Phi=\operatorname{cond}(P_\Phi^{1/2}HP_\Phi^{1/2}).
\]

The canonical primary row is centered covariance, observed assignment, and factor exponent \(\alpha=1/4\). Uncentered moments, aligned/reversed assignments, \(\alpha\)-sweeps, and damping sweeps remain controls.

## Architecture

The existing frozen core remains responsible for checkpoint loading, covariance estimation, curvature matvecs, preconditioners, Lanczos spectra, and control evaluation. The balanced orchestrator becomes the sole scientific promotion layer. It calibrates one curvature shift per block, reuses that shift in all later stages, evaluates endpoint and partial-trace stability, executes the final run, verifies final-versus-diagnostic agreement, and writes canonical tables.

Aggregation and paper export consume canonical balanced tables for scientific runs. Legacy tables remain available for compatibility and debug workflows but cannot satisfy the scientific export gate.

## Curvature stabilization

A calibration run computes one block-specific stabilization shift using a preregistered calibration budget. The shift is persisted in `curvature_shift_overrides.json`. Later diagnostic and final configurations set `analysis.curvature.shift_overrides_path` and do not recompute the shift for blocks present in the override file.

The frozen core records the raw Ritz endpoints, target ridge, applied shift, shift source, and override-file hash. The canonical ridge schema remains:

```yaml
analysis:
  curvature:
    psd_mode: shift
    ridge: 1.0e-5
    ridge_mode: relative_max
```

`fixed_relative_ridge` in the balanced policy must set `analysis.curvature.ridge` and force `ridge_mode: relative_max`; failure to expose these fields is an error.

## Spectral gain curves

Every Adam/Shampoo comparison emits one row per configured truncation threshold to `spectral_gain_curve.csv`. Rows identify block, covariance moment, assignment, alpha, damping mode/coefficient, condition numbers, signed gain, and saturation indicators. The balanced tau classifier must use this table. Missing primary tau rows make the primary inference unavailable.

## Partial-trace stability

Each run writes `partial_trace_artifacts/<block-id>.npz` containing left/right partial traces and their eigendecompositions. Consecutive stages are compared using:

- relative Frobenius matrix change;
- negative spectral-mass level and change;
- spectral-cluster projector distance, where clusters are formed with a relative eigengap tolerance;
- aligned/reversed intervention-factor relative change.

The partial-trace acceptance result is decomposed into PSD, matrix, subspace, and intervention-factor checks. Assignment controls requiring a curvature factor basis are admitted only when all four pass.

## Endpoint and final-run reliability

Adaptive endpoint acceptance requires the native Ritz-residual check plus stability of \(K_{\mathrm{Adam}}\), \(K_{\mathrm{Shampoo}}\), and \(\Delta G\) across consecutive budgets. The final primary row must independently pass its native endpoint check and agree with the selected diagnostic row within the same tolerances.

The primary inferential gate is:

```text
adaptive endpoint checks
AND final native endpoint checks
AND final-versus-diagnostic agreement
AND finite bootstrap count
AND nonzero bootstrap CI consistent with the point estimate
AND tau classification in {stable_nonzero, one_sided_with_coarse_saturation}
```

The external terminology is `numerically_accepted` or `checks_passed`. Existing `*_certified` columns remain compatibility aliases.

## Canonical outputs

The balanced final directory contains:

```text
canonical_block_metrics.csv
canonical_interventions.csv
canonical_alpha_sweep.csv
canonical_damping_sweep.csv
spectral_gain_curve.csv
partial_trace_convergence.csv
balanced_reliability_checks.csv
scientific_status.json
```

`scientific_status.json` separates pipeline completion from scientific availability:

```json
{
  "pipeline_status": "complete",
  "scientific_status": "accepted or inconclusive",
  "primary_inference_available": true
}
```

## Aggregation and export

`aggregate_frozen_runs` resolves a balanced pipeline root to its `final/` directory. If any source is balanced, all scientific sources must be balanced. Scientific aggregation uses canonical tables and `balanced_reliable_ordering`. The aggregate manifest records `reliability_mode`, `all_sources_balanced`, `canonical_tables_used`, minimum-seed status, and whether all primary rows passed numerical checks.

Paper export requires a complete balanced aggregate with canonical tables. Debug export may consume legacy aggregates but is watermarked.

## Damping controls

The damping sweep supports two modes:

- `joint`: change both Adam and Shampoo normalized damping coefficients;
- `shampoo_only`: hold Adam at its primary damping and vary only Shampoo.

Rows record separate Adam and Shampoo coefficients and the sweep mode.

## Synthetic verification

The existing dense flat-Kronecker checks are renamed as conditioning-mechanism checks. A new weighted-Chebyshev module computes extremal nodes, Lagrange weights, \(C_T(K)\), the scaled Chebyshev polynomial, and the exact constrained weighted least-squares optimum for small budgets. Tests verify positivity, normalization, equality of the optimum with \(C_T(K)\), and the three-group \(C_T(K)/3\) lower-bound certificate.

## Reproducibility and release

The package version is 1.2.0. Clean-shell smoke scripts set the repository root on `PYTHONPATH`. Scientific grouped bootstrap requires `num_batches % group_size == 0`. Documentation, manifests, tests, and checksums are updated together. The final archive must pass unit tests, both smoke workflows, compile checks, and its internal SHA-256 manifest.
