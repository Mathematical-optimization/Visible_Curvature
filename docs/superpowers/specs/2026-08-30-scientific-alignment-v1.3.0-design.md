# Visible-Curvature Experiment Package v1.3.0 — Scientific Alignment Design

## Goal

Upgrade the canonical balanced experiment package so that every exported empirical quantity matches the frozen-operator theory it is claimed to test, numerical estimands remain compatible across budgets and seeds, expensive repeated spectra are cached, and confirmatory outputs carry sufficient provenance and reporting metadata for paper use.

## Scope

This release preserves the existing frozen covariance/GGN architecture and public command-line entry points. It changes diagnostics, reliability promotion, aggregation, controls, reporting, configuration, and synthetic validation. It does not claim online Adam/Shampoo dominance or add trajectory training experiments.

## Design decisions

### 1. Predictor decomposition

Replace the single consumption-only `delta_g_predicted` interpretation with three explicit quantities:

- `baseline_width_mismatch = W_adam - (W_left + W_right)`;
- `delta_g_predicted_consumption = alpha(r_left W_left + r_right W_right) - 0.5 r_adam W_adam`;
- `delta_g_predicted_full_proxy = baseline_width_mismatch + delta_g_predicted_consumption`.

The full quantity is labeled a commuting–Kronecker proxy, not an exact predictor. The legacy `delta_g_predicted` column aliases the full proxy for schema continuity. Reliability requires finite predictor components, accepted left/right/Adam regressions, sufficient modes and log-width, bounded factor commutators/eigen residuals/negative mass, and non-floor-dominated roots.

### 2. Correct control estimands

Every control row records `K_H`, `G_adam = log K_H - log K_adam`, and `G_shampoo = log K_H - log K_shampoo` using compatible condition metrics. Joint damping is evaluated through `abs(delta_g)`. Shampoo-only damping is evaluated through `abs(G_shampoo)` or equivalently distance to the scalar limit, not `abs(delta_g)`. Alpha controls report signed paired response `delta_g(alpha)-delta_g(alpha=0.25)`; absolute amplification is secondary and only meaningful in matched-baseline subsets.

### 3. Metric compatibility and fail-closed promotion

Ordinary and truncated conditions are distinct estimands. Adaptive stage convergence, final agreement, seed aggregation, and scientific export require metric consistency. By default, only ordinary-condition primary rows may be scientifically promoted; truncated rows remain auditable secondary diagnostics. A policy switch may explicitly allow truncated primary results, but the metric and tau must then agree across stages and seeds.

### 4. Spectrum caching

Introduce a block-local cache for Lanczos spectra keyed by operator identity, damping, assignment, alpha, budget, and common-start identity. Reuse Adam spectra across intervention and alpha controls, and across Shampoo-only damping; reuse identical Shampoo spectra when controls overlap. Cache behavior is transparent and does not change numerical values.

### 5. Confirmatory preregistration and reporting

The confirmatory template selects twelve exact OPT-125M projection weights spanning shallow/middle/deep layers and q/out projections. Expensive interventions/alpha/damping controls may be restricted to a preregistered control subset. Scientific exports include block type, both condition numbers, delta gain, confidence interval, condition metric, tau, tail-localization state, seed count, and consensus.

### 6. Mechanism inference and paired contrasts

Aggregation produces a reliability-gated elasticity prediction summary and blockwise paired control contrasts. Predictor reporting includes sign accuracy, balanced accuracy, and Spearman correlation only on eligible rows. Assignment, alpha, and damping summaries use within-block paired contrasts before cross-block aggregation.

### 7. Provenance and sensitivity

Run manifests record package/source digest, Python/PyTorch/dependency versions, CUDA/cuDNN, GPU/CPU/OS, deterministic-algorithm status, and warnings. Add an optional curvature-ridge sensitivity sweep. Documentation states that covariance is centered mini-batch-mean block-gradient covariance collected in evaluation mode, the curvature is an empirical mean-CE GGN on a disjoint stream, and assignment interventions are frozen-operator interventions that need not realize a full covariance with the observed spectrum and diagonal.

### 8. Synthetic validation

Add an integrated, budget-dependent Theorem-3 witness validator that constructs the three Chebyshev mode groups, reciprocal closure, flat basis, common initialization energy weights, and verifies simultaneous scalar/aligned/reversed lower certificates and dimension bounds. Existing component validators remain.

## Compatibility

Existing scripts and CSV filenames remain. New columns and summary files are additive. `delta_g_predicted` is retained as an alias for `delta_g_predicted_full_proxy`. Version is bumped to 1.3.0 and schema metadata records the new release.

## Verification

The release must pass the complete unit suite, synthetic theory validator, focused smoke pipeline, balanced smoke pipeline, package checksums, and source-package import/build checks. Network-dependent OPT-125M execution remains outside the offline verification environment and must be reported as unexecuted.
