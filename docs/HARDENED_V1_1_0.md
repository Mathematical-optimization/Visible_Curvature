# Historical notice

This document describes v1.1.0 and is retained for provenance. It is superseded
by [`HARDENED_V1_2_0.md`](HARDENED_V1_2_0.md). Do not use v1.1.0 condition,
hypothesis, or model-scale execution semantics for the ICLR 2027 submission.

# Optimizer-Visible Curvature Experiments v1.1.0

## Primary execution path

Use `ovc_experiments.hardened_runner.analyze_block_streaming` for full-scale blocks. It accepts a replayable per-example matrix-gradient factory and accumulates Adam/Shampoo moments in O(d+r^2+c^2) memory. It never stores the N x d gradient matrix.

The legacy runner is retained for backward-compatible small smoke tests. It is not the confirmatory full-scale path.

## Curvature policy

GGN is the default primary curvature. `fisher` is normalized to `empirical_fisher` and is rejected as a primary assignment analysis unless the caller explicitly overrides the guard. It remains available as a representation/control panel.

## Spectral policy

Large-dimensional minima are endpoint-specific. A result is finite and uncensored only after residual and budget-stability certification, with inverse iteration/CG as a fallback. Unresolved minima return a censored infinite condition number. All solver diagnostics can be written to JSONL with `set_condition_diagnostics_path`.

## Active subspaces

Negative matrix powers use `strict_spd` by default. Singular matrices require explicit `positive_active` or `pseudoinverse` semantics. Thresholds are relative to the operator scale plus a machine-precision floor.

## Hypotheses

- H2 requires a positive association; negative Spearman correlations are not support.
- H3 requires signed gain reversal within run/seed/checkpoint/block.
- H4 tests Shampoo gain contraction to zero; delta_G generally tends to -G_adam unless Adam is co-damped.
- H5 compares alpha=1/4 directly with alpha=1/2 and requires amplification in both favorable and unfavorable regimes.
- H6 groups by checkpoint before testing scalar invariance.

## Reproducibility validators

`validate_flat_kron_pair` implements the Hadamard-flat reciprocal-closed Theorem 3 pair. `validate_weighted_chebyshev` validates the discrete weighted Chebyshev barrier against its closed form.

## Output artifacts

The hardened block runner writes `geometry.csv`, `solver_diagnostics.jsonl`, `solver_diagnostics.csv`, and `moments_summary.json`. Relative output paths are resolved against the supplied config file through `config_path`.

## H2 confirmatory evaluation

Use `evaluate_h2_leave_one_cluster_out`; in-sample positive correlations are labeled `descriptive_only`, not `supported`.
