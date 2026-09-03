# v1.1.0 hardened

## Correctness

- Replaced pooled-Ritz minimum logic with endpoint-specific certification and optional inverse iteration/CG.
- Unresolved smallest positive eigenvalues are censored; no arbitrary finite condition number is emitted.
- Added scale-relative active-spectrum thresholds and explicit strict-SPD/active-subspace/pseudoinverse policies.
- Added SLQ-q01 inconsistency guard and persistent solver diagnostics.

## Scalability

- Added O(d) diagonal operators.
- Added O(d+r^2+c^2) streaming Adam/Shampoo moment accumulation.
- Added a full-scale `analyze_block_streaming` path that never retains the N x d gradient tensor.

## Experimental design

- GGN/Hessian are primary; empirical Fisher is control-only by default.
- H2 requires a positive, held-out association for confirmatory support.
- H3 requires signed aligned/reversed gain reversal within checkpoint.
- H4 distinguishes Shampoo-to-scalar contraction from the limit of delta_G.
- H5 compares alpha=1/4 directly with alpha=1/2.
- H6 groups by run/seed/checkpoint/block.

## Reproducibility

- Added Hadamard-flat reciprocal-closed Theorem 3 validation.
- Added a weighted discrete Chebyshev witness validator.
- Added same-original-error frozen dynamics.
- Added side-specific normalized damping and factor-rank diagnostics.
