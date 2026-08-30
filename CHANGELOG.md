# Changelog

## v1.2.1 — Partial-trace performance and execution safety

- Replace per-cluster ambient projector spectral norms with the equivalent principal-angle overlap formula, eliminating repeated full-size SVDs during CPU-only partial-trace certification.
- Add regression tests for projector-distance equivalence and the no-full-projector performance contract.
- Add a process-lifetime exclusive lock for each balanced output root, preventing duplicate seed writers from corrupting artifacts.
- Emit timestamped progress messages that distinguish child core analysis from parent-side CPU certification.
- Allow generated seed policies to pin `CUDA_VISIBLE_DEVICES` and BLAS/OpenMP thread counts through `--gpus` and `--cpu-threads`.
- Bump package and release documentation to v1.2.1 without changing the scientific estimand or acceptance thresholds.

## v1.2.0 — Canonical balanced scientific pipeline

### Reliability and operator consistency

- Calibrate one curvature stabilization shift per block and freeze it across diagnostic, refinement, and final stages.
- Apply the balanced ridge policy to the shipped `analysis.curvature.ridge` / `ridge_mode` schema.
- Record shift source, target ridge, applied value, and override digest in `curvature_shift_records.csv`.
- Require final native endpoint acceptance and final-versus-diagnostic agreement.
- Persist direct spectral-gain curves over every declared truncation threshold.
- Treat missing tau curves as unavailable rather than silently falling back to a weaker sign rule.

### Partial-trace geometry

- Persist left/right partial traces, spectra, and aligned/reversed intervention factors.
- Add relative matrix-change, clustered-subspace projector, intervention-factor, and negative-mass checks.
- Compare rotations inside near-degenerate clusters as subspaces rather than individual eigenvectors.
- Gate aligned/reversed controls on the complete partial-trace geometry check.

### Canonical reporting

- Add `canonical_block_metrics.csv`, `canonical_interventions.csv`, `canonical_alpha_sweep.csv`, `canonical_damping_sweep.csv`, and `canonical_spectral_gain_curve.csv`.
- Make balanced reliability labels authoritative for seed consensus.
- Auto-resolve balanced run roots during aggregation.
- Refuse mixed balanced/legacy aggregation by default.
- Require `reliability_mode=balanced_canonical` for scientific paper export.
- Separate `pipeline_status`, `scientific_status`, and `primary_inference_available`.

### Controls and theory verification

- Split damping controls into `joint` and `shampoo_only` modes.
- Add an independent weighted-Chebyshev certificate for nodes, positive weights, constrained optimum, scaled-Chebyshev equality, and the `C_T(K)/3` three-group lower envelope.
- Rename the dense synthetic output to clarify that it verifies the flat-Kronecker conditioning mechanism; retain `theory_results.csv` for compatibility.

### Reproducibility and release

- Require equal-sized covariance bootstrap groups in scientific configs.
- Repair the balanced clean-shell smoke workflow.
- Update the package version, tests, documentation, release manifest, and checksum manifest to v1.2.0.

## v1.1.0 — Balanced numerical reliability

- Added adaptive endpoint and partial-trace budgets.
- Added optional lower-tail multistart refinement.
- Separated endpoint convergence from bootstrap uncertainty.
- Added balanced reliability certificates and tau-state classification.

## v1.0.0 — Focused ICLR 2027 experiment package

- Added dense verification of Theorem 1 and the flat-Kronecker conditioning construction.
- Added frozen causal-LM block analysis with centered/uncentered moments, GGN curvature, Adam/Shampoo operators, assignment controls, utilization-exponent controls, damping controls, grouped bootstrap, figures, provenance, and guarded export.
- Removed evolving optimizer-training claims and unrelated backends from the focused artifact.
