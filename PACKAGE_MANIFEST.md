# Package Manifest — Canonical Balanced v1.2.1

## Public runners

- `scripts/run_synthetic_theory.py` — dense conditioning checks and weighted-Chebyshev certificates
- `scripts/run_frozen_analysis.py` — one frozen causal-LM checkpoint analysis
- `scripts/run_balanced_reliability.py` — adaptive canonical reliability orchestration
- `scripts/make_balanced_policies.py` — seed-specific policy generation
- `scripts/validate_run.py` — core run/aggregate structural validation
- `scripts/validate_balanced_run.py` — balanced pipeline and optional scientific-status validation
- `scripts/summarize_balanced_results.py` — compact balanced status report
- `scripts/aggregate_runs.py` — balanced-aware seed aggregation and optional figures
- `scripts/make_figures.py` — retained result figures
- `scripts/export_paper_assets.py` — fail-closed LaTeX/table export

## Shipped configurations

- `configs/synthetic_theory.yaml`
- `configs/smoke.yaml`
- `configs/balanced_reliability_smoke.yaml`
- `configs/hf_opt125m_screening.yaml`
- `configs/hf_opt125m_confirmatory.yaml`
- `configs/hf_opt125m_balanced_reliability.yaml`

## Core modules

- `adapters.py` — tiny and Hugging Face causal-LM adapters
- `data.py` — synthetic and immutable Hugging Face token streams
- `block_registry.py` — Transformer matrix-block discovery and fused-QKV slicing
- `covariance.py` — centered Welford statistics and grouped bootstrap state
- `curvature.py` — GGN operators, partial traces, and fixed shift application
- `preconditioners.py` — Adam-form and exponent-generalized Shampoo-form operators
- `linear_algebra.py` — paired Lanczos, endpoint diagnostics, and matrix spectral functions
- `diagnostics.py` — elasticity, overlap, commutator, and signed predictors
- `interventions.py` — observed/aligned/reversed factor assignment
- `partial_trace_stability.py` — matrix, clustered-subspace, and factor stability checks
- `run_lock.py` — process-lifetime exclusive lock for each balanced output root
- `runtime_bootstrap.py` — applies GPU and CPU-thread policy settings before numerical imports
- `analysis_runner.py` — frozen checkpoint orchestration and artifact persistence
- `reliability_balanced.py` — nested budgets, canonical promotion, and scientific status
- `chebyshev.py` — weighted-Chebyshev minimax certificate
- `synthetic_theory.py` — dense conditioning and certificate sweeps
- `aggregate.py`, `figures.py`, `paper_export.py` — balanced-canonical reporting pipeline

## Synthetic generated files

- `theorem1_conditioning_results.csv`
- `flat_kronecker_conditioning_results.csv`
- `chebyshev_certificates.csv`
- `theory_results.csv`
- `theory_summary.json`
- `resolved_config.yaml`

## Core frozen-run generated files

- `block_metrics.csv`
- `bootstrap_metrics.csv`
- `interventions.csv`
- `alpha_sweep.csv`
- `damping_sweep.csv`
- `spectral_gain_curve.csv`
- `curvature_shift_records.csv`
- `partial_trace_artifacts/`
- `block_failures.csv`
- `run_manifest.json`
- `summary.json`
- `resolved_config.yaml`

## Balanced generated files

At the run root:

- `endpoint_convergence.csv`
- `partial_trace_convergence.csv`
- `balanced_reliability_certificates.csv`
- `curvature_shift_overrides.json`
- `balanced_reliability_summary.json`
- `diagnostic_stages/`, `generated_configs/`, `logs/`, `final/`, `COMPLETED`

In `final/`:

- `canonical_block_metrics.csv`
- `canonical_interventions.csv`
- `canonical_alpha_sweep.csv`
- `canonical_damping_sweep.csv`
- `canonical_spectral_gain_curve.csv`
- balanced-annotated compatibility tables
- `scientific_status.json`
- `balanced_reliability_summary.json`
- all core final-run artifacts

## Documentation and verification

- `README.md`
- `QUICKSTART_KO.md`
- `EXPERIMENT_PROTOCOL_KO.md`
- `AUTHOR_RUN_CHECKLIST_KO.md`
- `RELEASE_NOTES_KO.md`
- `docs/superpowers/specs/2026-08-30-canonical-balanced-pipeline-design.md`
- `docs/superpowers/plans/2026-08-30-canonical-balanced-pipeline-v1.2.0.md`
- `reproduce_smoke.sh`
- `reproduce_balanced_smoke.sh`
- `tests/`
- `SHA256SUMS.txt`
