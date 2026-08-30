# Package Manifest — Canonical Balanced v1.3.0

## Public runners

- `scripts/run_synthetic_theory.py` — analytic/dense conditioning, Chebyshev, and integrated witness checks
- `scripts/run_frozen_analysis.py` — one frozen causal-LM checkpoint analysis
- `scripts/run_balanced_reliability.py` — adaptive canonical reliability orchestration
- `scripts/make_balanced_policies.py` — seed-specific policy generation
- `scripts/validate_run.py` — core run/aggregate structural validation
- `scripts/validate_balanced_run.py` — balanced-pipeline validation
- `scripts/summarize_balanced_results.py` — compact balanced status report
- `scripts/aggregate_runs.py` — balanced-aware seed aggregation
- `scripts/make_figures.py` — retained empirical figures
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
- `block_registry.py` — matrix-block discovery and fused-QKV slicing
- `covariance.py` — centered Welford/Chan statistics and grouped bootstrap state
- `curvature.py` — GGN operators, partial traces, and frozen shift application
- `preconditioners.py` — Adam-form and exponent-generalized Shampoo-form operators
- `linear_algebra.py` — common-start Lanczos, endpoint diagnostics, and spectral functions
- `diagnostics.py` — elasticity, commutator, overlap, and decomposed gain proxy
- `interventions.py` — observed/aligned/reversed retained-factor assignment
- `partial_trace_stability.py` — matrix, clustered-subspace, and factor stability checks
- `provenance.py` — source, software, hardware, CUDA, and execution provenance
- `run_lock.py` — exclusive balanced-output-root lock
- `runtime_bootstrap.py` — process environment setup before numerical imports
- `analysis_runner.py` — frozen orchestration, spectrum cache, controls, and artifacts
- `reliability_balanced.py` — nested budgets, metric-compatible promotion, and status
- `chebyshev.py` — weighted-Chebyshev minimax certificate
- `synthetic_theory.py` — component and integrated theorem checks
- `mechanism.py` — eligibility-gated predictor and paired-control summaries
- `aggregate.py`, `figures.py`, `paper_export.py` — canonical reporting pipeline

## Synthetic generated files

- `theorem1_conditioning_results.csv`
- `flat_kronecker_conditioning_results.csv`
- `chebyshev_certificates.csv`
- `integrated_theorem3_witness.csv`
- `theory_results.csv`
- `theory_summary.json`
- `resolved_config.yaml`

## Core frozen-run generated files

- `block_metrics.csv`
- `bootstrap_metrics.csv`
- `interventions.csv`
- `alpha_sweep.csv`
- `damping_sweep.csv`
- `ridge_sweep.csv`
- `spectral_gain_curve.csv`
- `curvature_shift_records.csv`
- `partial_trace_artifacts/`
- `runtime_provenance.json`
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

## Aggregate generated files

- `paired_seed_summary.csv`
- `elasticity_prediction_rows.csv`
- `elasticity_prediction_summary.csv`
- `paired_control_contrasts.csv`
- `aggregate_summary.csv`
- `aggregate_manifest.json`
- combined canonical/core tables

## Documentation and verification

- `README.md`
- `QUICKSTART_KO.md`
- `EXPERIMENT_PROTOCOL_KO.md`
- `AUTHOR_RUN_CHECKLIST_KO.md`
- `RELEASE_NOTES_KO.md`
- `CHANGELOG.md`
- `docs/superpowers/specs/2026-08-30-scientific-alignment-v1.3.0-design.md`
- `docs/superpowers/plans/2026-08-30-scientific-alignment-v1.3.0.md`
- `reproduce_smoke.sh`
- `reproduce_balanced_smoke.sh`
- `tests/`
- `SHA256SUMS.txt`
