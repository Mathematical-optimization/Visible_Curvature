# Scientific Alignment v1.3.0 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Produce a theory-aligned, fail-closed, reproducible v1.3.0 experiment package with corrected predictors and controls, metric-compatible reliability, cached spectra, expanded confirmatory design, richer exports, and integrated synthetic validation.

**Architecture:** Preserve the existing frozen-analysis data flow and extend it at explicit boundaries: diagnostic components in `diagnostics.py`, row construction/caching in `analysis_runner.py`, promotion in `reliability_balanced.py`, cross-seed inference in `aggregate.py`, and paper-facing output in `figures.py`/`paper_export.py`. All behavior changes are test-first and additive where possible.

**Tech Stack:** Python 3.10+, PyTorch, NumPy, SciPy, pandas, matplotlib, pytest, YAML.

**Spec:** `docs/superpowers/specs/2026-08-30-scientific-alignment-v1.3.0-design.md`

## Global Constraints

- Preserve frozen-operator scope and existing CLI entry points.
- Do not identify truncated condition numbers with ordinary residual condition numbers.
- Keep `delta_g_predicted` as a backward-compatible alias for the full commuting–Kronecker proxy.
- Scientific promotion is fail-closed and ordinary-only by default.
- Every production behavior change starts with a failing regression test.
- Network-dependent OPT-125M execution is not required for offline completion.

---

### Task 1: Predictor decomposition and reliability gates

**Files:**
- Modify: `visible_curvature/diagnostics.py`
- Modify: `visible_curvature/analysis_runner.py`
- Modify: `visible_curvature/config.py`
- Test: `tests/test_visible_elasticity_extended.py`
- Test: `tests/test_predictor_reliability_v130.py`

**Interfaces:**
- Produces: `predicted_delta_g_components(...) -> dict[str, float]`
- Preserves: `predicted_delta_g(...) -> float`, now returning the full proxy.

- [x] Add failing tests for baseline mismatch, full proxy, Adam-regression gating, mode/width gating, and floor gating.
- [x] Run the focused tests and confirm expected failures.
- [x] Implement predictor components and additive metric columns.
- [x] Extend reliability configuration and reason codes.
- [x] Run focused tests and the full suite.

### Task 2: Correct control estimands and paired response columns

**Files:**
- Modify: `visible_curvature/analysis_runner.py`
- Modify: `visible_curvature/figures.py`
- Modify: `visible_curvature/paper_export.py`
- Test: `tests/test_control_estimands_v130.py`
- Test: `tests/test_damping_sweep_modes.py`

**Interfaces:**
- Produces control columns: `K_H`, `G_adam`, `G_shampoo`, `control_estimand`, `control_value`, `alpha_delta_from_practical`.

- [x] Add failing tests showing Shampoo-only damping targets `|G_shampoo|`, joint damping targets `|delta_g|`, and alpha uses signed paired changes.
- [x] Implement compatible gain calculation and control-row metadata.
- [x] Update figures and table generation to use the declared estimand.
- [x] Run focused tests and the full suite.

### Task 3: Metric consistency and ordinary-only scientific promotion

**Files:**
- Modify: `visible_curvature/reliability_balanced.py`
- Modify: `visible_curvature/aggregate.py`
- Modify: `visible_curvature/paper_export.py`
- Modify: `configs/hf_opt125m_balanced_reliability.yaml`
- Test: `tests/test_metric_consistency_v130.py`
- Test: `tests/test_reliability_balanced.py`
- Test: `tests/test_balanced_reporting.py`

**Interfaces:**
- Produces: `condition_metric_consensus`, `fallback_tau_consensus`, `metric_consensus_reason`.
- Policy: `reliability.allow_truncated_primary` defaults to `false`.

- [x] Add failing tests for stage metric transitions, seed metric disagreement, tau disagreement, and truncated-primary rejection.
- [x] Implement stage/final/seed compatibility checks and fail-closed labels.
- [x] Extend scientific export eligibility and metadata.
- [x] Run focused tests and the full suite.

### Task 4: Block-local spectrum cache

**Files:**
- Modify: `visible_curvature/analysis_runner.py`
- Test: `tests/test_spectrum_cache_v130.py`

**Interfaces:**
- Produces: `_LanczosSpectrumCache.get_or_compute(key, compute)` and cache diagnostics.

- [x] Add a failing call-count test for reuse across intervention/alpha/Shampoo-only controls.
- [x] Implement a deterministic block-local cache keyed by operator semantics and budget.
- [x] Wire the control loops to reuse spectra without changing rows.
- [x] Run focused tests and the full suite.

### Task 5: Mechanism summaries and paired control contrasts

**Files:**
- Modify: `visible_curvature/mechanism.py`
- Modify: `visible_curvature/aggregate.py`
- Modify: `visible_curvature/paper_export.py`
- Test: `tests/test_mechanism_summary_v130.py`
- Test: `tests/test_aggregate_controls_v130.py`

**Interfaces:**
- Produces files: `elasticity_prediction_rows.csv`, `elasticity_prediction_summary.csv`, `paired_control_contrasts.csv`.

- [x] Add failing tests for eligible-row filtering, sign/balanced accuracy, Spearman summary, and within-block contrasts.
- [x] Implement summary generation and manifest counts.
- [x] Add paper-export summaries without pooled-median substitution for paired effects.
- [x] Run focused tests and the full suite.

### Task 6: Runtime provenance and exact confirmatory selection

**Files:**
- Create: `visible_curvature/provenance.py`
- Modify: `visible_curvature/analysis_runner.py`
- Modify: `visible_curvature/config.py`
- Modify: `configs/hf_opt125m_confirmatory.yaml`
- Modify: `AUTHOR_RUN_CHECKLIST_KO.md`
- Test: `tests/test_runtime_provenance_v130.py`
- Test: `tests/test_exact_block_preregistration_v130.py`

**Interfaces:**
- Produces: `runtime_provenance()` and `source_tree_digest(root)`.
- Config: exact confirmatory block names and optional `controls.block_names` subset.

- [x] Add failing tests for software/hardware provenance and exact-pattern validation.
- [x] Implement provenance capture and manifest embedding.
- [x] Replace two-block regex template with twelve exact blocks and a preregistered control subset.
- [x] Run focused tests and the full suite.

### Task 7: Ridge sensitivity and integrated Theorem-3 validator

**Files:**
- Modify: `visible_curvature/analysis_runner.py`
- Modify: `visible_curvature/synthetic_theory.py`
- Modify: `visible_curvature/config.py`
- Test: `tests/test_ridge_sensitivity_v130.py`
- Test: `tests/test_integrated_theorem3_witness_v130.py`

**Interfaces:**
- Produces optional `ridge_sweep.csv`.
- Produces integrated validator rows with dimension, invariants, group energies, and simultaneous certificate ratios.

- [x] Add failing tests for the optional ridge plan and integrated witness identities.
- [x] Implement the sensitivity control with default disabled status.
- [x] Construct and validate the full three-group budget-dependent witness.
- [x] Run focused tests, synthetic validation, and the full suite.

### Task 8: Documentation, versioning, packaging, and final verification

**Files:**
- Modify: `README.md`
- Modify: `QUICKSTART_KO.md`
- Modify: `EXPERIMENT_PROTOCOL_KO.md`
- Modify: `PACKAGE_MANIFEST.md`
- Modify: `CHANGELOG.md`
- Modify: `RELEASE_NOTES_KO.md`
- Modify: `pyproject.toml`
- Modify: `visible_curvature/__init__.py`
- Regenerate: `SHA256SUMS.txt`

- [x] Update empirical hypotheses, covariance/GGN definition, intervention scope, metric semantics, provenance, and output tables.
- [x] Bump package version to 1.3.0 and regenerate manifest/checksums.
- [x] Run `pytest -q`.
- [x] Run the synthetic theory validator.
- [x] Run focused and balanced smoke workflows.
- [x] Verify scientific export still rejects debug runs and watermarks explicit debug exports.
- [x] Build the final ZIP and verify its checksums after extraction.
