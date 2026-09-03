# OVC Submission Experiment Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deliver v1.2.0 with strict full-space conditioning, correct H1–H6 evaluation, same-error dynamics, safe outputs, and a model/checkpoint streaming execution path.

**Architecture:** Consolidate correctness policy in focused spectral/preconditioner/reporting utilities, then route both legacy smoke and hardened model-scale paths through those utilities. Preserve existing public APIs where possible, but make scientific ambiguity explicit through censor records and policy fields.

**Tech Stack:** Python 3.11+, PyTorch, NumPy, SciPy, pandas, PyYAML, pytest, setuptools.

**Spec:** `docs/superpowers/specs/2026-09-03-ovc-submission-hardening-design.md`

## Global Constraints

- Default spectral/subspace policy is `strict_spd`.
- Censored measurements never produce finite gain or hypothesis support.
- No full N-by-d per-example gradient tensor in the primary model-scale path.
- Existing smoke commands remain backward compatible.
- All new behavior is introduced through failing regression tests first.

---

### Task 1: Strict spectral and inverse-root semantics

**Files:**
- Modify: `src/ovc_experiments/spectral.py`
- Modify: `src/ovc_experiments/hardened_spectral.py`
- Modify: `src/ovc_experiments/spectral_power.py`
- Modify: `src/ovc_experiments/preconditioners.py`
- Modify: `src/ovc_experiments/geometry.py`
- Test: `tests/test_strict_conditioning.py`

**Interfaces:**
- Produces: `estimate_condition(..., subspace_policy="strict_spd")`
- Produces: strict or explicit-active negative powers with scale-aware diagnostics.

- [ ] Write failing tests for singular effective operators, explicit positive-active mode, and scale invariance.
- [ ] Run the focused tests and confirm the expected failures.
- [ ] Implement strict exact/Lanczos censoring and remove the legacy/hardened wrapper split.
- [ ] Route geometry through the selected policy and preserve censor reasons.
- [ ] Run focused and existing spectral tests.

### Task 2: Factor bounds and separately normalized damping

**Files:**
- Create: `src/ovc_experiments/spectrum_utils.py`
- Modify: `src/ovc_experiments/interventions.py`
- Modify: `src/ovc_experiments/runners.py`
- Modify: `src/ovc_experiments/hardened_runner.py`
- Test: `tests/test_factor_damping.py`

**Interfaces:**
- Produces: `positive_spectrum_summary(matrix, relative_threshold, absolute_threshold)`.
- Produces: left/right damping and normalized damping metadata.

- [ ] Write failing tests for scale-aware factor bounds, no-positive-factor censoring, and unequal left/right factor scales.
- [ ] Run focused tests and confirm failures.
- [ ] Implement the spectrum summary and use it in all Shampoo construction/sweeps.
- [ ] Convert construction failures to censored rows.
- [ ] Run focused and intervention/runner tests.

### Task 3: Canonical H1–H6 evaluator

**Files:**
- Modify: `src/ovc_experiments/hardened_reporting.py`
- Modify: `src/ovc_experiments/reporting.py`
- Modify: `src/ovc_experiments/runners.py`
- Test: `tests/test_hypothesis_regressions.py`

**Interfaces:**
- Produces: `evaluate_hypotheses(...) -> DataFrame`.
- Produces: `summarize_hypotheses(...) -> dict` for legacy callers.

- [ ] Write failing tests using the actual `delta_G_0.25`, `gain`, `condition_number`, and `rho_over_min` schema.
- [ ] Confirm failures for negative H2, unsigned H3, one-sided H5, and cross-checkpoint H6 mixing.
- [ ] Implement schema canonicalization, target-specific censor filtering, and exact hypothesis rules.
- [ ] Add gain/delta/censor metadata to intervention rows.
- [ ] Run focused, reporting, smoke, and aggregate tests.

### Task 4: Same-original-error dynamics

**Files:**
- Modify: `src/ovc_experiments/runners.py`
- Modify: `src/ovc_experiments/same_error_dynamics.py`
- Test: `tests/test_runner_same_error_dynamics.py`

**Interfaces:**
- Produces paired transformed initial state `z0=P^{-1/2}e0` and auditable fingerprints.

- [ ] Write a failing runner-level test proving the old path uses different original errors.
- [ ] Confirm failure.
- [ ] Transform a common original error for each SPD preconditioner and skip censored operators.
- [ ] Record initialization fingerprint and initial objective.
- [ ] Run focused and dynamics tests.

### Task 5: Strict configuration and safe append-only outputs

**Files:**
- Modify: `src/ovc_experiments/config.py`
- Modify: `src/ovc_experiments/io.py`
- Modify: `src/ovc_experiments/hardened_spectral.py`
- Modify: `src/ovc_experiments/hardened_runner.py`
- Test: `tests/test_config_output_safety.py`

**Interfaces:**
- Produces strict nested YAML loading.
- Produces atomic CSV row append/schema merge and strict JSONL diagnostics.

- [ ] Write failing tests for unknown nested keys, multiple block appends, and non-finite JSONL values.
- [ ] Confirm failures.
- [ ] Implement strict dataclass construction, validation, atomic table append, and diagnostics writer/context.
- [ ] Run focused and I/O tests.

### Task 6: Streaming checkpoint/block orchestration

**Files:**
- Modify: `src/ovc_experiments/functional.py`
- Create: `src/ovc_experiments/streaming_runner.py`
- Modify: `src/ovc_experiments/cli.py`
- Modify: `src/ovc_experiments/config.py`
- Test: `tests/test_streaming_checkpoint_runner.py`

**Interfaces:**
- Produces: `iter_per_example_gradients(...)`.
- Produces: `run_streaming_geometry(config, ...)` and CLI `streaming-geometry`.

- [ ] Write failing tests for replayability, one-block-at-a-time output, and absence of raw gradient retention.
- [ ] Confirm failures.
- [ ] Implement the gradient iterator and streaming orchestration.
- [ ] Add the CLI command and output manifest.
- [ ] Run focused and smoke tests.

### Task 7: Versioning, documentation, and full verification

**Files:**
- Modify: `pyproject.toml`
- Modify: `src/ovc_experiments/__init__.py`
- Modify: `README.md`
- Create: `CHANGELOG_v1.2.0.md`
- Create: `VERIFICATION_REPORT_v1.2.0.md`

**Interfaces:**
- Produces the final v1.2.0 source archive.

- [ ] Update version and runbook documentation.
- [ ] Run `python -m compileall -q src tests`.
- [ ] Run `PYTHONPATH=src pytest -q`.
- [ ] Run `python scripts/validate_hardened.py`.
- [ ] Run `bash scripts/run_smoke_suite.sh` and `bash scripts/run_validation_suite.sh`.
- [ ] Build sdist and wheel and inspect archive contents.
- [ ] Create the final source ZIP and checksums.
