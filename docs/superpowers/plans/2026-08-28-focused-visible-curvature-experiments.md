# Focused Visible-Curvature Experiments Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce a compact, submission-ready experiment package containing only exact synthetic theorem checks and frozen Transformer checkpoint tests of signed Adam/Shampoo conditioning.

**Architecture:** Remove online and image subsystems, retain reusable covariance/curvature/Lanczos primitives, and expose two explicit runners: a dense synthetic-theory runner and a focused frozen-checkpoint runner. Generalize Shampoo utilization to configurable factor exponent `alpha`, separate primary point estimates from low-cost controls, and make provenance and paper export fail closed.

**Tech Stack:** Python 3.10+, PyTorch, NumPy, SciPy, pandas, PyYAML, matplotlib, pytest; optional Hugging Face Transformers and Datasets.

**Spec:** `docs/FOCUSED_EXPERIMENT_SPEC_KO.md`

## Global Constraints

- Supported models are only `tiny_causal_lm` and `hf_causal_lm`.
- Supported data backends are only `synthetic_tokens` and `hf_text`.
- Centered covariance and practical `alpha=0.25` observed assignment define the primary endpoint.
- Bootstrap is delta-only and never recomputes elasticity or intervention diagnostics.
- Scientific runs require immutable model, tokenizer, and dataset revisions and content-level data hashes.
- Online, image, ViT, random/scrambled intervention, NCI, and local-trajectory functionality must not remain in public CLI/configuration/docs.

---

### Task 1: Remove unsupported online and image surfaces

**Files:**
- Delete: `visible_curvature/online_optim.py`, `visible_curvature/online_runner.py`, `visible_curvature/provenance.py`, `scripts/run_online.py`, online/image configs and tests.
- Modify: `visible_curvature/adapters.py`, `visible_curvature/data.py`, `visible_curvature/figures.py`, `visible_curvature/paper_export.py`, `scripts/validate_run.py`, `scripts/export_paper_assets.py`, `scripts/run_mechanism_suite.py`, `pyproject.toml`, `requirements.txt`.
- Test: focused backend/config/release tests.

**Interfaces:**
- Produces `load_model_bundle()` accepting only `tiny_causal_lm` and `hf_causal_lm`.
- Produces `build_dataloader_factory()` accepting only `synthetic_tokens` and `hf_text`.

- [ ] Write tests asserting removed model/data/config modes raise errors and online files/configs are absent.
- [ ] Run focused tests and verify they fail against v0.5.0.
- [ ] Delete unsupported modules and simplify adapters/data/CLI/export surfaces.
- [ ] Run focused tests and verify they pass.

### Task 2: Generalize Shampoo exponent and stabilize factor roots

**Files:**
- Modify: `visible_curvature/preconditioners.py`, `visible_curvature/diagnostics.py`, `visible_curvature/linear_algebra.py`.
- Test: `tests/test_preconditioners_focused.py`, `tests/test_alpha_predictor.py`.

**Interfaces:**
- `ShampooFormPreconditioner(..., factor_exponent: float = 0.25)`.
- `predicted_delta_g(..., factor_exponent: float)`.
- Root metadata reports adaptive floor and floored eigenvalue fractions.

- [ ] Write failing tests for `alpha=0.25`, `alpha=0.5`, half-power actions, and predictor scaling.
- [ ] Verify failures.
- [ ] Implement exponent-generalized factor powers, float64 decomposition, and scale-relative floors.
- [ ] Verify targeted and regression tests.

### Task 3: Add exact synthetic theorem runner

**Files:**
- Create: `visible_curvature/synthetic_theory.py`, `scripts/run_synthetic_theory.py`, `configs/synthetic_theory.yaml`.
- Test: `tests/test_synthetic_theory.py`.

**Interfaces:**
- `run_synthetic_theory(cfg) -> Path` writes `theory_results.csv` and `theory_summary.json`.
- Rows include numerical and analytic condition numbers, invariant errors, `alpha`, `rho`, `r`, and ordering.

- [ ] Write failing tests for Theorem 1, Theorem 3 invariants, alpha doubling, and damping attenuation.
- [ ] Verify failures.
- [ ] Implement dense paired constructions and output writer.
- [ ] Verify tests and execute the synthetic config.

### Task 4: Fix data provenance and immutable scientific configuration

**Files:**
- Modify: `visible_curvature/data.py`, `visible_curvature/config.py`, `visible_curvature/analysis_runner.py`, `visible_curvature/aggregate.py`.
- Test: `tests/test_data_provenance_focused.py`, `tests/test_config_focused.py`.

**Interfaces:**
- `data.order_seed` controls source order independently from experiment seed.
- Manifest fields: `source_order_sha256`, `packed_token_stream_sha256`, `selected_chunk_content_sha256`, immutable revisions.

- [ ] Write failing tests showing packed content hashes change with source order while experiment seed does not alter order.
- [ ] Write failing tests requiring immutable revisions in scientific HF runs.
- [ ] Implement content hashing, order-seed separation, and fail-closed validation.
- [ ] Verify tests.

### Task 5: Focus the frozen analysis runner

**Files:**
- Rewrite: `visible_curvature/analysis_runner.py`.
- Modify: `visible_curvature/interventions.py`, `visible_curvature/mechanism.py`, `visible_curvature/reliability.py`.
- Delete: `visible_curvature/frozen.py` and trajectory-specific tests.
- Test: `tests/test_focused_analysis.py`, retained covariance/GGN/Lanczos tests.

**Interfaces:**
- Primary `block_metrics.csv`: centered and uncentered observed `alpha=0.25` rows.
- `interventions.csv`: centered, `alpha=0.25`, observed/aligned/reversed.
- `alpha_sweep.csv`: centered, configured alpha values, observed/aligned/reversed.
- `damping_sweep.csv`: centered, practical alpha, observed/aligned/reversed.
- `bootstrap_metrics.csv`: centered, observed, practical alpha, delta-only.

- [ ] Write failing integration tests for exact output schemas and absence of trajectory/NCI/random modes.
- [ ] Verify failures.
- [ ] Implement focused runner, cached curvature diagnostics, delta-only bootstrap, and distinct numerical/inferential reliability fields.
- [ ] Verify smoke integration and retained numerical tests.

### Task 6: Replace configs with screening and confirmatory tiers

**Files:**
- Create/replace: `configs/smoke.yaml`, `configs/hf_opt125m_screening.yaml`, `configs/hf_opt125m_confirmatory.yaml`.
- Delete: gold, representative, generic online/image configs.
- Test: `tests/test_focused_configs.py`.

**Interfaces:**
- Screening has no bootstrap/interventions/sweeps.
- Confirmatory has 100 delta-only bootstrap reps, three assignments, alpha `[0.25, 0.5]`, and a coarse normalized damping grid.

- [ ] Write failing config-budget tests.
- [ ] Verify failures.
- [ ] Add focused configs with pinned model/tokenizer/dataset commits and bounded budgets.
- [ ] Verify tests.

### Task 7: Focus aggregation, figures, validation, and paper export

**Files:**
- Modify: `visible_curvature/aggregate.py`, `visible_curvature/figures.py`, `visible_curvature/paper_export.py`, `scripts/aggregate_runs.py`, `scripts/export_paper_assets.py`, `scripts/validate_run.py`.
- Test: focused reporting/export tests.

**Interfaces:**
- Figures: block signed gain, assignment intervention, alpha response, damping attenuation.
- Scientific export requires complete manifests, compatible runtime identities, no block failures, and non-synthetic confirmatory inputs.

- [ ] Write failing tests for alpha figure/export and fail-closed manifest/completeness guards.
- [ ] Verify failures.
- [ ] Implement focused aggregation/reporting/export.
- [ ] Verify tests.

### Task 8: Rewrite documentation and one-command verification

**Files:**
- Rewrite: `README.md`, `QUICKSTART_KO.md`, `EXPERIMENT_PROTOCOL_KO.md`, `AUTHOR_RUN_CHECKLIST_KO.md`, `PACKAGE_MANIFEST.md`, `CHANGELOG.md`, `reproduce_smoke.sh`.
- Remove: obsolete validation logs/guides.
- Test: release metadata and shell verification.

**Interfaces:**
- `bash reproduce_smoke.sh` runs compile, tests, synthetic theory, frozen smoke, validation, aggregation, figure generation, scientific-export rejection, and watermarked debug export.

- [ ] Write failing release tests for removed terms/files and exact watermark contract.
- [ ] Verify failures.
- [ ] Rewrite documentation and smoke script around the two retained experiment modes.
- [ ] Run the full verification command.

### Task 9: Package and verify the release

**Files:**
- Modify version metadata and regenerate `SHA256SUMS.txt`.
- Create final ZIP outside the repository.

**Interfaces:**
- Release directory: `Visible_Curvature_Experiment_Code_Focused_v1.0.0`.
- Archive: `Visible_Curvature_Experiment_Code_Focused_v1.0.0.zip`.

- [ ] Run all tests with timings and warnings treated as failures where practical.
- [ ] Run `bash reproduce_smoke.sh` from a clean output directory.
- [ ] Verify no online/image/ViT artifacts remain via recursive search.
- [ ] Build wheel/sdist and inspect archive contents.
- [ ] Regenerate checksums and create the final ZIP.
