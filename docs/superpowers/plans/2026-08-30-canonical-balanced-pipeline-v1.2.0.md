# Canonical Balanced Reliability Pipeline v1.2.0 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deliver a v1.2.0 source archive whose balanced reliability results, aggregation, and paper export form one fail-closed scientific pipeline.

**Architecture:** Preserve the frozen analysis core and add narrowly scoped persistence and reliability interfaces: fixed shift overrides, tau-curve output, partial-trace artifacts, canonical balanced tables, and a balanced-aware aggregator/exporter. Add an independent weighted-Chebyshev verification module rather than embedding theorem code in the checkpoint runner.

**Tech Stack:** Python 3.10+, PyTorch, NumPy, pandas, SciPy, PyYAML, pytest, Bash.

**Spec:** `docs/superpowers/specs/2026-08-30-canonical-balanced-pipeline-design.md`

## Global Constraints

- Package version is exactly `1.2.0`.
- Existing frozen debug and screening workflows remain usable.
- Scientific paper export accepts only balanced canonical aggregates.
- No new runtime dependency is introduced.
- New behavior is developed test-first.
- Existing compatibility columns remain, but canonical columns are authoritative.

---

### Task 1: Release contract and clean-shell baseline

**Files:**
- Modify: `tests/test_release_focused.py`
- Modify: `tests/test_focused_configs.py`
- Modify: `reproduce_balanced_smoke.sh`
- Modify: `visible_curvature/__init__.py`
- Modify: `pyproject.toml`

**Interfaces:**
- Produces package version `1.2.0` and a balanced smoke script runnable from a clean shell.

- [ ] **Step 1: Update release tests to require v1.2.0 and the balanced public surface**

```python
assert visible_curvature.__version__ == "1.2.0"
assert project["version"] == "1.2.0"
assert {path.name for path in (ROOT / "configs").glob("*.yaml")} == {
    "synthetic_theory.yaml", "smoke.yaml", "hf_opt125m_screening.yaml",
    "hf_opt125m_confirmatory.yaml", "balanced_reliability_smoke.yaml",
    "hf_opt125m_balanced_reliability.yaml",
}
```

- [ ] **Step 2: Run release tests and verify failure on the old version**

Run: `python -m pytest tests/test_release_focused.py tests/test_focused_configs.py -q`

Expected: FAIL because the package still reports 1.1.0.

- [ ] **Step 3: Set version to 1.2.0 and export the repository root in the balanced smoke script**

```bash
export PYTHONPATH="${PYTHONPATH:+${PYTHONPATH}:}${ROOT}"
```

- [ ] **Step 4: Run the release tests**

Run: `python -m pytest tests/test_release_focused.py tests/test_focused_configs.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml visible_curvature/__init__.py tests/test_release_focused.py tests/test_focused_configs.py reproduce_balanced_smoke.sh
git commit -m "chore: establish v1.2.0 release contract"
```

### Task 2: Fixed curvature-shift overrides and real ridge policy

**Files:**
- Modify: `visible_curvature/curvature.py`
- Modify: `visible_curvature/analysis_runner.py`
- Modify: `visible_curvature/config.py`
- Modify: `visible_curvature/reliability_balanced.py`
- Test: `tests/test_curvature_shift_overrides.py`
- Test: `tests/test_reliability_balanced.py`

**Interfaces:**
- Produces `stabilize_curvature(..., shift_override: float | None = None, shift_source: str = "estimated")`.
- Produces per-run `curvature_shift_records.csv` and accepts `analysis.curvature.shift_overrides_path`.

- [ ] **Step 1: Write tests for an exact shift override and the shipped ridge schema**

```python
def test_shift_override_bypasses_stage_dependent_estimation():
    result = stabilize_curvature(op, psd_mode="shift", ridge=1e-5,
        lanczos_steps=4, lanczos_starts=1, seed=0, shift_override=0.75)
    assert result.shift == 0.75
    assert result.shift_source == "override"

def test_balanced_policy_sets_real_relative_max_ridge_schema():
    cfg, modified = prepare_core_config(base, stage=stage, output_dir=tmp_path,
        diagnostic=True, policy={"reliability": {"fixed_relative_ridge": 2e-5}})
    assert cfg["analysis"]["curvature"]["ridge"] == 2e-5
    assert cfg["analysis"]["curvature"]["ridge_mode"] == "relative_max"
```

- [ ] **Step 2: Run the tests and verify they fail for missing interfaces**

Run: `python -m pytest tests/test_curvature_shift_overrides.py tests/test_reliability_balanced.py -q`

- [ ] **Step 3: Implement override loading, recording, and strict ridge-field enforcement**

The core loads a JSON mapping from block name to shift, applies it unchanged, and records its SHA-256. `prepare_core_config` sets the existing `ridge` field and `ridge_mode` or raises `ReliabilityError`.

- [ ] **Step 4: Add a calibration stage to the balanced orchestrator**

Run one diagnostic calibration config, extract block shifts to `curvature_shift_overrides.json`, and inject its path into all adaptive and final configs.

- [ ] **Step 5: Run focused tests**

Run: `python -m pytest tests/test_curvature_shift_overrides.py tests/test_curvature_stabilization_v3.py tests/test_reliability_balanced.py -q`

- [ ] **Step 6: Commit**

```bash
git add visible_curvature/curvature.py visible_curvature/analysis_runner.py visible_curvature/config.py visible_curvature/reliability_balanced.py tests/test_curvature_shift_overrides.py tests/test_reliability_balanced.py
git commit -m "feat: freeze curvature stabilization across reliability stages"
```

### Task 3: Spectral gain curves and bootstrap grouping contract

**Files:**
- Modify: `visible_curvature/analysis_runner.py`
- Modify: `visible_curvature/config.py`
- Test: `tests/test_spectral_gain_curve.py`
- Test: `tests/test_covariance.py`

**Interfaces:**
- Produces `spectral_gain_curve.csv` with one row per comparison and tau.
- Scientific configs reject covariance batch counts not divisible by bootstrap group size.

- [ ] **Step 1: Write failing tests for tau rows and incomplete scientific groups**

```python
def test_tau_rows_preserve_comparison_identity():
    rows = build_spectral_gain_rows(metadata, adam_spec, shampoo_spec,
        taus=[1e-3, 1e-4], covariance_moment="centered",
        assignment="observed", alpha=0.25, sweep_mode="primary")
    assert [row["tau"] for row in rows] == [1e-3, 1e-4]
    assert all(row["assignment"] == "observed" for row in rows)

def test_scientific_grouped_bootstrap_requires_equal_group_sizes():
    with pytest.raises(ValueError, match="divisible"):
        validate_config(scientific_config(num_batches=10, group_size=4), "frozen")
```

- [ ] **Step 2: Verify failure**

Run: `python -m pytest tests/test_spectral_gain_curve.py tests/test_covariance.py -q`

- [ ] **Step 3: Implement tau-row generation at the comparison boundary**

Every primary and control comparison appends rows with `K_adam`, `K_shampoo`, `delta_g`, tau, saturation, assignment, alpha, and damping metadata.

- [ ] **Step 4: Write `spectral_gain_curve.csv` from `run_frozen_analysis`**

Add it to the run manifest required outputs.

- [ ] **Step 5: Enforce the scientific grouped-bootstrap divisibility rule**

Debug runs may retain an incomplete final group; scientific runs fail before model loading.

- [ ] **Step 6: Run focused tests and commit**

```bash
python -m pytest tests/test_spectral_gain_curve.py tests/test_covariance.py tests/test_focused_analysis.py -q
git add visible_curvature/analysis_runner.py visible_curvature/config.py tests/test_spectral_gain_curve.py tests/test_covariance.py
git commit -m "feat: persist spectral gain curves and bootstrap contract"
```

### Task 4: Partial-trace artifact and geometry stability checks

**Files:**
- Create: `visible_curvature/partial_trace_stability.py`
- Modify: `visible_curvature/analysis_runner.py`
- Modify: `visible_curvature/reliability_balanced.py`
- Test: `tests/test_partial_trace_stability.py`

**Interfaces:**
- Produces `save_partial_trace_artifact(path, left, right, ...)`.
- Produces `compare_partial_trace_artifacts(previous, current, thresholds) -> dict[str, float | bool | str]`.

- [ ] **Step 1: Write failing tests for rotation-sensitive and cluster-aware stability**

```python
def test_rotated_well_separated_eigenspaces_fail_subspace_check(tmp_path):
    previous = artifact(diag([1.0, 4.0]))
    current = artifact(rotation(pi / 4) @ diag([1.0, 4.0]) @ rotation(pi / 4).T)
    result = compare_partial_trace_artifacts(previous, current, thresholds())
    assert result["subspace_stable"] is False

def test_rotation_inside_degenerate_cluster_is_accepted(tmp_path):
    previous = artifact(diag([2.0, 2.0, 5.0]))
    current = artifact(block_rotation @ diag([2.0, 2.0, 5.0]) @ block_rotation.T)
    result = compare_partial_trace_artifacts(previous, current, thresholds(cluster_gap=1e-6))
    assert result["subspace_stable"] is True
```

- [ ] **Step 2: Verify failure because the module does not exist**

Run: `python -m pytest tests/test_partial_trace_stability.py -q`

- [ ] **Step 3: Implement artifact persistence and comparisons**

Use relative Frobenius change, negative mass, spectral clusters defined by adjacent relative gaps, projector distance, and normalized intervention-factor change.

- [ ] **Step 4: Persist artifacts from each block analysis**

Use a filesystem-safe SHA-256-derived block identifier and include an index JSON mapping identifiers to names.

- [ ] **Step 5: Extend stage summaries and certificates**

The balanced check table must expose `partial_trace_psd_checks_passed`, `partial_trace_matrix_stable`, `partial_trace_subspace_stable`, `intervention_factors_stable`, and the aggregate `partial_trace_checks_passed`.

- [ ] **Step 6: Run tests and commit**

```bash
python -m pytest tests/test_partial_trace_stability.py tests/test_partial_trace.py tests/test_reliability_balanced.py -q
git add visible_curvature/partial_trace_stability.py visible_curvature/analysis_runner.py visible_curvature/reliability_balanced.py tests/test_partial_trace_stability.py tests/test_reliability_balanced.py
git commit -m "feat: validate partial-trace geometry across probe budgets"
```

### Task 5: Final agreement gate and canonical tables

**Files:**
- Modify: `visible_curvature/reliability_balanced.py`
- Modify: `scripts/validate_balanced_run.py`
- Modify: `scripts/summarize_balanced_results.py`
- Test: `tests/test_balanced_canonical_outputs.py`

**Interfaces:**
- Produces canonical tables and `scientific_status.json`.
- Uses final native endpoint checks and final-versus-selected-stage agreement.

- [ ] **Step 1: Write failing tests for final override and canonical filtering**

```python
def test_final_endpoint_failure_overrides_adaptive_acceptance(...):
    final["endpoint_numerically_reliable"] = False
    annotate_final_outputs(...)
    row = pd.read_csv(final_dir / "canonical_block_metrics.csv").iloc[0]
    assert not row["balanced_primary_reliable"]
    assert "final_endpoint" in row["balanced_reliability_reasons"]

def test_summary_prints_only_primary_rows(...):
    assert "uncentered" not in rendered_primary_section
```

- [ ] **Step 2: Verify failure**

Run: `python -m pytest tests/test_balanced_canonical_outputs.py -q`

- [ ] **Step 3: Implement final-versus-diagnostic comparisons and fail-closed tau use**

Missing tau curves yield `tau_refinement_unavailable`; no native fallback is used for scientific promotion.

- [ ] **Step 4: Write canonical control tables with basis-aware reliability**

Aligned/reversed controls require the full partial-trace geometry check; observed controls require endpoint acceptance.

- [ ] **Step 5: Write machine-readable pipeline/scientific status and update scripts**

Validator prints separate `pipeline_status`, `scientific_status`, and `primary_inference_available` fields.

- [ ] **Step 6: Run tests and commit**

```bash
python -m pytest tests/test_balanced_canonical_outputs.py tests/test_reliability_balanced.py -q
git add visible_curvature/reliability_balanced.py scripts/validate_balanced_run.py scripts/summarize_balanced_results.py tests/test_balanced_canonical_outputs.py
git commit -m "feat: promote only canonical balanced scientific results"
```

### Task 6: Balanced-aware aggregation and paper export

**Files:**
- Modify: `visible_curvature/aggregate.py`
- Modify: `visible_curvature/paper_export.py`
- Modify: `visible_curvature/figures.py`
- Test: `tests/test_balanced_reporting.py`
- Modify: `tests/test_focused_reporting.py`

**Interfaces:**
- `aggregate_frozen_runs` auto-resolves balanced roots and selects canonical tables.
- Scientific paper export requires `reliability_mode == "balanced_canonical"`.

- [ ] **Step 1: Write failing tests for balanced labels and export rejection**

```python
def test_aggregate_uses_balanced_ordering_from_canonical_table(...):
    output = aggregate_frozen_runs(balanced_roots, aggregate, minimum_seed_count=3)
    assert json.loads((output / "aggregate_manifest.json").read_text())["canonical_tables_used"]
    assert pd.read_csv(output / "paired_seed_summary.csv").iloc[0]["reliable_ordering"] == "negative"

def test_scientific_export_rejects_legacy_aggregate(...):
    with pytest.raises(ValueError, match="balanced_canonical"):
        export_paper_assets(legacy_aggregate, paper)
```

- [ ] **Step 2: Verify failure**

Run: `python -m pytest tests/test_balanced_reporting.py -q`

- [ ] **Step 3: Implement source resolution and canonical table mapping**

Balanced root paths resolve to `root/final`; canonical inputs are copied to the aggregate's public table names while retaining canonical copies.

- [ ] **Step 4: Use `balanced_reliable_ordering` for seed consensus**

Record `all_sources_balanced`, `canonical_tables_used`, and `all_primary_rows_numerically_accepted` in the manifest.

- [ ] **Step 5: Enforce the paper-export gate and adapt figures**

Debug exports remain possible with watermarking.

- [ ] **Step 6: Run tests and commit**

```bash
python -m pytest tests/test_balanced_reporting.py tests/test_focused_reporting.py -q
git add visible_curvature/aggregate.py visible_curvature/paper_export.py visible_curvature/figures.py tests/test_balanced_reporting.py tests/test_focused_reporting.py
git commit -m "feat: connect balanced results to aggregation and paper export"
```

### Task 7: Mechanistic and joint damping controls

**Files:**
- Modify: `visible_curvature/analysis_runner.py`
- Modify: `visible_curvature/config.py`
- Modify: `configs/hf_opt125m_confirmatory.yaml`
- Modify: `configs/smoke.yaml`
- Test: `tests/test_damping_sweep_modes.py`

**Interfaces:**
- `analysis.damping_sweep.modes` accepts `joint` and `shampoo_only`.
- Damping rows expose `sweep_mode`, `adam_damping_coefficient`, and `shampoo_damping_coefficient`.

- [ ] **Step 1: Write failing tests for fixed Adam behavior in shampoo-only mode**

```python
def test_shampoo_only_sweep_reuses_primary_adam_spectrum():
    rows = run_control_fixture(modes=["shampoo_only"], coefficients=[0.0, 1.0])
    assert rows[0]["K_adam"] == rows[1]["K_adam"]
    assert {row["sweep_mode"] for row in rows} == {"shampoo_only"}
```

- [ ] **Step 2: Verify failure**

Run: `python -m pytest tests/test_damping_sweep_modes.py -q`

- [ ] **Step 3: Implement separate coefficient overrides and reuse the primary Adam spectrum**

- [ ] **Step 4: Update config validation and shipped configs**

- [ ] **Step 5: Run tests and commit**

```bash
python -m pytest tests/test_damping_sweep_modes.py tests/test_config_focused.py -q
git add visible_curvature/analysis_runner.py visible_curvature/config.py configs/hf_opt125m_confirmatory.yaml configs/smoke.yaml tests/test_damping_sweep_modes.py
git commit -m "feat: separate mechanistic and joint damping sweeps"
```

### Task 8: Weighted-Chebyshev minimax verification

**Files:**
- Create: `visible_curvature/chebyshev.py`
- Modify: `visible_curvature/synthetic_theory.py`
- Modify: `configs/synthetic_theory.yaml`
- Test: `tests/test_chebyshev_certificate.py`
- Modify: `tests/test_synthetic_theory.py`

**Interfaces:**
- Produces `weighted_chebyshev_certificate(K: float, T: int) -> dict[str, object]`.
- Produces `chebyshev_certificates.csv` and `flat_kronecker_conditioning_results.csv`.

- [ ] **Step 1: Write failing exact-certificate tests**

```python
@pytest.mark.parametrize("K,T", [(4.0, 2), (16.0, 3), (25.0, 4)])
def test_weighted_chebyshev_optimum_matches_closed_form(K, T):
    certificate = weighted_chebyshev_certificate(K, T)
    assert np.all(np.asarray(certificate["weights"]) > 0)
    assert np.isclose(np.sum(certificate["weights"]), 1.0)
    assert np.isclose(certificate["quadratic_optimum"], certificate["C_T"], rtol=1e-10)
    assert np.isclose(certificate["three_group_lower_bound"], certificate["C_T"] / 3.0)
```

- [ ] **Step 2: Verify failure because the module is absent**

Run: `python -m pytest tests/test_chebyshev_certificate.py -q`

- [ ] **Step 3: Implement nodes, stable Lagrange values at zero, weights, Chebyshev values, and constrained optimum**

Compute the constrained optimum using the closed-form weighted least-squares dual and independently evaluate the scaled Chebyshev polynomial at the nodes.

- [ ] **Step 4: Integrate the certificate sweep into the synthetic runner**

Retain `theory_results.csv` as a compatibility concatenation, but write the two scientifically distinct tables separately.

- [ ] **Step 5: Run tests and commit**

```bash
python -m pytest tests/test_chebyshev_certificate.py tests/test_synthetic_theory.py -q
git add visible_curvature/chebyshev.py visible_curvature/synthetic_theory.py configs/synthetic_theory.yaml tests/test_chebyshev_certificate.py tests/test_synthetic_theory.py
git commit -m "feat: verify weighted Chebyshev minimax certificates"
```

### Task 9: Documentation, validators, full verification, and archive

**Files:**
- Modify: `README.md`
- Modify: `CHANGELOG.md`
- Modify: `PACKAGE_MANIFEST.md`
- Modify: `QUICKSTART_KO.md`
- Modify: `EXPERIMENT_PROTOCOL_KO.md`
- Modify: `AUTHOR_RUN_CHECKLIST_KO.md`
- Modify: `RELEASE_NOTES_KO.md`
- Modify: `reproduce_smoke.sh`
- Modify: `SHA256SUMS.txt`

**Interfaces:**
- Produces the final `Visible_Curvature_Experiment_Code_Canonical_Balanced_v1.2.0.zip`.

- [ ] **Step 1: Update documentation to describe canonical outputs, GGN terminology, fixed shifts, numerical acceptance, and both damping sweeps**

- [ ] **Step 2: Run the complete unit suite**

Run: `python -m pytest -q`

Expected: all tests pass.

- [ ] **Step 3: Run compile and smoke verification**

```bash
python -m compileall -q visible_curvature scripts tests
VC_SKIP_TESTS=1 bash reproduce_smoke.sh
bash reproduce_balanced_smoke.sh
```

Expected: both workflows complete; balanced validation may report scientific status `inconclusive` for the tiny smoke run but pipeline status `complete`.

- [ ] **Step 4: Regenerate and verify SHA-256 manifest**

Exclude `SHA256SUMS.txt`, transient outputs, caches, and VCS metadata from the manifest.

```bash
sha256sum -c SHA256SUMS.txt
```

- [ ] **Step 5: Build the final ZIP and verify its contents in a fresh extraction**

```bash
zip -qr Visible_Curvature_Experiment_Code_Canonical_Balanced_v1.2.0.zip Visible_Curvature_Experiment_Code_Canonical_Balanced_v1.2.0
```

Run unit tests and checksum verification from the extracted archive.

- [ ] **Step 6: Commit**

```bash
git add README.md CHANGELOG.md PACKAGE_MANIFEST.md QUICKSTART_KO.md EXPERIMENT_PROTOCOL_KO.md AUTHOR_RUN_CHECKLIST_KO.md RELEASE_NOTES_KO.md reproduce_smoke.sh SHA256SUMS.txt
git commit -m "docs: finalize canonical balanced v1.2.0 release"
```
