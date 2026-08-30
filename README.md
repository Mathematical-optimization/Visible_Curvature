# Visible-Curvature Experiments — Canonical Balanced v1.2.0

This repository contains the source code for the frozen-operator experiments accompanying the ICLR 2027 submission **“Optimizer-Visible Curvature: Information Limits and Minimax Order Reversals.”**

The package evaluates a deliberately limited mechanism. Given a block curvature operator `H` and a fixed Adam- or Shampoo-form operator `P` constructed once from a reference second moment, it estimates

\[
K(P,H)=\operatorname{cond}\!\left(P^{1/2}HP^{1/2}\right),
\qquad
\Delta G=\log K_{\mathrm{Adam}}-\log K_{\mathrm{Shampoo}}.
\]

- `ΔG > 0`: frozen Shampoo-form residual conditioning is better than frozen Adam-form conditioning.
- `ΔG < 0`: frozen Shampoo-form residual conditioning is worse.
- `ΔG = 0`: the two residual condition metrics coincide at the reported precision.

The checkpoint curvature is an empirical **generalized Gauss–Newton (GGN)** operator for causal-language-model cross-entropy. It is not presented as the full nonconvex Hessian. The package does not establish online optimizer dominance, stochastic-risk dominance, or wall-clock superiority.

## 1. What v1.2.0 contains

The release has three connected workflows.

1. **Synthetic conditioning verification**
   - Theorem 1 aligned/scalar/reversed Adam-form sign reversal.
   - Flat-Kronecker paired invariants and Shampoo-form conditioning reversal.
   - `α=0.25` versus `α=0.5` utilization-strength control.
   - Damping attenuation.

2. **Independent weighted-Chebyshev verification**
   - Extremal nodes `μ_j(K,T)`.
   - Positive Lagrange weights summing to one.
   - Equality of the constrained weighted-polynomial optimum and `C_T(K)`.
   - Equality for the scaled Chebyshev polynomial.
   - The three-group lower-envelope value `C_T(K)/3` used by the full Krylov construction.

3. **Canonical balanced checkpoint pipeline**
   - Centered and uncentered gradient second moments from one gradient stream.
   - Frozen Adam- and Shampoo-form operators.
   - Paired Lanczos endpoint estimation.
   - Fixed curvature stabilization shift across all reliability stages.
   - Spectral-gain curves over the declared truncation grid.
   - Partial-trace matrix, clustered-subspace, and intervention-factor stability checks.
   - Grouped calibrated bootstrap.
   - Observed/aligned/reversed assignment controls.
   - `α` control and separate joint versus Shampoo-only damping sweeps.
   - Canonical tables used by aggregation and scientific paper export.

The original single-run frozen analysis remains available for screening and software debugging. Scientific confirmatory aggregation and export use the balanced canonical path.

## 2. Installation

Python 3.10 or newer is required.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e '.[dev]'
```

For Hugging Face checkpoint experiments:

```bash
pip install -e '.[network,dev]'
```

The core dependencies are NumPy, pandas, SciPy, matplotlib, PyYAML, and PyTorch. No new runtime dependency was introduced in v1.2.0.

## 3. Software verification

### 3.1 Focused smoke workflow

```bash
bash reproduce_smoke.sh
```

This command:

1. compiles the package and runs the full test suite;
2. runs synthetic conditioning and weighted-Chebyshev checks;
3. runs the tiny causal-LM frozen analysis;
4. validates and aggregates the debug output;
5. creates the retained figures;
6. verifies that a legacy/debug aggregate is rejected by scientific export;
7. creates a visibly watermarked debug export.

Set `VC_SKIP_TESTS=1` only when the full tests were already run in the same source tree.

### 3.2 Balanced-pipeline smoke workflow

```bash
bash reproduce_balanced_smoke.sh
```

This executes the adaptive reliability orchestrator with very small budgets. The expected pipeline status is `complete`; the scientific status may legitimately be `inconclusive`. The smoke output is not scientific evidence.

## 4. Synthetic verification

Run:

```bash
python scripts/run_synthetic_theory.py \
  --config configs/synthetic_theory.yaml
```

Outputs:

| File | Meaning |
|---|---|
| `theorem1_conditioning_results.csv` | Theorem 1 analytic and dense numerical condition numbers |
| `flat_kronecker_conditioning_results.csv` | Flat-Kronecker invariants and aligned/Adam/reversed condition numbers |
| `chebyshev_certificates.csv` | Weighted-Chebyshev nodes, weights, energies, optimum errors, and dimension bounds |
| `theory_results.csv` | Backward-compatible combined conditioning table |
| `theory_summary.json` | Aggregate pass/fail status and maximum numerical errors |
| `resolved_config.yaml` | Exact synthetic configuration |

A valid run has `all_checks_passed=true` in `theory_summary.json`.

The Chebyshev certificate is separate from the dense conditioning check because endpoint conditioning alone does not verify the full residual-polynomial/Krylov lower-bound device.

## 5. Checkpoint workflow

### 5.1 Screening

Screening is used only to select feasible, pre-registered block types and depths. It disables bootstrap and secondary controls.

```bash
python scripts/run_frozen_analysis.py \
  --config configs/hf_opt125m_screening.yaml
python scripts/validate_run.py \
  --output-dir outputs/hf_opt125m_screening_seed0
```

Do not select confirmatory blocks by inspecting the sign of the screening `ΔG`. Record the selection rule before running confirmatory seeds.

### 5.2 Create seed-specific balanced policies

Create an exact-block confirmatory YAML after screening, then generate at least three policies:

```bash
python scripts/make_balanced_policies.py \
  --base-config configs/generated/hf_opt125m_confirmatory_exact_blocks.yaml \
  --seeds 0 1 2
```

The generated policy paths are written under `configs/generated_balanced/` by default.

### 5.3 Run balanced confirmatory seeds

```bash
for seed in 0 1 2; do
  python scripts/run_balanced_reliability.py \
    --policy configs/generated_balanced/hf_opt125m_balanced_seed${seed}.yaml
  python scripts/validate_balanced_run.py \
    --output-root outputs/hf_opt125m_balanced_seed${seed}
done
```

A completed but numerically inconclusive run is a valid outcome. Use `--require-scientific-acceptance` only as a release gate after the protocol is frozen; do not tune numerical settings until a desired sign appears.

## 6. Balanced reliability protocol

### 6.1 Fixed curvature shift

The first diagnostic stage estimates one stabilization shift per block. The mapping is written to

```text
curvature_shift_overrides.json
```

and reused without re-estimation in subsequent diagnostic, refinement, and final stages. Each core run records the applied value, source, target ridge, and override digest in `curvature_shift_records.csv`.

The implementation is an adaptive PSD floor calibrated once and then frozen across stages. It should not be described as a newly estimated ridge at each Lanczos budget.

### 6.2 Endpoint checks

The endpoint schedule increases Lanczos steps and starts. Numerical acceptance requires both:

- the core Ritz-residual checks; and
- cross-budget stability of `K_adam`, `K_shampoo`, and `ΔG`.

The final high-budget result must again pass its native endpoint checks and agree with the selected diagnostic stage. Repeated point stability cannot override a failed endpoint check.

The labels in this package are **numerical acceptance checks**, not rigorous eigenvalue enclosures.

### 6.3 Partial-trace geometry checks

Aligned and reversed interventions depend on the estimated curvature-factor geometry. Across probe budgets, the pipeline checks:

- negative spectral mass;
- relative Frobenius change of left and right partial traces;
- clustered eigenspace projector distance;
- relative change of aligned and reversed intervention factors.

Near-degenerate eigenvalues are compared as invariant subspaces rather than as individually ordered eigenvectors. An aligned/reversed control is inferentially usable only when the full partial-trace geometry check passes.

### 6.4 Tau-refinement check

The core writes `spectral_gain_curve.csv`, containing `K_adam`, `K_shampoo`, `ΔG`, and saturation information at every declared `τ`. The balanced layer classifies the curve as:

- `stable_nonzero`;
- `one_sided_with_coarse_saturation`;
- `sign_flip`;
- `all_saturated`;
- `tau_refinement_unavailable`.

A missing curve does not silently fall back to a weaker sign check.

### 6.5 Bootstrap

The primary interval is a **calibrated low-budget grouped bootstrap** around the high-budget point estimate. Scientific configs require the covariance batch count to be divisible by the bootstrap group size, preventing unequal final groups from receiving equal bootstrap weight.

The interval is conditional on the fixed checkpoint, curvature batches, curvature shift, and probe stream. It is not a full training-trajectory uncertainty interval.

## 7. Mechanism controls

### 7.1 Assignment

The observed covariance-factor spectra are reassigned in the curvature partial-trace eigenspaces:

- `observed` — measured factor;
- `aligned` — larger factor eigenvalues assigned to larger curvature-factor eigenvalues;
- `reversed` — larger factor eigenvalues assigned to smaller curvature-factor eigenvalues.

Adam remains fixed to the observed coordinate diagonal. The intervention tests the factor-operator channel; it does not claim that every intervened factor pair is the pair of marginals of a single realized full covariance.

### 7.2 Utilization exponent

The default practical factor exponent is `α=0.25`. The `α=0.5` branch is an idealized frozen inverse-square-root control. It is used to test whether stronger utilization amplifies both favorable and unfavorable assignments; it is not a recommendation to replace practical Shampoo.

### 7.3 Damping

Two damping sweeps are reported:

- `joint` — change the normalized Adam and Shampoo damping coefficients together;
- `shampoo_only` — keep the primary Adam spectrum fixed and change only Shampoo damping.

The Shampoo-only sweep is the cleaner test of attenuation of the retained factor anisotropy. The joint sweep is a practical relative-hyperparameter control.

## 8. Output contract

### 8.1 Core frozen run

| File | Purpose |
|---|---|
| `block_metrics.csv` | Centered and uncentered observed primary point estimates and numerical diagnostics |
| `bootstrap_metrics.csv` | Grouped delta-only bootstrap replicates |
| `interventions.csv` | Assignment controls |
| `alpha_sweep.csv` | Utilization-exponent controls |
| `damping_sweep.csv` | Joint and Shampoo-only damping controls |
| `spectral_gain_curve.csv` | Per-`τ` direct gain curves and saturation states |
| `curvature_shift_records.csv` | Applied fixed shift and provenance |
| `partial_trace_artifacts/` | Left/right traces, spectra, and intervention factors used for geometry comparisons |
| `block_failures.csv` | Captured block-level exceptions |
| `run_manifest.json` | Runtime/protocol identity and required-output contract |
| `summary.json` | Run-level summary |
| `resolved_config.yaml` | Validated configuration |

### 8.2 Balanced run root

| File or directory | Purpose |
|---|---|
| `diagnostic_stages/` | Nested low-cost endpoint/probe runs |
| `endpoint_convergence.csv` | Per-stage endpoint estimates and acceptance fields |
| `partial_trace_convergence.csv` | Matrix, subspace, factor, and negative-mass comparisons |
| `balanced_reliability_certificates.csv` | Selected budgets and block-level numerical decisions |
| `curvature_shift_overrides.json` | Frozen blockwise shifts |
| `generated_configs/` | Exact stage and final configs plus change manifests |
| `logs/` | Runner and validator logs |
| `final/` | High-budget run and canonical promoted tables |
| `balanced_reliability_summary.json` | Root-level selected-budget summary |
| `COMPLETED` | Pipeline completion marker |

The final directory contains:

- `canonical_block_metrics.csv` — centered, observed, `α=0.25` primary rows only;
- `canonical_interventions.csv`;
- `canonical_alpha_sweep.csv`;
- `canonical_damping_sweep.csv`;
- `canonical_spectral_gain_curve.csv`;
- `balanced_block_metrics.csv` and balanced-annotated compatibility control tables;
- `scientific_status.json`;
- `balanced_reliability_summary.json`.

`scientific_status.json` separates:

```json
{
  "pipeline_status": "complete",
  "scientific_status": "accepted or inconclusive",
  "primary_inference_available": true
}
```

## 9. Aggregation and paper export

Aggregate balanced roots, not manually selected files from their `final/` directories:

```bash
python scripts/aggregate_runs.py \
  --run-dir outputs/hf_opt125m_balanced_seed0 \
  --run-dir outputs/hf_opt125m_balanced_seed1 \
  --run-dir outputs/hf_opt125m_balanced_seed2 \
  --output-dir outputs/hf_opt125m_balanced_aggregate \
  --minimum-seed-count 3 \
  --make-figures
```

The aggregator auto-resolves each balanced root to its canonical final tables. It refuses to mix balanced canonical and legacy sources unless `--allow-incompatible` is explicitly supplied for debugging.

Strict seed consensus is used:

- all reliable seeds positive → `positive`;
- all reliable seeds negative → `negative`;
- any conflicting or inconclusive seed, or insufficient seed count → `inconclusive`.

Before scientific export, verify `aggregate_manifest.json` contains at least:

```text
reliability_mode = balanced_canonical
all_sources_balanced = true
canonical_tables_used = true
all_primary_rows_numerically_accepted = true
all_balanced_sources_scientifically_accepted = true
minimum_seed_count_met = true
no_block_failures = true
compatible_protocol_hashes = true
compatible_runtime_identities = true
immutable_revisions = true
```

Export:

```bash
python scripts/export_paper_assets.py \
  --output-dir outputs/hf_opt125m_balanced_aggregate \
  --paper-root /absolute/path/to/paper/experiments
```

Scientific export fails closed unless the aggregate is balanced-canonical and all provenance/reliability gates pass. Debug export requires `--allow-debug-export` and stamps generated assets with:

```text
DEBUG EXPORT -- NOT SCIENTIFIC EVIDENCE
```

## 10. Provenance requirements

Scientific Hugging Face configurations require full 40-character immutable revisions for model, tokenizer, and dataset. The manifest additionally records source-row order, packed token-stream content, selected chunks, covariance/curvature intervals, protocol hash, and runtime identity.

Keep `data.order_seed` fixed across confirmatory seeds. Change only the experiment seed and output location unless a preregistered protocol explicitly says otherwise. Prefer the same accelerator model and software environment for every seed.

## 11. Interpretation boundary

The software measures the residual conditioning of frozen operators at selected checkpoints and blocks. It does not prove:

- online Adam or Shampoo convergence ordering;
- stochastic expected-risk ordering;
- whole-network or architecture-wide optimizer dominance;
- generalization superiority;
- wall-clock or memory superiority.

Report positive, negative, and inconclusive blocks. Do not discard a block because its sign is unfavorable or because a numerical gate produces an inconclusive result.
