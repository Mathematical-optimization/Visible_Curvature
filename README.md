# Visible-Curvature Experiments — Canonical Balanced v1.3.0

This repository contains the frozen-operator experiment package accompanying the ICLR 2027 submission **“Optimizer-Visible Curvature: Information Limits and Minimax Order Reversals.”**

The package studies one deliberately narrow object. At a fixed checkpoint, it constructs an Adam-form or Shampoo-form preconditioner once from a reference second-moment statistic and estimates

\[
K(P,H)=\operatorname{cond}\!\left(P^{1/2}HP^{1/2}\right),
\qquad
\Delta G=\log K_{\mathrm{Adam}}-\log K_{\mathrm{Shampoo}}.
\]

- `ΔG > 0`: the frozen Shampoo-form operator has the smaller residual condition metric.
- `ΔG < 0`: the frozen Adam-form operator has the smaller residual condition metric.
- `ΔG = 0`: the two reported residual condition metrics coincide at the available precision.

The curvature operator is an empirical causal-language-model cross-entropy generalized Gauss–Newton operator. The second moment is built from centered mini-batch-mean block gradients collected in evaluation mode. Covariance and curvature use disjoint batch intervals. The package does **not** establish online optimizer dominance, stochastic-risk dominance, generalization superiority, or wall-clock superiority.

## 1. What changed in v1.3.0

v1.3.0 aligns the empirical controls and reporting with the exact scope of the frozen theory.

1. **Elasticity predictor decomposition**
   - `baseline_width_mismatch = W_adam - (W_left + W_right)`;
   - `delta_g_predicted_consumption` records only inverse-root utilization;
   - `delta_g_predicted_full_proxy` adds the baseline-width mismatch;
   - the legacy `delta_g_predicted` column aliases the full proxy.

2. **Stronger diagnostic gates**
   - left, right, and Adam regression quality;
   - minimum mode count and curvature log-width;
   - factor commutator, eigensolver residual, and negative-mass limits;
   - preconditioner floor-dominance checks;
   - explicit reason codes for degenerate factors and nonfinite predictors.

3. **Theory-aligned controls**
   - joint damping reports `|ΔG|`;
   - Shampoo-only damping reports `|G_shampoo|`, equivalently distance from the scalar-limit value `-G_adam`;
   - the alpha control reports the signed within-block change `ΔG(α)-ΔG(1/4)`;
   - assignment, alpha, and damping summaries are paired within block before cross-block aggregation.

4. **Metric-compatible reliability**
   - ordinary and relative-`τ` truncated conditions are distinct estimands;
   - adaptive stages, final comparison, and seed aggregation require metric agreement;
   - truncated rows also require the same `τ`;
   - primary scientific promotion is ordinary-only by default.

5. **Lower compute cost**
   - block-local Lanczos spectra are cached across repeated Adam and Shampoo control operators;
   - the twelve primary OPT-125M blocks are separated from a four-block expensive-control subset.

6. **Expanded reproducibility**
   - source-tree, software, CUDA/cuDNN, hardware, and deterministic-execution metadata;
   - twelve exact preregistered block names in the confirmatory template;
   - optional ridge-sweep control;
   - integrated three-group Theorem-3 witness verification.

## 2. Installation

Python 3.10 or newer is required.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e '.[dev]'
```

For Hugging Face checkpoint runs:

```bash
pip install -e '.[network,dev]'
```

For a CPU-only verification environment, install a CPU PyTorch wheel first and then install the package with `pip install -e '.[dev]'`.

## 3. Offline verification

### 3.1 Full focused verification

```bash
bash reproduce_smoke.sh
```

The script performs the following checks:

1. Python byte-code compilation and the complete test suite;
2. analytic and dense synthetic conditioning checks;
3. independent weighted-Chebyshev certificates;
4. the integrated budget-dependent Theorem-3 witness;
5. a tiny causal-LM frozen analysis;
6. structural validation, aggregation, and the four retained figures;
7. rejection of debug output by scientific export;
8. explicit watermarking of an allowed debug export.

Set `VC_SKIP_TESTS=1` only after the complete tests have passed in the same source tree.

### 3.2 Balanced-orchestrator smoke

```bash
bash reproduce_balanced_smoke.sh
```

The expected pipeline state is `complete`. The scientific state may be `inconclusive` because the smoke budgets are intentionally tiny. Smoke output is not scientific evidence.

## 4. Synthetic theory verification

```bash
python scripts/run_synthetic_theory.py \
  --config configs/synthetic_theory.yaml
```

Generated files:

| File | Meaning |
|---|---|
| `theorem1_conditioning_results.csv` | Theorem-1 aligned/scalar/reversed condition identities |
| `flat_kronecker_conditioning_results.csv` | Flat-Kronecker invariants and Shampoo order reversal |
| `chebyshev_certificates.csv` | Independent weighted-Chebyshev minimax certificates |
| `integrated_theorem3_witness.csv` | Common-initialization three-group witness and simultaneous certificates |
| `theory_results.csv` | Combined backward-compatible conditioning table |
| `theory_summary.json` | Aggregate pass/fail state and maximum errors |
| `resolved_config.yaml` | Exact synthetic configuration |

A valid run has both `all_checks_passed=true` and `integrated_theorem3_all_checks_passed=true` in `theory_summary.json`.

## 5. Checkpoint workflow

### 5.1 Screening

```bash
python scripts/run_frozen_analysis.py \
  --config configs/hf_opt125m_screening.yaml
python scripts/validate_run.py \
  --output-dir outputs/hf_opt125m_screening_seed0
```

Screening is limited to numerical feasibility and architecture-based block selection. Do not select blocks by inspecting the sign of screening `ΔG`.

### 5.2 Confirmatory block set

`configs/hf_opt125m_confirmatory.yaml` contains twelve exact projection-weight names:

- layers `0, 2, 4, 6, 8, 11`;
- `q_proj.weight` and `out_proj.weight` at each selected layer.

The four expensive-control blocks are the `q_proj` and `out_proj` weights at layers `0` and `11`. Every primary block still receives the centered observed `α=1/4` endpoint and bootstrap analysis.

Scientific confirmatory validation converts `blocks.exact_names` into anchored regular expressions, checks uniqueness, and rejects discovery-only selection.

### 5.3 Generate seed policies

```bash
python scripts/make_balanced_policies.py \
  --base-config configs/hf_opt125m_confirmatory.yaml \
  --seeds 0 1 2 \
  --gpus 0 1 2 \
  --cpu-threads 8
```

Omit `--gpus` for sequential single-GPU execution. The generated policies keep the same exact block set, token order, model revision, dataset revision, and protocol while changing only the declared numerical seed and output root.

### 5.4 Run and validate each balanced seed

```bash
for seed in 0 1 2; do
  python scripts/run_balanced_reliability.py \
    --policy configs/generated_balanced/hf_opt125m_balanced_seed${seed}.yaml
  python scripts/validate_balanced_run.py \
    --output-root outputs/hf_opt125m_balanced_seed${seed}
done
```

A completed but inconclusive seed is a valid scientific outcome. Do not alter numerical thresholds or rerun blocks until a desired sign appears.

### 5.5 Aggregate seeds

```bash
python scripts/aggregate_runs.py \
  --run-dir outputs/hf_opt125m_balanced_seed0 \
  --run-dir outputs/hf_opt125m_balanced_seed1 \
  --run-dir outputs/hf_opt125m_balanced_seed2 \
  --output-dir outputs/hf_opt125m_aggregate \
  --minimum-seed-count 3
```

Seed consensus is unanimous and fail-closed. Any signed conflict, unresolved seed, metric mismatch, `τ` mismatch, or insufficient seed count yields `inconclusive`.

## 6. Estimands and controls

### 6.1 Compatible gains

For each comparison, `K_H`, `K_adam`, and `K_shampoo` are evaluated with the same declared condition metric. The exported gains are

\[
G_{\mathrm{Adam}}=\log K_H-\log K_{\mathrm{Adam}},
\qquad
G_{\mathrm{Shampoo}}=\log K_H-\log K_{\mathrm{Shampoo}}.
\]

The identity `ΔG = G_shampoo - G_adam` is recorded as `delta_g_from_gains`.

### 6.2 Elasticity proxy

The full commuting–Kronecker proxy is

\[
\widehat{\Delta G}_{\mathrm{full}}
=
\underbrace{W_A-(W_L+W_R)}_{\text{baseline width mismatch}}
+
\underbrace{\alpha(r_LW_L+r_RW_R)-\tfrac12r_AW_A}_{\text{utilization response}}.
\]

This is a gated diagnostic proxy, not an exact predictor for a general noncommuting block.

### 6.3 Assignment intervention

Observed factor eigenvalue multisets are reassigned to curvature partial-trace eigenspaces as `aligned` or `reversed`. The observed Adam-form operator is held fixed. This is a surgical frozen-operator intervention; it need not correspond to a full covariance with the observed full spectrum and diagonal.

### 6.4 Damping

- `joint`: both normalized damping coefficients vary; the attenuation target is `|ΔG|`.
- `shampoo_only`: Adam remains fixed; the attenuation target is `|G_shampoo|`, and `ΔG` approaches `-G_adam` rather than zero in a general block.

### 6.5 Alpha

The primary control is the signed paired contrast

\[
\Delta_\alpha=\Delta G(1/2)-\Delta G(1/4).
\]

An increase in `|ΔG|` is not assumed unless the baseline widths and scalar-comparator conditions justify that secondary interpretation. `α=1/2` is an idealized frozen control, not a proposed practical replacement.

### 6.6 Ridge sweep

When enabled, `ridge_sweep.csv` evaluates the same frozen covariance operators under the preregistered relative-ridge coefficients. This is separate from the `τ` sweep: ridge changes the stabilized curvature operator, while `τ` changes only the reported lower-tail condition metric.

## 7. Numerical reliability

The balanced orchestrator increases endpoint steps/starts and partial-trace probe counts. Scientific promotion requires:

1. native Ritz-residual acceptance;
2. stable `K_adam`, `K_shampoo`, and `ΔG` across consecutive budgets;
3. unchanged condition metric and, for truncated rows, unchanged `τ`;
4. a fixed blockwise curvature shift after the first calibration stage;
5. stable partial-trace matrices, clustered eigenspaces, and intervention factors;
6. final-versus-selected-stage agreement;
7. enough finite grouped-bootstrap replicates;
8. a bootstrap interval strictly on one side of zero;
9. point-sign and interval-sign agreement;
10. ordinary primary condition by default.

The endpoint checks are numerical acceptance tests, not rigorous eigenvalue enclosures. Relative-`τ` truncated rows remain available as secondary diagnostics even when they are not promotable.

## 8. Core output files

| File | Content |
|---|---|
| `block_metrics.csv` | Primary centered and uncentered block metrics, endpoints, gains, predictor components, diagnostics, and bootstrap summary |
| `bootstrap_metrics.csv` | Grouped covariance-bootstrap replicate endpoints |
| `interventions.csv` | Observed/aligned/reversed frozen-operator controls |
| `alpha_sweep.csv` | Raw `ΔG(α)` and signed contrast from `α=1/4` |
| `damping_sweep.csv` | Joint and Shampoo-only rows with declared estimand and control value |
| `ridge_sweep.csv` | Optional curvature-ridge sweep rows |
| `spectral_gain_curve.csv` | Relative-`τ` condition curves for every direct comparison |
| `curvature_shift_records.csv` | Applied shift, source, target ridge, and override digest |
| `partial_trace_artifacts/` | Matrices and factor data used for geometry stability checks |
| `runtime_provenance.json` | Source, software, hardware, CUDA, and deterministic-execution metadata |
| `run_manifest.json` | Immutable model/data identity, selected blocks, stream intervals, and required outputs |
| `block_failures.csv` | Captured block-level exceptions |

## 9. Balanced and aggregate outputs

Balanced seed roots contain adaptive stage outputs, fixed shift overrides, reliability certificates, canonical final tables, and `scientific_status.json`.

Aggregation additionally produces:

| File | Content |
|---|---|
| `paired_seed_summary.csv` | Metric-compatible seed medians, intervals, and unanimous sign consensus |
| `elasticity_prediction_rows.csv` | Reliability-gated actual/predicted sign rows |
| `elasticity_prediction_summary.csv` | Eligible count, sign accuracy, balanced accuracy, and Spearman statistic |
| `paired_control_contrasts.csv` | Within-block assignment, alpha, and damping contrasts |
| `aggregate_manifest.json` | Protocol, runtime, numerical, provenance, and export eligibility checks |

Scientific LaTeX export is fail-closed. Debug export requires an explicit flag and is marked with `DEBUG EXPORT -- NOT SCIENTIFIC EVIDENCE`.

## 10. Reproducibility boundary

Every run records:

- immutable model, tokenizer, and dataset revisions;
- fixed token-order and selected-content hashes;
- exact block names and control subset;
- source-tree digest and git state;
- Python, PyTorch, numerical-library, CUDA, and cuDNN versions;
- CPU, GPU, platform, and thread-environment metadata;
- deterministic-algorithm settings;
- covariance and curvature stream intervals.

For confirmatory aggregation, use the same GPU architecture and software environment for all seeds. Numerical seeds are replication of endpoint/probe/bootstrap randomness on one fixed checkpoint and token stream; they are not independent model-training replicates.

## 11. Scope limitations

This package measures fixed-checkpoint residual conditioning. It does not model endogenous preconditioner evolution, momentum, bias correction, stale roots, grafting, fresh stochastic-gradient coupling, whole-network trajectories, generalization, or systems cost. Report positive, negative, and inconclusive blocks together.
