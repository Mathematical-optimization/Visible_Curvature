# Experiment Output Schema

All tabular outputs are CSV. JSON files are strict RFC-compatible JSON:
non-finite floating-point values are written as `null`. Tensor bundles are
PyTorch `.pt` dictionaries and may contain tensors, metadata mappings, and
`None` for unavailable dense matrices.

## Run-level files

| Path | Meaning |
|---|---|
| `resolved_config.yaml` | Exact configuration used by the runner |
| `manifest.json` | Python/platform/library versions, CUDA availability, Git revision, config SHA-256 |
| `latest_checkpoint.txt` | Path to the final checkpoint created by `train` or `smoke` |
| `probe_batch.pt` | Fixed geometry batch serialized on CPU |


## Primary `streaming/` outputs (v1.2.0)

`streaming-geometry` is the model-scale path. It processes one block at a time,
does not retain an \(N\times d\) gradient matrix, and refuses to overwrite an
existing nonempty `streaming/geometry.csv`.

| Path | Meaning |
|---|---|
| `streaming/manifest.json` | Checkpoint/probe hashes, exact example indices, curvature microbatch size, block list, and resolved streaming policy |
| `streaming/geometry.csv` | One canonical frozen-geometry row per prespecified block |
| `streaming/interventions.csv` | Canonical alpha, damping, and assignment rows when enabled |
| `streaming/solver_diagnostics.jsonl` | Strict JSONL endpoint/censor records for curvature, Adam, and Shampoo operators |
| `streaming/solver_diagnostics.csv` | Tabular conversion of the same solver records |
| `streaming/moments_summary.jsonl` | Per-block moment count, matrix shape, effective ranks, and no-raw-gradient audit flag |
| `streaming/progress.jsonl` | Append-only block completion or pre-analysis censor records |

### `streaming/geometry.csv`

Identification columns include `run_name`, `model_family`, `seed`, `checkpoint`,
`checkpoint_step`, `block_name`, `block_shape`, `dimension`, `factor_elements`,
`curvature_kind`, and `example_count`. The primary numerical columns are:

| Column | Meaning |
|---|---|
| `K_curvature` | Full-space curvature condition, finite only after the selected policy is certified |
| `K_adam` | Adam-form effective condition |
| `K_shampoo` | Shampoo-form effective condition at the configured natural exponent |
| `G_adam` | `log(K_curvature)-log(K_adam)` |
| `G_shampoo` | `log(K_curvature)-log(K_shampoo)` |
| `delta_G` | `G_shampoo-G_adam = log(K_adam)-log(K_shampoo)` |
| `curvature_censored` | Whether the full-space curvature result is unusable |
| `adam_censored` | Whether the Adam effective result is unusable |
| `shampoo_censored` | Whether the Shampoo effective result is unusable |
| `*_censor_reason` | Machine-readable reason; censored rows have non-finite gains serialized as empty CSV fields / `null` in JSON |
| `shampoo_damping_left/right` | Absolute damping applied to each retained factor |
| `left/right_lambda_min_active` | Smallest scale-resolved positive factor eigenvalue |
| `left/right_lambda_max_active` | Largest active factor eigenvalue |
| `left/right_rho_over_m` | Factor-specific damping divided by its active minimum |
| `left/right_rho_over_M` | Factor-specific damping divided by its active maximum |
| `left/right_active_rank` | Scale-aware active factor rank |
| `left/right_numerical_null_rank` | Number of factor modes at or below the relative numerical threshold |
| `left/right_negative_rank` | Number of resolved negative factor modes |
| `left/right_effective_rank` | Entropy-style effective rank diagnostic |
| `adam_active_fraction` | Fraction of diagonal statistic entries above its scale-relative threshold |

Under the default `strict_spd` policy, a resolved null or negative direction,
an uncertified minimum, or an invalid/singular preconditioner is censored. A
finite condition number is never computed after optimizer-specific null-mode
deletion. `positive_active` is an explicit sensitivity policy, not the default.

### `streaming/interventions.csv`

Every row carries the same run/checkpoint/block identifiers and base
`K_curvature`, `K_adam`, and `G_adam`. Common columns are:

| Column | Meaning |
|---|---|
| `intervention` | `alpha`, `damping`, or `assignment` |
| `branch` | `natural`, `aligned`, `reversed`, `random-i`, or `unresolved` |
| `alpha` | Shampoo factor utilization exponent |
| `condition_number` / `K_shampoo` | Effective Shampoo condition for the intervention |
| `gain` / `G_shampoo` | Gain relative to the same curvature operator |
| `delta_G` | Shampoo-minus-Adam gain when both are valid |
| `damping_left/right` | Absolute factor-specific damping |
| `rho_left/right_over_min/max` | Side-specific dimensionless damping |
| `censored` | Whether this intervention row is excluded from hypothesis evaluation |
| `censor_reason` | Invalid factor scale, unresolved endpoint, dimension limit, or inherited base censor reason |

Assignment interventions are materialized only when the block dimension does
not exceed `streaming.assignment_max_dim`. Dense factor accumulation is refused
when \(r^2+c^2\) exceeds `streaming.max_factor_elements`; the block is censored
with `factor_storage_exceeds_limit` rather than silently tiled.

### Solver diagnostics

Each JSONL/CSV record identifies `operator` (`curvature`, `adam_effective`, or
`shampoo_effective`) and contains the estimated endpoints, endpoint residuals,
solver method, starts, Lanczos budget, positive threshold, resolved
null/negative counts, `censored`, and `censor_reason`. Non-finite JSON values are
written as `null`.

## `training/training.csv`

| Column | Meaning |
|---|---|
| `step` | One-indexed optimizer step |
| `loss` | Mean training loss at that step |
| `optimizer` | `adamw` or `shampoo` |

Checkpoints are stored under `training/checkpoints/checkpoint_step_XXXXXX.pt`.
OVC checkpoints contain `model_state`, `optimizer_state`, `step`, `losses`,
`optimizer_name`, and `parameter_names`.

## `geometry/geometry.csv`

### Identification and setup

| Column | Meaning |
|---|---|
| `run_name` | Run identifier |
| `checkpoint` | Checkpoint path |
| `block_name` | `model.named_parameters()` name |
| `block_shape` | Original tensor shape joined by `x` |
| `block_numel` | Number of block parameters |
| `curvature_kind` | `fisher`, `ggn`, or `exact_hessian` |
| `curvature_shift` | Added isotropic curvature shift |
| `moment_centered` | Whether centered moments were used |
| `num_examples` | Probe examples used for moments |
| `shampoo_damping` | Absolute factor damping used for the natural operator |

### Conditioning and gain

| Column | Meaning |
|---|---|
| `K_curvature` | Condition number of the regularized block curvature |
| `K_adam` | Condition number of the population Adam-form effective curvature |
| `G_adam` | `log(K_curvature)-log(K_adam)` |
| `K_shampoo_<alpha>` | Shampoo effective condition number for the indicated exponent |
| `G_shampoo_<alpha>` | Shampoo conditioning gain |
| `delta_G_<alpha>` | `G_shampoo_<alpha>-G_adam` |
| `K_optimizer_state` | Effective condition number of the saved optimizer-state operator, when recoverable |
| `G_optimizer_state` | Gain of the saved optimizer-state operator |

### Mechanism diagnostics

| Column | Meaning |
|---|---|
| `response_adam` | Matched log-response slope for the diagonal statistic |
| `response_adam_spearman` | Rank association of matched Adam statistic and curvature |
| `response_shampoo` | Matched log-response slope for the Kronecker statistic |
| `response_shampoo_spearman` | Rank association of matched Shampoo statistic and curvature |
| `projected_commutator_adam` | Scale-free commutator in the selected curvature subspace |
| `projected_commutator_shampoo` | Shampoo counterpart |
| `leading_overlap_affinity` | Leading-subspace squared-overlap affinity |
| `NCI_adam` | Scale-normalized trace interaction when dense curvature is available |
| `NCI_shampoo_<alpha>` | Shampoo NCI counterpart |
| `K_curvature_q99_q01` | SLQ 99/1 spectral quantile ratio |

### Numerical status

| Column | Meaning |
|---|---|
| `curvature_censored` | Smallest positive curvature eigenvalue unresolved or unreliable |
| `adam_censored` | Adam effective minimum unresolved or unreliable |
| `shampoo_<alpha>_censored` | Shampoo counterpart |
| `optimizer_state_censored` | Saved-state counterpart |
| `optimizer_state_kind` | `adamw`, `shampoo`, or missing |
| `optimizer_state_step` | State step recovered from checkpoint |
| `optimizer_state_scope` | What part of the practical operator is represented |

`geometry/blocks/<safe_block>.pt` stores per-example gradients, centered and
uncentered moments, curvature eigendata, damping, checkpoint metadata, and
optimizer-state metadata. `geometry/spectra/<safe_block>.pt` stores spectra and
solver data used for follow-up analysis.

## `interventions/interventions.csv`

Common columns:

| Column | Meaning |
|---|---|
| `run_name` | Run identifier |
| `block_name` | Parameter block |
| `K_curvature` | Scalar/reference curvature condition number |
| `intervention` | `assignment`, `alpha`, `damping`, `grafting`, `finite_sample`, or `tiling` |
| `branch` | Natural/aligned/reversed/random/tile label |
| `alpha` | Utilization exponent |
| `rho_over_min` | Damping divided by the smallest active factor eigenvalue |
| `rho_over_max` | Damping divided by the largest active factor eigenvalue, when applicable |
| `scale` | Positive scalar grafting multiplier |
| `sample_size` | Moment sample size for finite-sample runs |
| `condition_number` | Effective condition number after intervention |
| `gain` | Conditioning gain relative to the same block curvature |

## `dynamics/dynamics.csv`

| Column | Meaning |
|---|---|
| `run_name` | Run identifier |
| `block_name` | Parameter block |
| `preconditioner` | Identity, Adam, Shampoo exponent, or saved optimizer state |
| `method` | `gradient_descent`, `chebyshev`, or `conjugate_gradient` |
| `iteration` | Iteration/HVP count |
| `relative_objective` | Quadratic objective divided by initial objective |
| `gradient_norm` | Norm of the quadratic gradient/equivalent linear-system residual |
| `step_size` | Scalar update coefficient for GD/CG; missing for iteration zero or Chebyshev recurrence |
| `condition_number` | Measured effective condition number used for context |
| `initial_error_sha256` | Fingerprint of the shared original-coordinate \(e_0\) used for every valid preconditioner in the block |
| `initial_objective` | \(\tfrac12e_0^\top H e_0\), shared across the paired comparison |

## `continuations/continuations.csv`

| Column | Meaning |
|---|---|
| `run_name` | Run identifier |
| `block_name` | Updated block; every other model parameter is frozen |
| `preconditioner` | Fixed operator used during continuation |
| `iteration` | Continuation step including zero |
| `loss` | Fixed-batch mean loss |
| `relative_loss` | Loss divided by the initial loss |
| `gradient_norm` | Euclidean block-gradient norm |
| `step_size` | External scalar step size |
| `effective_max_eigenvalue` | Measured maximum effective curvature used to scale the step |
| `condition_number` | Frozen effective condition number |

## `checkpoint_sweep/*.csv`

- `geometry.csv`: geometry rows with leading `checkpoint_step` and
  `checkpoint_index` columns.
- `interventions.csv`: intervention rows across checkpoints.
- `dynamics.csv`: frozen dynamics across checkpoints.
- `continuations.csv`: optional combined continuations.
- `staleness.csv`: factor source/target checkpoint comparison.

`staleness.csv` contains:

| Column | Meaning |
|---|---|
| `source_step` | Checkpoint providing frozen Shampoo factors |
| `target_step` | Checkpoint providing curvature |
| `checkpoint_lag` | Difference in checkpoint index, not raw optimizer steps |
| `condition_number` | Stale-factor effective condition |
| `fresh_condition_number` | Same-target fresh-factor condition |
| `condition_ratio_to_fresh` | Stale divided by fresh condition |
| `gain` | Stale-factor gain |
| `censored` | Numerical censoring flag |

## `summary/hypotheses.json`

Contains H1--H6 status as `supported`, `not_supported`,
`insufficient_data`, or (for an in-sample positive H2 association)
`descriptive_only`, with the direct numerical evidence used for each status.
A smoke result is only a pipeline check; its H-status is not scientific
evidence.

## Aggregate outputs

`aggregate` writes:

| Path | Meaning |
|---|---|
| `geometry_aggregate.csv` | Concatenated geometry rows |
| `geometry_delta_gain.pdf` | Aggregate gain visualization |
| `statistics.json` | Cluster-bootstrap sign fractions, held-out prediction, and optional paired-assignment summary |
| `mechanism_predictions.csv` | Leave-one-cluster-out predictions when enough clusters exist |
| `interventions_aggregate.csv` | Concatenated intervention rows when `--interventions` is supplied |
| `assignment_paired_effects.csv` | Within-run/block reversed-minus-aligned condition effects |

## Synthetic outputs

`synthetic/synthetic_results.csv` includes experiment-specific columns. In
`flat_kron_pair`, the main audit relation is

```text
K_plus * K_minus == K_H ** 2
```

up to numerical precision. `figures/synthetic_fan.pdf` displays the aligned,
scalar, and reversed branches across the configured parameter grid.

## Canonical H1--H6 interpretation

- H1 uses finite, uncensored `delta_G` or legacy `delta_G_0.25` rows and requires
  recurring positive and negative signs beyond the configured tolerance.
- H2 requires a positive association. In-sample correlation is
  `descriptive_only`; `supported` requires held-out cluster evaluation.
- H3 requires paired aligned positive gain and reversed negative gain. Merely
  observing `K_aligned < K_reversed` is insufficient.
- H4 tests contraction of `|G_shampoo|` as normalized factor damping increases.
  With Adam fixed, `delta_G` generally tends to `-G_adam`, not zero.
- H5 requires alpha `1/2` to preserve the sign and increase magnitude relative
  to alpha `1/4` in both favorable and unfavorable regimes.
- H6 tests scalar-grafting invariance within the full available
  run/model/seed/checkpoint/block grouping key.

Target-relevant censor flags are applied before each hypothesis is evaluated; a
censored measurement cannot contribute finite gain or support.
