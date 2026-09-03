# Optimizer-Visible Curvature Experiments v1.2.0

## Purpose

v1.2.0 is the submission-hardening release for the ICLR 2027 experiment
package. It corrects scientific failure modes in the v1.1.0 prototype and
adds a public, one-block-at-a-time streaming command for model/checkpoint
geometry.

The package evaluates a frozen preconditioning mechanism. It does not establish
online stochastic optimizer dominance or wall-clock superiority.

## Primary execution path

Use:

```bash
ovc-experiments streaming-geometry \
  --config experiments/configs/streaming_decoder_ggn.yaml
```

or the equivalent module command:

```bash
PYTHONPATH=experiments/src python -m ovc_experiments streaming-geometry \
  --config experiments/configs/streaming_decoder_ggn.yaml
```

The runner:

1. loads one model and an optional immutable checkpoint;
2. fixes and hashes the configured probe indices;
3. discovers the prespecified matrix blocks;
4. processes one block at a time;
5. replays curvature microbatches over the full probe set;
6. streams per-example matrix gradients into Adam/Shampoo moments without
   retaining an `N x d` tensor;
7. writes geometry, diagnostics, and optional alpha/damping/assignment rows
   immediately; and
8. releases block-local state before the next block.

The command refuses to overwrite an existing nonempty
`streaming/geometry.csv`. Use a unique `run.name` for every
model/seed/checkpoint/moment panel.

## Strict full-space conditioning

`curvature.subspace_policy: strict_spd` is the default and the confirmatory
policy.

A full-space result is censored if any of the following applies:

- a resolved null or negative effective-curvature direction exists;
- the smallest endpoint is not certified to the configured residual and budget
  tolerance;
- an Adam statistic plus damping is nonpositive;
- a Shampoo factor plus damping is singular under strict-SPD semantics; or
- normalized factor damping has no scale-resolved positive reference
  eigenvalue.

Censored measurements carry a machine-readable reason and do not produce a
finite gain. The implementation never deletes optimizer-specific null modes
and then reports the condition number of the remaining modes as a full-space
result.

`positive_active` remains available only as an explicit sensitivity policy.
Comparisons under that policy require a declared common active subspace; they
must not select a different favorable subspace for each optimizer.

## Scale-aware roots and factor damping

Negative matrix powers use a threshold relative to the operator scale plus a
machine-precision floor. They no longer apply a fixed absolute cutoff that can
silently change under scalar rescaling.

For a Shampoo block, left and right retained factors are diagnosed separately.
The output records:

- active, numerical-null, and negative ranks;
- active minimum and maximum eigenvalues;
- absolute `shampoo_damping_left` and `shampoo_damping_right`; and
- `left/right_rho_over_m` and `left/right_rho_over_M`.

When `geometry.shampoo_damping_ratio` is supplied, each side is normalized by
its own active minimum. No arbitrary `1.0` fallback is used when a factor has no
resolved positive scale.

## Canonical H1--H6 rules

A single evaluator handles legacy wide geometry (`delta_G_0.25`) and canonical
streaming geometry (`delta_G`). Target-relevant censored rows are removed
before evaluation.

- **H1:** finite block gains must contain recurring positive and negative signs
  beyond the configured tolerance.
- **H2:** association must be positive. An in-sample positive correlation is
  `descriptive_only`; `supported` requires held-out cluster evaluation.
- **H3:** within the same run/model/seed/checkpoint/block, aligned assignment
  must have positive gain and reversed assignment negative gain. The inequality
  `K_aligned < K_reversed` alone is insufficient.
- **H4:** increasing factor-normalized damping must contract
  `|G_shampoo|` toward zero. With Adam fixed, `delta_G` generally tends to
  `-G_adam`, not zero.
- **H5:** changing alpha from `1/4` to `1/2` must preserve sign and amplify
  magnitude in both favorable and unfavorable regimes.
- **H6:** a positive scalar graft must preserve condition number within the
  full available run/model/seed/checkpoint/block key.

The `aggregate` command accepts both canonical and legacy delta-gain columns.

## Same-original-error dynamics

Frozen dynamics draw one original-coordinate error `e0` for a block. Every
valid preconditioner is compared from that same `e0`. For an SPD preconditioner
`P`, the symmetric effective problem starts from

```text
z0 = P^(-1/2) e0.
```

Every paired dynamics row records `initial_error_sha256` and
`initial_objective`. Censored or singular operators do not receive transformed
CG/Chebyshev trajectories.

## Memory and factor limits

The primary streaming path does not retain raw per-example gradients. Its
moment memory is `O(r*c + r^2 + c^2)` for an `r x c` matrix block.

The left/right Shampoo factors remain dense. If

```text
r^2 + c^2 > streaming.max_factor_elements,
```

the block is censored with `factor_storage_exceeds_limit`. The runner does not
silently substitute a tiled or diagonal approximation. Large embedding/output
blocks require an explicit implementation-matched tiling study.

Assignment interventions require materializing the curvature proxy and are
therefore limited by `streaming.assignment_max_dim`. Exceeding that limit
censors only the assignment panel, not the base geometry.

## Output safety

- unknown YAML keys fail at the full nested key path;
- JSON and JSONL replace non-finite numbers by `null` and use `allow_nan=False`;
- geometry rows append atomically under the single-writer contract;
- solver diagnostics use an explicit writer rather than a destructive global
  file reset;
- streaming progress is append-only; and
- existing streaming geometry is never overwritten.

See `OUTPUT_SCHEMA.md` for all fields.

## Validation before model-scale runs

From the manuscript-source root:

```bash
PYTHONPATH=experiments/src python experiments/scripts/validate_hardened.py
bash experiments/scripts/run_smoke_suite.sh
bash experiments/scripts/run_validation_suite.sh
```

The smoke suite covers synthetic recovery, legacy broad-surface pipelines, and
both streaming GGN examples. The validation suite additionally runs the full
test suite, compilation, saved-root/checkpoint-sweep paths, and aggregation.

## Remaining scope limitations

- The streaming command performs one checkpoint per run; launch separate
  immutable configs for a checkpoint series and aggregate their CSV files.
- Full matrix-free matched-response, overlap, commutator, and NCI diagnostics
  remain legacy/small-block analyses in this release.
- No distributed multi-node scheduler is included.
- No automatic tiled Shampoo construction is inferred for oversized factors.
