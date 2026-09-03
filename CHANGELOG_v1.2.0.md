# Changelog v1.2.0

## Scientific correctness

- Made `strict_spd` the default conditioning/subspace policy.
- Censor resolved null/negative directions, uncertified minima, and invalid
  singular preconditioners instead of reporting finite full-space conditions.
- Added explicit `positive_active` sensitivity semantics without silent
  optimizer-specific null-mode deletion.
- Replaced fixed absolute inverse-root cutoffs with scale-aware thresholds.
- Removed arbitrary factor-scale fallback values.
- Normalize and report left/right Shampoo damping separately.

## Hypothesis evaluation

- Consolidated H1--H6 evaluation across legacy wide and canonical long schemas.
- H2 now rejects negative association and distinguishes in-sample
  `descriptive_only` evidence from held-out support.
- H3 now requires aligned-positive/reversed-negative signed gain reversal.
- H4 now tests contraction of `|G_shampoo|`; documentation records the correct
  fixed-Adam limit `delta_G -> -G_adam`.
- H5 now requires amplification in both favorable and unfavorable regimes.
- H6 now groups by all available run/model/seed/checkpoint/block identifiers.
- Target-relevant censored observations cannot contribute support.

## Dynamics

- Compare all valid preconditioners from one original-coordinate error `e0`.
- Record `initial_error_sha256` and `initial_objective` in trajectory rows.
- Skip transformed CG/Chebyshev runs for singular or censored preconditioners.

## Scalable execution

- Added `ovc-experiments streaming-geometry`.
- Added replayable per-example gradient iteration and full-probe curvature
  microbatching.
- Process and persist one block at a time without retaining an `N x d`
  gradient tensor.
- Added strict factor-memory and assignment-materialization limits with
  explicit censor reasons.
- Added streaming decoder and ViT GGN example configurations.
- Added alpha, factor-normalized damping, and assignment interventions to the
  streaming path.

## Output and reproducibility

- Reject unknown nested YAML keys with their full key path.
- Append hardened block rows safely instead of overwriting earlier blocks.
- Emit strict JSON/JSONL with non-finite values mapped to `null`.
- Replaced destructive global solver-diagnostic handling with explicit writers.
- Added canonical streaming output documentation.
- Made aggregate plots/statistics accept canonical `delta_G` as well as legacy
  `delta_G_0.25`.
- Synchronized package, project, and hardened release versions at `1.2.0`.
- Removed tracked Python bytecode and excluded generated output/build files from
  source distributions.
