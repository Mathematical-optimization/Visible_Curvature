# OVC Submission Experiment Hardening Design

## Status

Approved scope: implement the scientifically necessary corrections identified in the v1.1.0 review before model-scale ICLR 2027 experiments.

## Goal

Produce a single, internally consistent v1.2.0 source package whose default experimental outputs cannot report a finite conditioning gain after silently deleting null directions, whose H1–H6 summaries match the paper's stated hypotheses, and whose frozen dynamics compare all preconditioners from the same original-coordinate error.

## Scope

### 1. Spectral and preconditioner semantics

- The default condition-number policy is `strict_spd`.
- Exact or certified spectra containing unresolved null/negative directions are censored; no finite full-space condition number is reported.
- Positive-active/pseudoinverse analysis remains available only through an explicit policy.
- Negative matrix powers use scale-aware thresholds and do not silently convert a singular factor into an apparently well-conditioned full-space operator.
- Adam and Shampoo construction failures are represented as censored measurements with a reason, not as crashes or finite gains.
- Left and right Shampoo damping are normalized and recorded separately.

### 2. Canonical hypothesis evaluation

- A single evaluator supports the package's existing wide CSV schema and a canonical long schema.
- H2 requires a positive association and is `descriptive_only` unless held-out evaluation is supplied.
- H3 requires an aligned positive gain and reversed negative gain within run/model/seed/checkpoint/block.
- H4 tests contraction of `|G_shampoo|`; it does not claim `delta_G -> 0` when Adam is fixed.
- H5 compares alpha 1/4 with 1/2 and requires amplification in both favorable and unfavorable regimes.
- H6 groups by all available run/model/seed/checkpoint/block identifiers.
- Censored rows are filtered only with target-relevant censor columns.

### 3. Same-error frozen dynamics

- Draw one original-coordinate error `e0` per block.
- For each SPD preconditioner, transform to `z0 = P^{-1/2} e0` before running the symmetric effective operator.
- Skip dynamics for censored/singular operators.
- Record an initialization fingerprint and initial objective so paired comparisons are auditable.

### 4. Configuration, outputs, and reproducibility

- Unknown YAML keys fail fast at every dataclass level.
- Configuration values governing spectral policy are validated.
- Hardened block output appends atomically rather than overwriting earlier blocks.
- JSON/JSONL output is strict: non-finite values become `null`.
- Solver diagnostics are managed by an explicit writer/context rather than destructive global truncation.
- Package and documentation version become 1.2.0.

### 5. Scalable primary path

- Add a replayable per-example gradient iterator to the functional block API.
- Add a checkpoint/block streaming command that processes one block at a time, writes results immediately, and releases block-local state.
- The scalable path supports GGN primary geometry, Adam/Shampoo frozen operators, and essential alpha/damping/assignment interventions without retaining the N-by-d gradient tensor.
- Legacy dense runners remain available only for smoke tests and backward compatibility.

## Non-goals

- No claim that online Adam/Shampoo trajectories are ordered by the frozen geometry.
- No distributed multi-node scheduler.
- No attempt to materialize ideal full factors for embedding/output dimensions that require tiling; such blocks are censored or require explicit tile configuration.
- No automatic paper text editing in this source package.

## Acceptance criteria

1. A regression with `H = I` and `P = diag(1, 0)` is censored under the default policy.
2. Negative H2 correlation, unsigned H3 ordering, and one-sided H5 data cannot be labeled supported.
3. Multiple hardened block calls preserve all geometry rows and diagnostics.
4. Unknown YAML keys raise a clear `ValueError` including the full key path.
5. Dynamics rows for different preconditioners share the same original `e0` fingerprint and initial objective.
6. The full test suite, validation scripts, smoke suite, source build, and wheel build pass.
