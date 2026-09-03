# Optimizer-Visible Curvature Experiments v1.2.0 — Verification Report

## Scope

This report covers the v1.2.0 submission-hardening source package. The checked
revision includes strict full-space conditioning semantics, factor-specific
damping, canonical H1--H6 evaluation, same-original-error dynamics, safe
outputs, and the public streaming geometry path.

## Environment

- Date: 2026-09-03
- Python: 3.13.5
- Platform: Linux 6.18.35 x86_64, glibc 2.41
- PyTorch: 2.10.0+cpu
- NumPy: 2.3.5
- SciPy: 1.17.0
- pandas: 2.2.3
- PyYAML: 6.0.3
- matplotlib: 3.10.8
- pytest: 9.0.2
- CUDA available: no

## Automated tests and compilation

Commands:

```bash
PYTHONPATH=src pytest -q
python -m compileall -q src tests
```

Result:

```text
98 passed, 0 failed, 18 warnings
compileall exit status 0
```

The warnings are PyTorch deprecation warnings for `torch.jit.script` reached by
higher-order autograd tests. They are not test failures.

The regression suite includes the following release-critical cases:

- `H = I`, `P = diag(1, 0)` is censored under `strict_spd` rather than reported
  as condition one;
- scale-rescaled singular factors retain the same censor semantics;
- left and right Shampoo damping are normalized separately;
- negative H2 association is not support;
- unsigned H3 ordering is not signed reversal;
- unfavorable-only H5 data is not support;
- H6 does not pool different checkpoints;
- all valid frozen dynamics share one original `e0` fingerprint and initial
  objective;
- unknown nested YAML keys fail with their full path;
- strict JSONL contains no `NaN` or `Infinity` literals;
- multiple hardened block calls append instead of overwriting; and
- the checkpoint/block streaming runner uses replayable gradients and the full
  configured probe set for microbatched GGN curvature.

## Hardened validator

Command:

```bash
PYTHONPATH=src python scripts/validate_hardened.py
```

Result:

```json
{
  "diagonal_condition": 1000000.0,
  "flat_kron_passed": true,
  "generic_large_operator_safe": true,
  "streaming_count": 16,
  "weighted_chebyshev_passed": true
}
```

This verifies the reciprocal-closed flat-Kronecker pair, the weighted discrete
Chebyshev barrier, a large diagonal condition, safe large-operator endpoint
handling, and streaming moment accumulation.

## Full smoke and validation suites

Commands:

```bash
bash scripts/run_smoke_suite.sh
bash scripts/run_validation_suite.sh
```

Both scripts exited with status 0. The validation suite reruns the full test and
compile steps and produced the following artifacts.

| Panel | Geometry rows | Intervention rows | Dynamics rows | Continuation rows |
|---|---:|---:|---:|---:|
| Decoder / empirical Fisher | 3 | 120 | 495 | 90 |
| ViT / empirical Fisher | 3 | 120 | 462 | 90 |
| Decoder / GGN | 1 | 23 | 75 | 20 |
| ViT / GGN | 1 | 23 | 75 | 20 |
| Decoder / saved Shampoo state | 1 | 23 | 75 | 20 |

Additional results:

- synthetic theorem recovery wrote 12 rows;
- decoder checkpoint sweep: 4 checkpoints, 8 geometry rows, 20 staleness rows;
- ViT checkpoint sweep: 4 checkpoints, 8 geometry rows, 20 staleness rows;
- decoder streaming GGN: 1/1 block completed, no factor-storage censor;
- ViT streaming GGN: 1/1 block completed, no factor-storage censor; and
- aggregate smoke: 16 geometry rows with paired-assignment/statistics outputs.

The deterministic smoke hypothesis summaries demonstrate the corrected
classification behavior: negative H2 correlations are `not_supported`, H3 is
`not_supported` without aligned-positive/reversed-negative gains, and H5 is
`not_supported` when only the unfavorable regime is present. These smoke
statuses are pipeline checks, not scientific evidence for or against network
typicality.

## Distribution build and installation

The source distribution and wheel were built through the PEP 517 setuptools
backend. Archive inspection found no `__pycache__`, `.pyc`, generated `outputs`,
or build directories. A clean target installation of the wheel reported:

```text
ovc_experiments.__version__ == 1.2.0
```

The installed CLI exposed:

```text
train, geometry, streaming-geometry, interventions, dynamics,
continuation, synthetic, smoke, checkpoint-sweep, aggregate
```

The final release artifacts are accompanied by an external SHA-256 checksum
file.

## Scientific boundary

The validation above establishes software behavior on unit tests, synthetic
identities, and small deterministic CPU models. It does not constitute the
paper's multi-seed, multi-scale Transformer/ViT experiment. It does not show
that signed assignments are typical in trained networks and does not establish
online stochastic optimizer or wall-clock dominance. Confirmatory claims still
require immutable user checkpoints, fixed probe sets, prespecified blocks,
multiple seeds/model scales, uncertainty reporting, and complete disclosure of
censored blocks.
