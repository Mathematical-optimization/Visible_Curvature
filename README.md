# Optimizer-Visible Curvature Experiments

This directory contains the executable experiment suite for
**Optimizer-Visible Curvature: Assignment, Utilization, and Minimax Order
Reversals**.

The package implements the paper's frozen checkpoint program:

- synthetic recovery of the information-pair and flat-Kronecker identities;
- per-example gradient moments for Adam- and Shampoo-form operators;
- block-restricted empirical Fisher, generalized Gauss--Newton (GGN), and
  exact-Hessian operators;
- residual condition numbers and gains
  \(G_\Phi=\log\operatorname{cond}(H)-\log K_\Phi\);
- matched-direction response, commutator, eigenspace-overlap, and NCI
  diagnostics;
- aligned/random/reversed assignment, utilization-exponent, damping,
  finite-sample, tiling, and stale-factor interventions;
- frozen quadratic GD/Chebyshev/CG trajectories and fixed-batch one-block
  continuations;
- checkpoint sweeps and cluster-aware aggregate statistics;
- deterministic CPU smoke models for a decoder-only Transformer and a ViT.

The code evaluates a frozen mechanism. It does not convert the manuscript's
results into a theorem about stochastic online optimizer dominance or
wall-clock superiority.

## 1. Installation

From the manuscript-source root:

```bash
python -m venv .venv
source .venv/bin/activate                    # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -e "./experiments[dev]"
```

A non-editable installation is also supported:

```bash
python -m pip install ./experiments
```

Minimum supported Python version is 3.11. The package depends on PyTorch,
NumPy, SciPy, pandas, PyYAML, and matplotlib. Parquet is optional; CSV is the
canonical output format.

## 2. Quick validation

```bash
bash experiments/scripts/run_smoke_suite.sh
```

This runs:

1. synthetic theorem recovery;
2. the legacy tiny decoder-only Transformer pipeline;
3. the legacy tiny ViT pipeline;
4. strict-SPD streaming GGN checks for one decoder block and one ViT block.

The legacy smoke runs exercise the broad feature surface. The two streaming
runs exercise the primary bounded-memory path used for model-scale geometry.

A broader CPU validation, including GGN, saved Shampoo roots, and a checkpoint
sweep, is available through:

```bash
bash experiments/scripts/run_validation_suite.sh
```

Run the automated tests with:

```bash
pytest experiments/tests -q
```

## 3. CLI

All commands accept a YAML configuration except `aggregate`.

```bash
ovc-experiments synthetic --config experiments/configs/synthetic_theorems.yaml
ovc-experiments train --config experiments/configs/smoke_decoder.yaml
ovc-experiments geometry --config experiments/configs/smoke_decoder.yaml
ovc-experiments streaming-geometry --config experiments/configs/streaming_decoder_ggn.yaml
ovc-experiments interventions --config experiments/configs/smoke_decoder.yaml
ovc-experiments dynamics --config experiments/configs/smoke_decoder.yaml
ovc-experiments continuation --config experiments/configs/smoke_decoder.yaml
ovc-experiments smoke --config experiments/configs/smoke_decoder.yaml
ovc-experiments checkpoint-sweep --config experiments/configs/checkpoint_sweep_decoder.yaml
```

Aggregate multiple geometry files using cluster bootstrap and held-out
mechanism prediction:

```bash
ovc-experiments aggregate \
  --geometry run-a/geometry.csv run-b/geometry.csv run-c/geometry.csv \
  --interventions run-a/interventions.csv run-b/interventions.csv run-c/interventions.csv \
  --output-dir aggregate-results \
  --sign-threshold 0.22314355131420976 \
  --bootstrap-replicates 2000 \
  --seed 0
```

The module entry point is equivalent:

```bash
python -m ovc_experiments --help
```

## 4. Shipped configurations

| Configuration | Purpose |
|---|---|
| `synthetic_theorems.yaml` | Theorem-pair and \(\alpha r\)-fan recovery |
| `smoke_decoder.yaml` | Decoder, empirical-Fisher, AdamW checkpoint state |
| `smoke_vit.yaml` | ViT, empirical-Fisher, AdamW checkpoint state |
| `smoke_decoder_ggn.yaml` | Decoder GGN operator check |
| `smoke_vit_ggn.yaml` | ViT GGN operator check |
| `smoke_decoder_shampoo.yaml` | Saved Shampoo factor/root-state check |
| `checkpoint_sweep_decoder.yaml` | Multi-checkpoint decoder geometry and staleness |
| `checkpoint_sweep_vit.yaml` | Multi-checkpoint ViT geometry and staleness |
| `streaming_decoder_ggn.yaml` | Strict-SPD, bounded-memory decoder GGN example |
| `streaming_vit_ggn.yaml` | Strict-SPD, bounded-memory ViT GGN example |

The smoke configurations are intentionally small. For confirmatory studies,
start from a streaming GGN configuration, use a distinct `run.name` for each
model/seed/checkpoint/moment panel, and increase model size, examples, block
coverage, Lanczos starts, and seeds only after the validation suites pass.

## 5. Configuration map

```yaml
run:          # name, seed, device, dtype, deterministic mode
model:        # built-in family or Python factory, optional checkpoint
data:         # built-in synthetic data, tensor file, or Python factory
task:         # causal LM, classification, or Python task adapter
blocks:       # regex include/exclude, size filters, optional Shampoo tiling
curvature:    # fisher / ggn / exact_hessian, shift, Lanczos and SLQ controls
moments:      # centered flag, loop/vmap backend, probe count
geometry:     # alpha and damping grids, condition estimation, interventions
streaming:    # curvature microbatching, factor memory limit, assignment limit
continuation: # fixed-batch one-block continuation controls
training:     # AdamW or inspectable MatrixShampoo trajectory
sweep:        # checkpoint analyses, staleness lags, optional continuations
output_dir:   # root directory for run artifacts
```

`dtype: float64` is recommended for geometry estimation. Training-only runs
may use float32 or bfloat16, but the geometry probe should be repeated in
float64 when feasible.

## 6. Using an existing model and checkpoint

### Python model factory

Set `model.family: python` and pass a dotted factory path in
`model.kwargs.factory`:

```yaml
model:
  family: python
  checkpoint: /absolute/path/to/checkpoint.pt
  kwargs:
    factory: my_project.ovc_factories.build_model
    config_path: /absolute/path/to/model_config.json
```

The factory receives all remaining keys as keyword arguments and must return a
`torch.nn.Module`.

Checkpoint loading supports:

- this package's payload with `model_state` and optional `optimizer_state`;
- a mapping containing `state_dict`;
- a bare PyTorch `state_dict`.

A package checkpoint additionally permits extraction of the actual frozen
AdamW diagonal or saved MatrixShampoo roots. External bare checkpoints still
support population-moment Adam/Shampoo analysis, but no optimizer-state panel
is produced unless optimizer state is present in the supported package format.

### Dataset factory or local tensor file

A local tensor dataset can be stored as a `.pt` mapping whose tensors share the
same leading dimension:

```yaml
data:
  family: tensor_file
  path: /absolute/path/to/probe_batch_dataset.pt
  num_examples: 1024
  batch_size: 32
```

For a Python factory:

```yaml
data:
  family: python
  num_examples: 1024
  batch_size: 32
  kwargs:
    factory: my_project.ovc_factories.build_probe_dataset
    split: validation
```

The returned object must implement the PyTorch `Dataset` interface.

### Task adapter

Built-in tasks are `causal_lm` and `classification`. They support tensor,
mapping, tuple/list, Hugging-Face-style `.logits`, and dictionary `logits`
outputs.

```yaml
task:
  family: causal_lm
  input_key: input_ids
  target_key: labels
  ignore_index: -100
```

For unusual losses or model signatures, use a Python factory returning an
object with the `TaskAdapter` protocol implemented in
`ovc_experiments.tasks`.

## 7. Recommended confirmatory workflow

1. Train or point to immutable checkpoints and record their checksums.
2. For each model/seed/checkpoint, run `streaming-geometry` with centered PSD
   GGN, `curvature.subspace_policy: strict_spd`, and a unique `run.name`.
3. Repeat the same prespecified blocks with uncentered moments. Treat actual
   optimizer-state panels as separate controls rather than substitutes for the
   population-moment operators.
4. Inspect censor reasons, endpoint residuals, factor active ranks, and
   left/right normalized damping before interpreting \(\Delta G_b\).
5. Use the streaming alpha/damping/assignment panels where their dimensions are
   certified. Use legacy dense diagnostics only on prespecified small blocks for
   response, overlap, NCI, finite-sample, tiling, staleness, and trajectory
   checks.
6. Run same-original-error frozen dynamics and selected one-block continuations
   as secondary external-validity analyses.
7. Aggregate all prespecified model/seed/checkpoint files with `aggregate`; the
   command accepts both the legacy `delta_G_0.25` schema and the canonical
   streaming `delta_G` schema. Do not select only blocks showing the expected
   sign.

See `RUNBOOK_KO.md` for a detailed Korean execution protocol and
`OUTPUT_SCHEMA.md` for artifact definitions.

## 8. Numerical conventions

- `K_curvature` is the condition number of the regularized block curvature.
- `K_adam`, `K_shampoo_*`, and `K_optimizer_state` are condition numbers of
  \(P^{1/2}HP^{1/2}\).
- `G_* = log(K_curvature) - log(K_*)` removes external scalar scale.
- `delta_G_alpha = G_shampoo_alpha - G_adam`.
- `strict_spd` is the default full-space policy. A resolved null or negative
  direction, an uncertified minimum, or a singular preconditioner censors the
  result; it never yields a finite full-space gain.
- `positive_active` is an explicit sensitivity analysis and must use a common,
  declared active subspace rather than optimizer-specific mode deletion.
- Shampoo damping reports separate left/right values and
  `rho_left/right_over_min/max`; the compatibility fields `rho_over_min/max` are
  retained when meaningful.
- The saved Shampoo optimizer-state operator is the stale root direction before
  any gradient-dependent grafting scalar.
- One-block positive scalar grafting is represented by `ScaledPreconditioner`
  and must leave the residual condition number unchanged up to numerical error.

## 9. Output directories

A run creates:

```text
<output_dir>/<run.name>/
  resolved_config.yaml
  manifest.json
  latest_checkpoint.txt
  streaming/
    manifest.json
    geometry.csv
    interventions.csv
    solver_diagnostics.jsonl
    solver_diagnostics.csv
    moments_summary.jsonl
    progress.jsonl
  training/
  geometry/
  interventions/
  dynamics/
  continuations/
  checkpoint_sweep/
  summary/
  figures/
```

Not every command creates every subdirectory. The primary streaming path never
retains the \(N\times d\) per-example-gradient matrix. Legacy dense commands may
write tensor bundles under `geometry/blocks/`; they are intended for smoke tests
and prespecified small-block diagnostics only. Streaming factor storage is still
dense in the two Shampoo factor dimensions. Blocks exceeding
`streaming.max_factor_elements` are censored and require an explicit tiled
analysis rather than silent approximation.

## 10. Development

```bash
PYTHONPATH=experiments/src pytest experiments/tests -q
python -m compileall -q experiments/src
```

The package uses deterministic seeds, atomic JSON/CSV/tensor writes, strict
JSON serialization (`NaN`/`Inf` become `null`), and a provenance manifest with
software versions, configuration checksum, device availability, and Git
revision when available.


## v1.2.0 submission-hardened path

Use the public command

```bash
ovc-experiments streaming-geometry --config experiments/configs/streaming_decoder_ggn.yaml
```

for checkpoint/block geometry. It processes one parameter block at a time,
streams per-example gradients, microbatches the full probe set for GGN/Hessian
matvecs, writes each block immediately, and releases block-local state. The
lower-level API `ovc_experiments.hardened_runner.analyze_block_streaming`
remains available for custom operators.

Before model-scale jobs, run:

```bash
PYTHONPATH=experiments/src python experiments/scripts/validate_hardened.py
bash experiments/scripts/run_smoke_suite.sh
bash experiments/scripts/run_validation_suite.sh
```

See `docs/HARDENED_V1_2_0.md`, `RUNBOOK_KO.md`, and
`OUTPUT_SCHEMA.md`. v1.1.0 documentation is retained only as historical
reference.
