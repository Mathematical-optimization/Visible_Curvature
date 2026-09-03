#!/usr/bin/env bash
set -euo pipefail
EXPERIMENT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJECT_ROOT="$(cd "$EXPERIMENT_ROOT/.." && pwd)"
cd "$PROJECT_ROOT"
export PYTHONPATH="$EXPERIMENT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

reset_run() {
  rm -rf -- "$EXPERIMENT_ROOT/outputs/$1"
}

for run_name in \
  synthetic-theorems \
  smoke-decoder \
  smoke-vit \
  smoke-decoder-ggn \
  smoke-vit-ggn \
  smoke-decoder-shampoo \
  checkpoint-sweep-decoder \
  checkpoint-sweep-vit \
  streaming-decoder-ggn \
  streaming-vit-ggn \
  aggregate-smoke
do
  reset_run "$run_name"
done

pytest "$EXPERIMENT_ROOT/tests" -q
python -m compileall -q "$EXPERIMENT_ROOT/src" "$EXPERIMENT_ROOT/tests"
python "$EXPERIMENT_ROOT/scripts/validate_hardened.py"
python -m ovc_experiments synthetic --config "$EXPERIMENT_ROOT/configs/synthetic_theorems.yaml"
python -m ovc_experiments smoke --config "$EXPERIMENT_ROOT/configs/smoke_decoder.yaml"
python -m ovc_experiments smoke --config "$EXPERIMENT_ROOT/configs/smoke_vit.yaml"
python -m ovc_experiments smoke --config "$EXPERIMENT_ROOT/configs/smoke_decoder_ggn.yaml"
python -m ovc_experiments smoke --config "$EXPERIMENT_ROOT/configs/smoke_vit_ggn.yaml"
python -m ovc_experiments smoke --config "$EXPERIMENT_ROOT/configs/smoke_decoder_shampoo.yaml"
python -m ovc_experiments checkpoint-sweep --config "$EXPERIMENT_ROOT/configs/checkpoint_sweep_decoder.yaml"
python -m ovc_experiments checkpoint-sweep --config "$EXPERIMENT_ROOT/configs/checkpoint_sweep_vit.yaml"
python -m ovc_experiments streaming-geometry --config "$EXPERIMENT_ROOT/configs/streaming_decoder_ggn.yaml"
python -m ovc_experiments streaming-geometry --config "$EXPERIMENT_ROOT/configs/streaming_vit_ggn.yaml"
python -m ovc_experiments aggregate \
  --geometry \
    "$EXPERIMENT_ROOT/outputs/checkpoint-sweep-decoder/checkpoint_sweep/geometry.csv" \
    "$EXPERIMENT_ROOT/outputs/checkpoint-sweep-vit/checkpoint_sweep/geometry.csv" \
  --interventions \
    "$EXPERIMENT_ROOT/outputs/checkpoint-sweep-decoder/checkpoint_sweep/interventions.csv" \
    "$EXPERIMENT_ROOT/outputs/checkpoint-sweep-vit/checkpoint_sweep/interventions.csv" \
  --output-dir "$EXPERIMENT_ROOT/outputs/aggregate-smoke" \
  --bootstrap-replicates 200 \
  --seed 17
