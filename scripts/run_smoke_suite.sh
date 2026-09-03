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
  streaming-decoder-ggn \
  streaming-vit-ggn
do
  reset_run "$run_name"
done

python "$EXPERIMENT_ROOT/scripts/validate_hardened.py"
python -m ovc_experiments synthetic --config "$EXPERIMENT_ROOT/configs/synthetic_theorems.yaml"
python -m ovc_experiments smoke --config "$EXPERIMENT_ROOT/configs/smoke_decoder.yaml"
python -m ovc_experiments smoke --config "$EXPERIMENT_ROOT/configs/smoke_vit.yaml"
python -m ovc_experiments streaming-geometry --config "$EXPERIMENT_ROOT/configs/streaming_decoder_ggn.yaml"
python -m ovc_experiments streaming-geometry --config "$EXPERIMENT_ROOT/configs/streaming_vit_ggn.yaml"
