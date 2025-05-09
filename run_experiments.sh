#!/usr/bin/env bash
set -euo pipefail

STAGE=${1:-train}
TRIALS=${2:-50}
PARAMS="configs/params_baseline.yaml"

if [[ "$STAGE" == "train" ]]; then
  for batch in 16 32 64; do
    for imgsz in 320 640; do
      echo "=== Запуск train: batch=$batch, imgsz=$imgsz ==="
      dvc exp run \
        --set-param ${PARAMS}:training.batch=$batch \
        --set-param ${PARAMS}:training.imgsz=$imgsz \
        train
    done
  done

elif [[ "$STAGE" == "tune" ]]; then
  echo "=== Запуск Optuna tuning: trials=$TRIALS ==="
  dvc exp run \
    --set-param params_tune.yaml:tune.trials=$TRIALS \
    tune
else
  >&2 echo "Unknown stage: $STAGE (train|tune)"
  exit 1
fi
