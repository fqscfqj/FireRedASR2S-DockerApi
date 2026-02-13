#!/usr/bin/env bash
set -euo pipefail

: "${MODEL_PATH:=/models}"
: "${VRAM_TTL:=300}"
: "${MODEL_DOWNLOAD_MODE:=lazy}"
: "${HOST:=0.0.0.0}"
: "${PORT:=8000}"
: "${LOG_LEVEL:=info}"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

mkdir -p "${MODEL_PATH}"

if [ "$#" -gt 0 ]; then
  exec "$@"
fi

exec uvicorn app.main:app --host "${HOST}" --port "${PORT}" --workers 1 --log-level "${LOG_LEVEL}"

