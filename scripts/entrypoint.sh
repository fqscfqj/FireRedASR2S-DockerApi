#!/usr/bin/env bash
set -euo pipefail

: "${MODEL_PATH:=/models}"
: "${VRAM_TTL:=300}"
: "${MODEL_DOWNLOAD_MODE:=lazy}"
: "${HOST:=0.0.0.0}"
: "${PORT:=8000}"
: "${LOG_LEVEL:=info}"

mkdir -p "${MODEL_PATH}"

if [ "$#" -gt 0 ]; then
  exec "$@"
fi

exec uvicorn app.main:app --host "${HOST}" --port "${PORT}" --log-level "${LOG_LEVEL}"

