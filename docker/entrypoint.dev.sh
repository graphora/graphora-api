#!/usr/bin/env bash
set -euo pipefail
cd /app

if [ "${SKIP_UV_SYNC:-0}" != "1" ]; then
  if [ ! -f "${UV_PROJECT_ENVIRONMENT}/pyvenv.cfg" ] || [ "${FORCE_UV_SYNC:-0}" = "1" ]; then
    echo "[entrypoint] Installing Python dependencies via uv sync..."
    uv sync --frozen
  fi
fi

exec "$@"
