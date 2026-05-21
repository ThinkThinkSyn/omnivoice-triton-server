#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONDONTWRITEBYTECODE="${PYTHONDONTWRITEBYTECODE:-1}"
export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"
PYTHON_BIN="${OMNIVOICE_PYTHON:-python}"

cd "$ROOT"
exec "$PYTHON_BIN" -m omnivoice-triton-server start "$@"
