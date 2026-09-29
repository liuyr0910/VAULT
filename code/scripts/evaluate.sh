#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON=${PYTHON:-python}
exec "$PYTHON" scripts/evaluate.py \
    --predictions "${PREDICTIONS:-$ROOT/outputs/predictions}" \
    --output "${METRICS:-$ROOT/outputs/metrics}" "$@"
