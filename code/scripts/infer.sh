#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON=${PYTHON:-python}
exec "$PYTHON" inf_script/infer_eventvault.py \
    --task all --model "${MERGED_MODEL:-$ROOT/outputs/merged-model}" \
    --input-jsonl "${ANNOTATIONS:-$ROOT/data/annotations/test.jsonl}" \
    --event-root "${EVENT_ROOT:-$ROOT/data/events/test}" \
    --output-jsonl "${PREDICTIONS:-$ROOT/outputs/predictions}/{task}.jsonl" "$@"
