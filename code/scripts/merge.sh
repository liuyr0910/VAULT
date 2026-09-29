#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON=${PYTHON:-python}
exec "$PYTHON" sft/merge_eventvault_lora.py \
    --base-model "${BASE_MODEL:-$ROOT/models/Qwen3-VL-8B-Instruct}" \
    --adapter "${ADAPTER:-$ROOT/outputs/lora/checkpoint-420}" \
    --output "${MERGED_MODEL:-$ROOT/outputs/merged-model}" "$@"
