#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
: "${BASE_MODEL:?Set BASE_MODEL to the local base-model directory}"
: "${ADAPTER:?Set ADAPTER to the selected checkpoint directory}"
exec "${PYTHON:-python}" sft/merge_timestamp_event_images_lora.py \
  --base-model "$BASE_MODEL" --adapter "$ADAPTER" --output "${MERGED_MODEL:-outputs/merged-model}" "$@"
