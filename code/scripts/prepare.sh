#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON=${PYTHON:-python}
"$PYTHON" scripts/build_sft.py
"$PYTHON" sft/prepare_event_guided_timestamp_sft.py \
  --input-json sft/data/all_tasks_mixed.json \
  --event-root data/events/train --require-events \
  --output-json sft/data/all_tasks_mixed_event_guided_timestamp.json \
  --cache-root sft/cache/rgb --min-frames 16 --max-frames 24 \
  --max-edge "${MAX_EDGE:-640}" --workers "${WORKERS:-2}" "$@"
"$PYTHON" sft/prepare_timestamp_event_images_sft.py \
  --input-json sft/data/all_tasks_mixed_event_guided_timestamp.json \
  --output-json sft/data/all_tasks_mixed_timestamp_event_images.json \
  --cache-root sft/cache/events \
  --config sft/event_rgb/configs/adaptive_gray_pair.json --workers "${WORKERS:-2}"
