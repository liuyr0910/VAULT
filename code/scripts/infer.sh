#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
exec "${PYTHON:-python}" inf_script/infer-all-timestamp-event-images.py \
  --model "${MERGED_MODEL:-outputs/merged-model}" \
  --input-jsonl data/annotations/test.jsonl --event-root data/events/test \
  --output-jsonl 'outputs/predictions/{task}.jsonl' --task all "$@"
