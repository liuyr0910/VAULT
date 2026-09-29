#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON=${PYTHON:-python}
WORKERS=${WORKERS:-2}
"$PYTHON" scripts/build_sft.py --seed 42
"$PYTHON" sft/prepare_event_guided_timestamp_sft.py --workers "$WORKERS" --max-edge 352
"$PYTHON" sft/prepare_events.py --config configs/event_config.json --workers "$WORKERS"
"$PYTHON" sft/prepare_eventvault.py --workers "$WORKERS"
