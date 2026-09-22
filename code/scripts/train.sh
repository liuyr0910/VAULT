#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# The original training launcher exposes batch size, GPU count and output overrides.
exec bash "$ROOT/sft/script/sft_8b_timestamp_event_images.sh" "$@"
