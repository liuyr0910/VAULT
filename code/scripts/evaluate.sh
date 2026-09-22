#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON=${PYTHON:-python}
PREDICTIONS=${PREDICTIONS:-outputs/predictions}
METRICS=${METRICS:-outputs/metrics}
"$PYTHON" eval_script/eval_cls.py --input "$PREDICTIONS/c.jsonl" --output "$METRICS/classification.csv" --detail "$METRICS/classification_samples.csv"
"$PYTHON" eval_script/eval_tmp.py --input "$PREDICTIONS/t.jsonl" --output "$METRICS/temporal.csv"
"$PYTHON" eval_script/eval_vqa-circular.py --input "$PREDICTIONS/vqa.jsonl" --output "$METRICS/vqa.csv"
"$PYTHON" eval_script/eval_des_bmcr.py --input "$PREDICTIONS/d.jsonl" --output "$METRICS/description.csv"
