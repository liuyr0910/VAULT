#!/usr/bin/env python3
"""Evaluate the four-task inference outputs and save a machine-readable summary."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "eval_script"))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--predictions", type=Path, default=ROOT / "outputs/predictions")
    p.add_argument("--output", type=Path, default=ROOT / "outputs/metrics")
    p.add_argument(
        "--caption-metrics",
        action="store_true",
        help="Also compute BLEU-4/METEOR/CIDEr/ROUGE-L",
    )
    p.add_argument(
        "--judge-scores",
        type=Path,
        help="Existing five-dimension GPT judge CSV; no API calls",
    )
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    from eval_cls import compute_comprehensive_metrics
    from eval_tmp import compute_metrics_new_format
    from eval_vqa_circular import evaluate_vqa_circular

    cls = compute_comprehensive_metrics(
        a.predictions / "c.jsonl",
        a.output / "classification.csv",
        a.output / "classification_samples.csv",
    )
    f1 = cls["sample_F1"]
    iou = compute_metrics_new_format(
        a.predictions / "t.jsonl", a.output / "temporal.csv"
    )
    vqa = evaluate_vqa_circular(a.predictions / "vqa.jsonl", a.output / "vqa.csv")
    summary = dict(
        metric_scale="IoU/F1/VQA: [0,1]; S_GPT: [0,100]",
        mIoU=iou,
        sample_F1=f1,
        VQA_robust_accuracy=vqa["robust_accuracy"],
        VQA_standard_accuracy=vqa["standard_accuracy"],
        VQA_details=vqa,
    )
    if a.caption_metrics:
        from eval_des_bmcr import compute_caption_metrics

        summary["caption_metrics"] = compute_caption_metrics(
            a.predictions / "d.jsonl", a.output / "description.csv"
        )
    if a.judge_scores:
        from eval_des_gpt import aggregate_scores, read_scores

        scores = read_scores(a.judge_scores)
        summary["S_GPT"] = aggregate_scores(scores)
    with (a.output / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
