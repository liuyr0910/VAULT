#!/usr/bin/env python3
"""Compute five-dimension description scores and aggregate S_GPT."""

import argparse
import ast
import csv
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
S_GPT_PROMPTS = ROOT / "configs/S_GPT_prompts.json"
DIMENSIONS = (
    "1.semantic_correctness",
    "2.detail_coverage",
    "3.event_causality",
    "4.anomaly_saliency",
    "5.hallucination_penalty",
)


def read_scores(path):
    with Path(path).open() as handle:
        rows = [
            row
            for row in csv.DictReader(handle)
            if row.get("video_name", row.get("video_id", ""))
            not in ("", "AVERAGE", "TOTAL_AVERAGE", "AVERAGE_SCORES")
        ]
    for row in rows:
        for dimension in DIMENSIONS:
            row[dimension] = float(row[dimension])
    return rows


def aggregate_scores(rows):
    means = {
        dimension: sum(float(row[dimension]) for row in rows) / len(rows)
        for dimension in DIMENSIONS
    }
    weighted = sum(
        weight * means[dimension]
        for weight, dimension in zip((2.5, 2.0, 2.0, 3.5), DIMENSIONS[:4])
    )
    return dict(
        videos=len(rows),
        dimension_means=means,
        score=weighted * (1 + means[DIMENSIONS[-1]] / 20),
    )


def parse_reply(content):
    clean = content.strip()
    if clean.startswith("```"):
        clean = "\n".join(clean.splitlines()[1:-1]).strip()
    return float(ast.literal_eval(clean)["score"])


def judge(input_path, output_path, model):
    from openai import OpenAI

    client = OpenAI(
        api_key=os.environ["OPENAI_API_KEY"],
        base_url=os.environ.get("OPENAI_BASE_URL") or None,
    )
    prompts = json.loads(S_GPT_PROMPTS.read_text())
    results = []
    for line in input_path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        gt = row["gt_info"]
        ref = gt["description"] if isinstance(gt, dict) else gt
        pred = row["prediction"]
        result = dict(video_name=Path(row["video_path"]).name, ref=ref, pred=pred)
        for dimension in DIMENSIONS:
            reply = client.chat.completions.create(
                model=model,
                temperature=0,
                messages=[
                    {"role": "system", "content": prompts[dimension]},
                    {
                        "role": "user",
                        "content": f"Reference Description: {ref}\nPredicted Description: {pred}",
                    },
                ],
            )
            result[dimension] = parse_reply(reply.choices[0].message.content)
        results.append(result)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["video_name", "ref", "pred", *DIMENSIONS]
        )
        writer.writeheader()
        writer.writerows(results)
    return aggregate_scores(results)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--input", type=Path)
    mode.add_argument("--scores-csv", type=Path)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "outputs/metrics/description_gpt.csv"
    )
    parser.add_argument("--model", default="gpt-4o-mini")
    args = parser.parse_args()
    result = (
        aggregate_scores(read_scores(args.scores_csv))
        if args.scores_csv
        else judge(args.input, args.output, args.model)
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
