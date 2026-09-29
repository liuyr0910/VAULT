"""Build the selected experiment's four-task training instructions."""

import argparse
import json
import random
from collections import Counter
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "inf_script"))
from task_prompts import CATEGORY_LIST, DESCRIPTION_PROMPT


def build_records(rows, seed=42, data_root=ROOT):
    tasks = {key: [] for key in ("desc", "cls", "loc", "vqa")}
    categories = ", ".join(sorted(CATEGORY_LIST))
    for index, row in enumerate(rows):
        video = Path(row["video_path"])
        identifier = str(row.get("video_id", row.get("id", f"clip_{index:06d}")))

        def add(task, prompt, answer, suffix=None):
            tasks[task].append(
                dict(
                    id=f"{identifier}_{suffix or task}",
                    conversations=[
                        {
                            "from": "user",
                            "value": f"<video>{(data_root / video).resolve()}</video>\n{prompt}",
                        },
                        {"from": "assistant", "value": str(answer)},
                    ],
                )
            )

        if row.get("description"):
            add("desc", DESCRIPTION_PROMPT, row["description"])
        cats = row.get("category", [])
        if isinstance(cats, str):
            cats = [cats]
        add(
            "cls",
            f"Classify the video content into one or multiple of the following categories:\n[{categories}]\nRespond with the category name(s) only, separated by commas if multiple.",
            ", ".join(cats),
        )
        segments = row.get("temporal_segments")
        if segments:
            anomaly = [s for s in segments if s != [-1, -1]]
            response = (
                ", ".join((f"{float(s):.1f} - {float(e):.1f}" for s, e in anomaly))
                or "-1 - -1"
            )
            query = cats[0] if cats[0] != "normal" else "the anomaly"
            add(
                "loc",
                f"Locate the timestamps for: '{query}'.\nFormat: start - end (seconds).\nIf multiple events occur, separate them with commas.\nIf the video contains no anomalies (is normal), output -1 - -1.",
                response,
            )
        for i, qa in enumerate(row.get("vqa_pairs", [])):
            options, correct = (qa["options"], qa["correct_key"])
            options_text = "\n".join((f"{k}: {options[k]}" for k in sorted(options)))
            prompt = f"Question: {qa['question']}\nOptions:\n{options_text}\nRespond with the correct option letter (e.g., A) and the content."
            add("vqa", prompt, f"{correct}: {options[correct]}", f"vqa_{i}")
    records = [record for group in tasks.values() for record in group]
    random.Random(seed).shuffle(records)
    return records


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--annotations", type=Path, default=ROOT / "data/annotations/train.jsonl"
    )
    p.add_argument(
        "--output", type=Path, default=ROOT / "sft/data/all_tasks_mixed.json"
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--data-root", type=Path, default=ROOT)
    a = p.parse_args()
    rows = [
        json.loads(line)
        for line in a.annotations.read_text().splitlines()
        if line.strip()
    ]
    records = build_records(rows, a.seed, a.data_root.resolve())
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open("w") as handle:
        json.dump(records, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    counts = Counter(
        ("vqa" if "_vqa_" in r["id"] else r["id"].rsplit("_", 1)[-1] for r in records)
    )
    print(
        json.dumps(
            dict(
                videos=len(rows), records=len(records), seed=a.seed, tasks=dict(counts)
            )
        )
    )


if __name__ == "__main__":
    main()
