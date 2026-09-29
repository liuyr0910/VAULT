"""Standard and four-permutation robust accuracy for circular VQA."""

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path


def extract_option_char(prediction):
    if not isinstance(prediction, str):
        return ""
    match = re.search("(^|answer:\\s*|option\\s*)([A-D])\\b", prediction.strip(), re.I)
    return match.group(2).upper() if match else ""


def evaluate_vqa_circular(jsonl_file, csv_file):
    groups = defaultdict(list)
    for line in Path(jsonl_file).read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        groups[row["video_path"], row["question_id"]].append(row)
    rows = []
    standard_correct = total_runs = robust_correct = incomplete = 0
    for (video, qid), runs in groups.items():
        shifts = [r.get("shift_id") for r in runs]
        complete = set(shifts) == {0, 1, 2, 3}
        incomplete += int(not complete)
        answers = []
        for r in runs:
            gt = str(r.get("gt_correct_key", "")).strip().upper()
            pred = extract_option_char(r.get("prediction", ""))
            answers.append((pred, gt))
        matches = sum((p == g for p, g in answers))
        robust = complete and matches == 4
        standard_correct += matches
        total_runs += len(runs)
        robust_correct += int(robust)
        rows.append(
            dict(
                video_id=Path(video).name,
                question_id=qid,
                question=runs[0].get("question", ""),
                runs_count=len(runs),
                complete=int(complete),
                details=" | ".join((f"{p}({g})" for p, g in answers)),
                is_robust_correct=int(robust),
            )
        )
    target = Path(csv_file)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    result = dict(
        questions=len(groups),
        runs=total_runs,
        incomplete_questions=incomplete,
        standard_accuracy=standard_correct / total_runs,
        robust_accuracy=robust_correct / len(groups),
    )
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    evaluate_vqa_circular(a.input, a.output)
