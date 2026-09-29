import json
import csv
import os
from tqdm import tqdm
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from collections import defaultdict
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.rouge.rouge import Rouge


def compute_caption_metrics(json_file, csv_file):
    rows = []
    gts_dict = defaultdict(list)
    res_dict = {}
    print(f"Loading data from: {json_file}")
    with open(json_file, encoding="utf-8") as handle:
        lines = [json.loads(line) for line in handle if line.strip()]
    print(f"Processing {len(lines)} samples...")
    for data in tqdm(lines, desc="Parsing data"):
        video_id = os.path.basename(data["video_path"])
        gt_info = data["gt_info"]
        gt = (
            str(gt_info["description"] if isinstance(gt_info, dict) else gt_info)
            .strip()
            .lower()
        )
        pred = str(data["prediction"]).strip().lower()
        gts_dict[video_id].append(gt)
        res_dict[video_id] = [pred]
        rows.append({"video_id": video_id, "gt": gt, "pred": pred})
    smooth_fn = SmoothingFunction().method1
    bleu_scores, meteor_scores = ([], [])
    print("Computing BLEU & METEOR ...")
    for row in tqdm(rows, desc="Description metrics"):
        gt_tokens = [row["gt"].split()]
        pred_tokens = row["pred"].split()
        bleu4 = sentence_bleu(
            gt_tokens,
            pred_tokens,
            weights=(0.25, 0.25, 0.25, 0.25),
            smoothing_function=smooth_fn,
        )
        bleu_scores.append(bleu4)
        meteor = meteor_score([row["gt"].split()], row["pred"].split())
        meteor_scores.append(meteor)
        row["BLEU-4"] = bleu4
        row["METEOR"] = meteor
    print("Computing CIDEr & ROUGE-L ...")
    cider_scorer = Cider()
    rouge_scorer = Rouge()
    cider_avg, cider_scores_dict = cider_scorer.compute_score(gts_dict, res_dict)
    rouge_avg, rouge_scores_dict = rouge_scorer.compute_score(gts_dict, res_dict)
    score_keys = list(gts_dict.keys())
    cider_map = {k: s for k, s in zip(score_keys, cider_scores_dict)}
    rouge_map = {k: s for k, s in zip(score_keys, rouge_scores_dict)}
    for row in rows:
        vid = row["video_id"]
        row["CIDEr"] = cider_map.get(vid, 0.0)
        row["ROUGE-L"] = rouge_map.get(vid, 0.0)
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["video_id", "gt", "pred", "BLEU-4", "METEOR", "CIDEr", "ROUGE-L"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        writer.writerow(
            {
                "video_id": "TOTAL_AVERAGE",
                "gt": "",
                "pred": "",
                "BLEU-4": f"{sum(bleu_scores) / len(bleu_scores):.4f}",
                "METEOR": f"{sum(meteor_scores) / len(meteor_scores):.4f}",
                "CIDEr": f"{cider_avg:.4f}",
                "ROUGE-L": f"{rouge_avg:.4f}",
            }
        )
    return {
        "BLEU-4": sum(bleu_scores) / len(bleu_scores),
        "METEOR": sum(meteor_scores) / len(meteor_scores),
        "CIDEr": cider_avg,
        "ROUGE-L": rouge_avg,
    }


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    compute_caption_metrics(a.input, a.output)
