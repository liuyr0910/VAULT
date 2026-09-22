import json
import csv
import os
import nltk
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

    # ====== Read JSON / JSONL ======
    print(f"📂 Loading data from: {json_file}")
    with open(json_file, "r", encoding="utf-8") as f:
        # Support both list-of-dicts and JSON Lines
        try:
            # Try loading as a full JSON list first
            lines = json.load(f)
            if isinstance(lines, dict):
                lines = [lines]
        except json.JSONDecodeError:
            # Fallback to JSON Lines
            f.seek(0)
            lines = []
            for line in f:
                if line.strip():
                    try:
                        lines.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue

    print(f"📊 Processing {len(lines)} samples...")

    for idx, data in enumerate(tqdm(lines, desc="Parsing Data")):
        # ==========================================
        # ✅ 修改点：适配新的 JSON 结构
        # ==========================================
        
        # 1. Get Video ID
        # Extract filename from path if necessary
        raw_path = data.get("video_path", data.get("video_name", f"sample_{idx}"))
        video_id = os.path.basename(raw_path)

        # 2. Get Ground Truth (Reference)
        # New format: "gt_info": {"description": "...", "category": ...}
        gt_info = data.get("gt_info", {})
        if isinstance(gt_info, dict):
            gt = str(gt_info.get("description", "")).strip().lower()
        else:
            # Fallback for old string format
            gt = str(gt_info).strip().lower()

        # 3. Get Prediction
        # ✅ 修改核心：优先读取顶层 "prediction" 字段
        pred = data.get("prediction", "")
        
        # 如果顶层没有，尝试回退到旧逻辑 predictions -> task1_description
        if not pred:
            predictions = data.get("predictions", {})
            if isinstance(predictions, dict):
                pred = predictions.get("task1_description", "")
            else:
                pred = str(predictions)

        pred = str(pred).strip().lower()
        
        # Skip if empty (optional, but good for stability)
        if not gt or not pred:
            # print(f"⚠️ Skipping {video_id}: Missing GT or Pred")
            continue

        # ==========================================
        # End of Modification
        # ==========================================

        # Prepare data for pycocoevalcap (needs list of strings for GT)
        gts_dict[video_id].append(gt)
        res_dict[video_id] = [pred]

        rows.append({
            "video_id": video_id,
            "gt": gt,
            "pred": pred,
        })

    if not rows:
        print("❌ No valid data found to evaluate.")
        return {}

    # ====== BLEU-4 & METEOR ======
    smooth_fn = SmoothingFunction().method1
    bleu_scores, meteor_scores = [], []

    print("Computing BLEU & METEOR ...")
    for row in tqdm(rows, desc="NLTK Metrics"):
        # NLTK expects tokenized lists
        # Simple whitespace tokenization
        gt_tokens = [row["gt"].split()] 
        pred_tokens = row["pred"].split()

        # BLEU-4
        bleu4 = sentence_bleu(gt_tokens, pred_tokens, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smooth_fn)
        bleu_scores.append(bleu4)

        # METEOR
        # Note: meteor_score expects list of strings for references, and a list of strings for hypothesis (tokens)
        try:
            meteor = meteor_score([row["gt"].split()], row["pred"].split())
        except LookupError:
            print("⚠️ NLTK wordnet not found. Running: nltk.download('wordnet')")
            nltk.download('wordnet')
            nltk.download('omw-1.4')
            meteor = meteor_score([row["gt"].split()], row["pred"].split())
            
        meteor_scores.append(meteor)

        row["BLEU-4"] = bleu4
        row["METEOR"] = meteor

    # ====== CIDEr & ROUGE-L ======
    print("Computing CIDEr & ROUGE-L ...")
    
    # pycocoevalcap scorers compute scores for the whole corpus
    cider_scorer = Cider()
    rouge_scorer = Rouge()

    # compute_score returns (overall_score, dict_of_scores_per_image)
    # Note: res_dict values must be list of strings
    cider_avg, cider_scores_dict = cider_scorer.compute_score(gts_dict, res_dict)
    rouge_avg, rouge_scores_dict = rouge_scorer.compute_score(gts_dict, res_dict)

    # Re-map metrics to rows for CSV
    # pycocoevalcap returns per-sample scores in dictionary insertion order.
    # Sorting the keys here assigns scores to the wrong videos.
    score_keys = list(gts_dict.keys())
    
    # Create a lookup for the array results
    cider_map = {k: s for k, s in zip(score_keys, cider_scores_dict)}
    rouge_map = {k: s for k, s in zip(score_keys, rouge_scores_dict)}

    # Map back to rows
    for row in rows:
        vid = row["video_id"]
        row["CIDEr"] = cider_map.get(vid, 0.0)
        row["ROUGE-L"] = rouge_map.get(vid, 0.0)

    # ====== Write to File ======
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["video_id", "gt", "pred", "BLEU-4", "METEOR", "CIDEr", "ROUGE-L"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

        # Summary Row
        writer.writerow({
            "video_id": "TOTAL_AVERAGE",
            "gt": "",
            "pred": "",
            "BLEU-4": f"{sum(bleu_scores) / len(bleu_scores):.4f}",
            "METEOR": f"{sum(meteor_scores) / len(meteor_scores):.4f}",
            "CIDEr": f"{cider_avg:.4f}",
            "ROUGE-L": f"{rouge_avg:.4f}"
        })

    return {
        "BLEU-4": sum(bleu_scores) / len(bleu_scores),
        "METEOR": sum(meteor_scores) / len(meteor_scores),
        "CIDEr": cider_avg,
        "ROUGE-L": rouge_avg
    }


if __name__ == '__main__':
    import argparse
    from pathlib import Path
    p=argparse.ArgumentParser()
    p.add_argument('--input', required=True)
    p.add_argument('--output', required=True)
    a=p.parse_args()
    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    compute_caption_metrics(a.input, a.output)
