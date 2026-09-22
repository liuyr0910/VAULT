import json
import csv
import re
import os
from tqdm import tqdm
from collections import defaultdict

def extract_option_char(pred_str):
    """
    从模型预测字符串中提取选项字母 (A/B/C/D)。
    """
    if not pred_str or not isinstance(pred_str, str):
        return ""
    
    clean_str = pred_str.strip()
    # 匹配开头或 Answer 后的字母
    match = re.search(r'(^|answer:\s*|option\s*)([A-D])\b', clean_str, re.IGNORECASE)
    
    if match:
        return match.group(2).upper()
    
    if len(clean_str) == 1 and clean_str.upper() in ['A', 'B', 'C', 'D']:
        return clean_str.upper()
        
    return ""

def evaluate_vqa_circular(jsonl_file, csv_file):
    print(f"📂 Loading data from: {jsonl_file}")
    
    # 1. 加载并按 question_id 分组
    # 因为是 Circular Eval，同一个 question_id 会有 4 条记录
    grouped_data = defaultdict(list)
    with open(jsonl_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                q_id = item.get("question_id")
                grouped_data[q_id].append(item)

    print(f"📊 Grouped into {len(grouped_data)} unique questions.")
    
    # 统计计数器
    # robust: 4次全对才算对 | standard: 只要对一次就算一次
    stats = {
        "overall": {"robust_correct": 0, "standard_correct": 0, "total_qs": 0, "total_runs": 0},
        "by_category": defaultdict(lambda: {"robust_correct": 0, "total_qs": 0})
    }
    
    csv_rows = []

    for q_id, runs in tqdm(grouped_data.items(), desc="Evaluating"):
        # 默认假设这一组全对
        all_runs_correct = True
        q_category = "unknown" # 假设数据中包含 category，如果没有则默认为 unknown
        question_text = ""
        
        run_results = []
        
        # 处理这道题的 4 个位移
        for run in runs:
            question_text = run.get("question", "")
            gt_key = run.get("gt_correct_key", "").strip().upper()
            raw_pred = run.get("prediction", "")
            pred_key = extract_option_char(raw_pred)
            
            is_match = (pred_key == gt_key) and (gt_key != "")
            
            if not is_match:
                all_runs_correct = False
            
            stats["overall"]["standard_correct"] += 1 if is_match else 0
            stats["overall"]["total_runs"] += 1
            
            run_results.append(f"{pred_key}({gt_key})")

        # 更新 Robust 统计（每题计一次）
        stats["overall"]["total_qs"] += 1
        if all_runs_correct and len(runs) == 4: # 确保 4 次全齐
            stats["overall"]["robust_correct"] += 1
            stats["by_category"][q_category]["robust_correct"] += 1
        
        stats["by_category"][q_category]["total_qs"] += 1

        # 记录到 CSV
        csv_rows.append({
            "question_id": q_id,
            "question": question_text,
            "runs_count": len(runs),
            "details": " | ".join(run_results),
            "is_robust_correct": 1 if all_runs_correct and len(runs) == 4 else 0
        })

    # --- 计算指标 ---
    total_qs = stats["overall"]["total_qs"]
    robust_correct = stats["overall"]["robust_correct"]
    standard_correct = stats["overall"]["standard_correct"]
    total_runs = stats["overall"]["total_runs"]

    robust_acc = (robust_correct / total_qs * 100) if total_qs > 0 else 0.0
    standard_acc = (standard_correct / total_runs * 100) if total_runs > 0 else 0.0

    print("\n" + "="*50)
    print(f"🔥 Circular VQA Evaluation Results")
    print("="*50)
    print(f"Unique Questions: {total_qs}")
    print(f"Total Inference Runs: {total_runs}")
    print("-" * 50)
    print(f"Standard Accuracy (Avg): {standard_acc:.2f}%")
    print(f"Robust Accuracy (All 4 Correct): {robust_acc:.2f}%")
    print("-" * 50)
    
    if total_qs > 0:
        print(f"{'Category':<20} | {'Robust Acc':<12} | {'Count':<10}")
        for cat, s in sorted(stats["by_category"].items()):
            cat_acc = (s["robust_correct"] / s["total_qs"] * 100)
            print(f"{cat:<20} | {cat_acc:<10.2f}% | {s['total_qs']:<10}")
    print("="*50)

    # 保存结果
    if csv_file:
        os.makedirs(os.path.dirname(csv_file), exist_ok=True)
        with open(csv_file, "w", newline="", encoding="utf-8") as f:
            fieldnames = ["question_id", "question", "runs_count", "details", "is_robust_correct"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_rows)
            
            writer.writerow({})
            writer.writerow({"question_id": "ROBUST ACCURACY", "is_robust_correct": f"{robust_acc:.2f}%"})
            writer.writerow({"question_id": "STANDARD ACCURACY", "is_robust_correct": f"{standard_acc:.2f}%"})
            
        print(f"\n✅ Evaluation CSV saved to: {csv_file}")


if __name__ == '__main__':
    import argparse
    from pathlib import Path
    p=argparse.ArgumentParser()
    p.add_argument('--input', required=True)
    p.add_argument('--output', required=True)
    a=p.parse_args()
    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    evaluate_vqa_circular(a.input, a.output)
