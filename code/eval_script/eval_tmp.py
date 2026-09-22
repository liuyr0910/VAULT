import json
import csv
import re
import os
from tqdm import tqdm

def parse_new_format(pred_str):
    """
    解析新的模型输出格式字符串。
    输入示例: "0.0 - 29.0 seconds" 或 "12.5 - 15.0 seconds"
    返回: list of lists, e.g., [[0.0, 29.0]]
    """
    if not pred_str or not isinstance(pred_str, str):
        return []

    # 预处理：移除 seconds 单位，转小写
    clean_str = pred_str.lower().replace("seconds", "").replace("sec", "").strip()

    # 正则策略：匹配 "数字 - 数字" 的模式
    # \d+\.?\d* 匹配整数或浮点数
    # \s*-\s* 匹配中间的连字符（兼容空格）
    pattern = r"(-?\d+(?:\.\d*)?)\s*-\s*(-?\d+(?:\.\d*)?)"
    
    segments = []
    matches = re.findall(pattern, clean_str)
    
    for m in matches:
        try:
            start = float(m[0])
            end = float(m[1])
            segments.append([start, end])
        except ValueError:
            continue
            
    return segments

def merge_segments(segments):
    """
    合并重叠的时间段 (Union of intervals)。
    保持原逻辑不变。
    """
    if not segments:
        return []
    
    valid_segs = []
    for s in segments:
        if len(s) >= 2:
            start, end = float(s[0]), float(s[1])
            # 过滤掉无效片段
            if start < end:
                valid_segs.append([start, end])

    if not valid_segs:
        return []

    valid_segs.sort(key=lambda x: x[0])
    
    merged = []
    curr_start, curr_end = valid_segs[0]
    
    for next_start, next_end in valid_segs[1:]:
        if next_start <= curr_end:
            curr_end = max(curr_end, next_end)
        else:
            merged.append([curr_start, curr_end])
            curr_start, curr_end = next_start, next_end
            
    merged.append([curr_start, curr_end])
    return merged

def calculate_global_iou(gt_segs, pred_segs):
    """
    计算 Global 1D IoU。
    保持原逻辑不变：空列表视为 Normal (无异常)。
    """
    # 定义 Normal 检测逻辑
    def is_normal(segs):
        if not segs: return True
        for s in segs:
            # 如果存在有效区间且不是 [-1, -1] 标记
            if len(s) >= 2 and s[0] != -1 and s[1] != -1:
                return False
        return True

    gt_is_normal = is_normal(gt_segs)
    pred_is_normal = is_normal(pred_segs)

    # 1. 均正常 -> IoU 1.0
    if gt_is_normal and pred_is_normal:
        return 1.0
    # 2. 一个正常一个异常 -> IoU 0.0
    if gt_is_normal or pred_is_normal:
        return 0.0

    # 3. 计算区间 IoU
    gt_merged = merge_segments(gt_segs)
    pred_merged = merge_segments(pred_segs)
    
    if not gt_merged or not pred_merged:
        return 0.0

    # Intersection
    intersection_duration = 0.0
    for p in pred_merged:
        for g in gt_merged:
            start = max(p[0], g[0])
            end = min(p[1], g[1])
            if end > start:
                intersection_duration += (end - start)
    
    # Union
    gt_total = sum(x[1] - x[0] for x in gt_merged)
    pred_total = sum(x[1] - x[0] for x in pred_merged)
    
    union_duration = gt_total + pred_total - intersection_duration
    
    if union_duration <= 0:
        return 0.0
        
    return intersection_duration / union_duration

def compute_metrics_new_format(json_file, csv_file):
    rows = []
    total_iou = 0.0
    count = 0

    print(f"📂 Loading data from: {json_file}")
    
    data_list = []
    with open(json_file, "r", encoding="utf-8") as f:
        # 尝试整体读取，失败则尝试逐行读取 (jsonl)
        try:
            data_list = json.load(f)
            if isinstance(data_list, dict):
                data_list = [data_list]
        except json.JSONDecodeError:
            f.seek(0)
            data_list = [json.loads(line) for line in f if line.strip()]

    print(f"📊 Processing {len(data_list)} samples...")

    for item in tqdm(data_list):
        # 1. 获取 Video ID
        video_path = item.get("video_path", "")
        video_id = os.path.basename(video_path)

        # 2. 解析 GT
        # 格式: {"gt_info": {"temporal_segments": [[0, 26]]}}
        gt_info = item.get("gt_info", {})
        gt_raw = gt_info.get("temporal_segments", [])
        
        # 3. 解析 Pred
        # 格式: "prediction": "0.0 - 29.0 seconds"
        pred_str = item.get("prediction", "")
        pred_parsed = parse_new_format(pred_str)

        # 4. 计算 IoU
        iou = calculate_global_iou(gt_raw, pred_parsed)
        
        total_iou += iou
        count += 1

        rows.append({
            "video_id": video_id,
            "gt": str(gt_raw),
            "pred_raw": pred_str,  # 记录原始字符串方便debug
            "pred_parsed": str(pred_parsed),
            "IoU": f"{iou:.4f}"
        })

    avg_iou = total_iou / count if count > 0 else 0.0

    # 写入 CSV
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["video_id", "gt", "pred_raw", "pred_parsed", "IoU"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

        writer.writerow({
            "video_id": "AVERAGE",
            "gt": "",
            "pred_raw": "",
            "pred_parsed": "",
            "IoU": f"{avg_iou:.4f}"
        })

    return avg_iou


if __name__ == '__main__':
    import argparse
    from pathlib import Path
    p=argparse.ArgumentParser()
    p.add_argument('--input', required=True)
    p.add_argument('--output', required=True)
    a=p.parse_args()
    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    compute_metrics_new_format(a.input, a.output)
