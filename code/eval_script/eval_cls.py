import json
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import precision_recall_fscore_support, classification_report

CATEGORY_LIST = [
    "traffic accident",
    "building collapse",
    "fighting",
    "hijack",
    "environmental pollution",
    "fire",
    "burglary",
    "marine accident",
    "vandalism",
    "facility malfunction",
    "illegal hunting",
    "robbery",
    "riot",
    "shooting",
    "natural hazard",
    "normal",
    "theft",
    "traffic violation",
    "object falling",
    "people falling",
    "explosion",
    "abuse",
    "assault",
    "arrest",
    "drowning",
    "animal aggression",
]
KNOWN_CATEGORIES = {c.lower() for c in CATEGORY_LIST}


def parse_categories(cat_data, is_prediction=False):
    """Parse category labels from structured references or prediction text."""
    if not cat_data:
        return set()
    if isinstance(cat_data, list):
        return {str(c).lower().strip() for c in cat_data if str(c).strip()}
    s = str(cat_data).lower().strip()
    s = (
        s.replace("[", "")
        .replace("]", "")
        .replace("'", "")
        .replace('"', "")
        .replace(".", "")
    )
    if s.startswith("category:"):
        s = s[9:].strip()
    found_categories = set()
    parts = [c.strip() for c in s.split(",")]
    for p in parts:
        if p in KNOWN_CATEGORIES:
            found_categories.add(p)
    if is_prediction:
        for category in KNOWN_CATEGORIES:
            if category in s:
                found_categories.add(category)
    if not found_categories and parts:
        cleaned_parts = {p for p in parts if p}
        if cleaned_parts:
            return cleaned_parts
    return found_categories


def compute_comprehensive_metrics(json_file, csv_report_file, csv_detail_file):
    print(f"Loading data from: {json_file}")
    with open(json_file, encoding="utf-8") as handle:
        lines = [json.loads(line) for line in handle if line.strip()]
    y_true_list = []
    y_pred_list = []
    video_ids = []
    raw_preds = []
    print(f"Parsing {len(lines)} samples...")
    for count, item in enumerate(tqdm(lines, desc="Classification")):
        video_id = os.path.basename(item["video_path"])
        video_ids.append(video_id)
        gt_raw = item["gt_info"]["category"]
        pred_raw = item["prediction"]
        raw_preds.append(str(pred_raw)[:100])
        gt_set = parse_categories(gt_raw, is_prediction=False)
        pred_set = parse_categories(pred_raw, is_prediction=True)
        y_true_list.append(list(gt_set))
        y_pred_list.append(list(pred_set))
    mlb = MultiLabelBinarizer(classes=sorted(list(KNOWN_CATEGORIES)))
    mlb.fit([list(KNOWN_CATEGORIES)])
    Y_true_bin = mlb.transform(y_true_list)
    Y_pred_bin = mlb.transform(y_pred_list)
    p_micro, r_micro, f1_micro, _ = precision_recall_fscore_support(
        Y_true_bin, Y_pred_bin, average="micro", zero_division=0
    )
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        Y_true_bin, Y_pred_bin, average="macro", zero_division=0
    )
    report_dict = classification_report(
        Y_true_bin,
        Y_pred_bin,
        target_names=mlb.classes_,
        output_dict=True,
        zero_division=0,
    )
    pd.DataFrame(report_dict).transpose().to_csv(csv_report_file)
    print("Computing Sample-level Details...")
    sample_metrics = {"em": [], "p": [], "r": [], "f1": []}
    detail_rows = []
    for i, vid in enumerate(video_ids):
        gt_s = set(y_true_list[i])
        pred_s = set(y_pred_list[i])
        is_gt_normal = not gt_s or list(gt_s) == ["normal"]
        is_pred_normal = not pred_s or list(pred_s) == ["normal"]
        if is_gt_normal and is_pred_normal:
            em, p, r, f1 = (1.0, 1.0, 1.0, 1.0)
        elif is_gt_normal or is_pred_normal:
            em, p, r, f1 = (0.0, 0.0, 0.0, 0.0)
        else:
            if "normal" in gt_s and len(gt_s) > 1:
                gt_s.remove("normal")
            if "normal" in pred_s and len(pred_s) > 1:
                pred_s.remove("normal")
            intersection = len(gt_s & pred_s)
            em = 1.0 if gt_s == pred_s else 0.0
            p = intersection / len(pred_s) if len(pred_s) > 0 else 0.0
            r = intersection / len(gt_s) if len(gt_s) > 0 else 0.0
            f1 = 2 * p * r / (p + r) if p + r > 0 else 0.0
        sample_metrics["em"].append(em)
        sample_metrics["p"].append(p)
        sample_metrics["r"].append(r)
        sample_metrics["f1"].append(f1)
        detail_rows.append(
            {
                "video_id": vid,
                "gt": ", ".join(sorted(list(gt_s))),
                "pred": ", ".join(sorted(list(pred_s))),
                "raw_pred": raw_preds[i],
                "ExactMatch": int(em),
                "Precision": f"{p:.4f}",
                "Recall": f"{r:.4f}",
                "F1": f"{f1:.4f}",
            }
        )
    pd.DataFrame(detail_rows).to_csv(csv_detail_file, index=False)
    avg_sample_em = np.mean(sample_metrics["em"])
    avg_sample_p = np.mean(sample_metrics["p"])
    avg_sample_r = np.mean(sample_metrics["r"])
    avg_sample_f1 = np.mean(sample_metrics["f1"])
    print("\n" + "=" * 60)
    print("FINAL EVALUATION REPORT")
    print("=" * 60)
    print(f"1. [Sample-based Average] (average over videos)")
    print(f"    Compute metrics per video, then average.")
    print(f"    ---------------------------------------------")
    print(f"    Precision:   {avg_sample_p:.4f}")
    print(f"    Recall:      {avg_sample_r:.4f}")
    print(f"    F1-Score:    {avg_sample_f1:.4f}")
    print(f"    Exact Match: {avg_sample_em:.4f}")
    print("\n" + "-" * 60)
    print(f"2. [Label-based / Global Metrics] (label-based metrics)")
    print(f"    Micro pools all labels; macro averages category metrics.")
    print(f"    ---------------------------------------------")
    print(f"    Micro F1:    {f1_micro:.4f}")
    print(f"    Macro F1:    {f1_macro:.4f}")
    print("\n" + "=" * 60)
    print(f"Outputs saved:")
    print(f"   - Per-class Report:  {csv_report_file}")
    print(f"   - Sample Details:    {csv_detail_file}")
    return dict(
        sample_F1=float(avg_sample_f1),
        micro_F1=float(f1_micro),
        macro_F1=float(f1_macro),
        videos=len(detail_rows),
    )


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--detail", required=True)
    a = p.parse_args()
    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    Path(a.detail).parent.mkdir(parents=True, exist_ok=True)
    compute_comprehensive_metrics(a.input, a.output, a.detail)
