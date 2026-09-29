#!/usr/bin/env python3
"""Assemble final EventVAULT RGB and event training inputs."""

import argparse
import copy
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "inf_script"))
from eventvault import redistribute, make_plan
from event_guided_sampling import RGBSample, VideoReader, cpu
from event_guided_timestamp_sampling import build_sampling_prompt
from preparation_utils import save_json


def prepare_one(job):
    rgb_path, event_path, cache_root = job
    old = json.loads(Path(rgb_path).read_text())
    event = json.loads(Path(event_path).read_text())
    folder = Path(cache_root) / Path(old["video_path"]).stem
    folder.mkdir(parents=True, exist_ok=True)
    frames = np.load(old["video_array_path"], allow_pickle=False)
    reader = VideoReader(old["video_path"], ctx=cpu(0), num_threads=1)
    height, width = reader[0].shape[:2]
    max_edge = old["max_edge"]
    sizes = []
    for sample in old["rgb_samples"]:
        box = sample.get("pixel_box")
        w, h = (box[2] - box[0], box[3] - box[1]) if box else (width, height)
        scale = min(1.0, max_edge / max(w, h))
        sizes.append((max(1, round(w * scale)), max(1, round(h * scale))))
    canvas = max(w for w, h in sizes), max(h for w, h in sizes)
    rows, array, metadata, event, _ = redistribute(
        old["video_path"], old["rgb_samples"], frames, old, event, canvas, max_edge
    )
    plan = make_plan(old["duration"], rows, old["sampling"]["event_analysis"])
    samples = [
        RGBSample(
            Image.new("RGB", (1, 1)),
            s["timestamp"],
            s["frame_index"],
            s["role"],
            s["roi"],
            s["pixel_box"],
        )
        for s in rows
    ]
    np.save(folder / "video.npy", array, allow_pickle=False)
    rgb = dict(
        old,
        video_array_path=str((folder / "video.npy").resolve()),
        frames_indices=metadata["frames_indices"],
        rgb_samples=rows,
        sampling=plan.as_dict(),
        qwen_patch_timestamps=[s["frame_index"] / old["fps"] for s in rows],
        prompt=build_sampling_prompt(old["duration"], plan, samples),
    )
    rgb_path_new = folder / "metadata.json"
    event_path_new = folder / "event.json"
    save_json(rgb, rgb_path_new)
    # Reuse event PNGs; input assembly only changes their RGB block references.
    save_json(event, event_path_new)
    return (
        rgb_path,
        str(rgb_path_new.resolve()),
        str(event_path_new.resolve()),
        rgb["prompt"],
        old["prompt"],
        rgb["video_array_path"],
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--input-json", default=str(ROOT / "sft/data/all_tasks_event_inputs.json")
    )
    p.add_argument(
        "--output-json", default=str(ROOT / "sft/data/all_tasks_mixed_eventvault.json")
    )
    p.add_argument("--cache-root", default=str(ROOT / "sft/cache/eventvault"))
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--limit-videos", type=int, default=0)
    a = p.parse_args()
    rows = json.loads(Path(a.input_json).read_text())
    unique = dict(
        (
            row["event_guided_timestamp_sampling"]["metadata_path"],
            row["event_image_supplement"]["metadata_path"],
        )
        for row in rows
    )
    jobs = [(rgb, event, a.cache_root) for rgb, event in unique.items()]
    if a.limit_videos:
        jobs = jobs[: a.limit_videos]
    if a.workers == 1:
        results = list(map(prepare_one, jobs))
    else:
        with ProcessPoolExecutor(a.workers) as pool:
            results = list(pool.map(prepare_one, jobs))
    mapping = {result[0]: result[1:] for result in results}
    output = []
    for row in rows:
        key = row["event_guided_timestamp_sampling"]["metadata_path"]
        if key not in mapping:
            continue
        rgb, event, prompt, old_prompt, array = mapping[key]
        record = copy.deepcopy(row)
        for turn in record["conversations"]:
            if turn["from"] in ("user", "human"):
                turn["value"] = turn["value"].replace(old_prompt, prompt, 1)
        record["video"] = [array]
        record["event_guided_timestamp_sampling"]["metadata_path"] = rgb
        record["event_image_supplement"]["metadata_path"] = event
        output.append(record)
    save_json(output, a.output_json)
    print(f"Prepared {len(output)} records from {len(mapping)} videos")


if __name__ == "__main__":
    main()
