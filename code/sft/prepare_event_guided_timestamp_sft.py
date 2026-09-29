#!/usr/bin/env python3
"""Prepare EventVAULT RGB timestamps and spatial crops."""

import argparse
import copy
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path

import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "inf_script"))
from event_guided_sampling import (
    EventH5Index,
    EventSamplerConfig,
    analyze_event_h5,
    build_sampling_plan,
    extract_rgb_samples,
    get_video_metadata,
)
from event_guided_timestamp_sampling import (
    build_sampling_prompt,
    build_timestamped_video,
)
from preparation_utils import (
    VIDEO_TAG,
    save_json,
    load_records,
    unique_video_paths,
    video_path_from_record,
)


def materialize_video(job):
    video_path, event_path, cache_root, config, max_edge = job
    folder = Path(cache_root) / Path(video_path).stem
    folder.mkdir(parents=True, exist_ok=True)
    total_frames, fps, duration = get_video_metadata(video_path)
    plan = build_sampling_plan(
        analyze_event_h5(event_path, duration, config), duration, config
    )
    samples = extract_rgb_samples(video_path, plan, max_edge=max_edge)
    video = build_timestamped_video(
        samples, total_frames=total_frames, fps=fps, duration=duration
    )
    target = folder / "video.npy"
    np.save(target, video.frames, allow_pickle=False)
    meta = dict(
        video_path=video_path,
        event_path=event_path,
        video_array_path=str(target.resolve()),
        total_num_frames=total_frames,
        fps=fps,
        duration=duration,
        width=video.canvas_width,
        height=video.canvas_height,
        frames_indices=video.metadata["frames_indices"],
        do_sample_frames=False,
        num_rgb_samples=len(samples),
        num_raw_video_frames=len(video.frames),
        qwen_patch_timestamps=video.patch_timestamps,
        max_edge=max_edge,
        event_sampler_config=asdict(config),
        prompt=build_sampling_prompt(duration, plan, samples),
        sampling=plan.as_dict(),
        rgb_samples=[sample.metadata() for sample in samples],
    )
    save_json(meta, folder / "metadata.json")
    return video_path, meta


def replace_video_with_timestamp_cache(record, cache):
    output = copy.deepcopy(record)
    for turn in output["conversations"]:
        if turn["from"] in ("user", "human"):
            turn["value"] = VIDEO_TAG.sub(
                "<video>\n" + cache["prompt"], turn["value"], count=1
            )
    output["video"] = [cache["video_array_path"]]
    output["event_guided_timestamp_sampling"] = dict(
        metadata_path=str(Path(cache["video_array_path"]).parent / "metadata.json")
    )
    return output


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-json", default=str(ROOT / "sft/data/all_tasks_mixed.json"))
    p.add_argument(
        "--output-json",
        default=str(ROOT / "sft/data/all_tasks_mixed_event_guided_timestamp.json"),
    )
    p.add_argument("--event-root", default=str(ROOT / "data/events/train"))
    p.add_argument("--cache-root", default=str(ROOT / "sft/cache/rgb"))
    p.add_argument("--analysis-bins", type=int, default=96)
    p.add_argument("--min-frames", type=int, default=16)
    p.add_argument("--max-frames", type=int, default=24)
    p.add_argument("--max-edge", type=int, default=352)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--limit-items", type=int, default=0)
    a = p.parse_args()
    records = load_records(a.input_json)
    if a.limit_items:
        records = records[: a.limit_items]
    config = EventSamplerConfig(
        analysis_bins=a.analysis_bins, min_frames=a.min_frames, max_frames=a.max_frames
    )
    index = EventH5Index(a.event_root)
    jobs = [
        (v, index.resolve(v), a.cache_root, config, a.max_edge)
        for v in unique_video_paths(records)
    ]
    if a.workers == 1:
        mapping = dict(
            tqdm(map(materialize_video, jobs), total=len(jobs), desc="RGB caches")
        )
    else:
        with ProcessPoolExecutor(a.workers) as pool:
            mapping = dict(
                tqdm(
                    pool.map(materialize_video, jobs),
                    total=len(jobs),
                    desc="RGB caches",
                )
            )
    output = [
        replace_video_with_timestamp_cache(row, mapping[video_path_from_record(row)])
        for row in records
    ]
    save_json(output, a.output_json)
    print(f"Prepared {len(output)} records from {len(mapping)} videos")


if __name__ == "__main__":
    main()
