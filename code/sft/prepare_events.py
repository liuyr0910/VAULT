#!/usr/bin/env python3
"""Add AEER event images to prepared RGB task records."""

import argparse
import copy
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "inf_script"))
from event_representation import load_config, build_bundle, bind_sampler_config
from event_guided_sampling import EventSamplerConfig
from preparation_utils import save_json


def prepare_one(job):
    rgb_path, cache_root, config_path = job
    rgb = json.loads(Path(rgb_path).read_text())
    config = bind_sampler_config(
        load_config(config_path), EventSamplerConfig(**rgb["event_sampler_config"])
    )
    images, meta = build_bundle(
        rgb["event_path"], rgb["rgb_samples"], rgb["fps"], rgb["duration"], config
    )
    folder = Path(cache_root) / Path(rgb["video_path"]).stem
    folder.mkdir(parents=True, exist_ok=True)
    paths = []
    for i, image in enumerate(images):
        target = folder / f"{i:02d}.png"
        image.save(target)
        paths.append(str(target.resolve()))
    meta.update(image_paths=paths)
    target = folder / "event.json"
    save_json(meta, target)
    return rgb_path, str(target.resolve())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--input-json",
        default=str(ROOT / "sft/data/all_tasks_mixed_event_guided_timestamp.json"),
    )
    p.add_argument(
        "--output-json", default=str(ROOT / "sft/data/all_tasks_event_inputs.json")
    )
    p.add_argument("--config", default=str(ROOT / "configs/event_config.json"))
    p.add_argument("--cache-root", default=str(ROOT / "sft/cache/events"))
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--limit-videos", type=int, default=0)
    a = p.parse_args()
    rows = json.loads(Path(a.input_json).read_text())
    unique = list(
        dict.fromkeys(
            row["event_guided_timestamp_sampling"]["metadata_path"] for row in rows
        )
    )
    if a.limit_videos:
        unique = unique[: a.limit_videos]
    jobs = [(path, a.cache_root, a.config) for path in unique]
    if a.workers == 1:
        mapping = dict(map(prepare_one, jobs))
    else:
        with ProcessPoolExecutor(a.workers) as pool:
            mapping = dict(pool.map(prepare_one, jobs))
    output = []
    for row in rows:
        path = row["event_guided_timestamp_sampling"]["metadata_path"]
        if path in mapping:
            record = copy.deepcopy(row)
            record["event_image_supplement"] = dict(metadata_path=mapping[path])
            output.append(record)
    save_json(output, a.output_json)
    print(f"Prepared {len(output)} records from {len(mapping)} videos")


if __name__ == "__main__":
    main()
