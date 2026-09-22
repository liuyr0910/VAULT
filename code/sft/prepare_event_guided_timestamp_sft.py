#!/usr/bin/env python3
"""Build timestamp-aware event-guided native-video SFT data.

The event decisions and RGB extraction are identical to
``infer-all-event-guided-sampling-timestamp.py``.  Every unique source video
is sampled once and cached as a uint8 ``.npy`` array plus ``metadata.json``.
All task records reuse that cache and contain one ``<video>`` token.

Run this CPU preprocessing step before the matching LoRA training script.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INFERENCE_DIR = PROJECT_ROOT / "inf_script"
if str(INFERENCE_DIR) not in sys.path:
    sys.path.insert(0, str(INFERENCE_DIR))

from event_guided_sampling import (  # noqa: E402
    EventH5Index,
    EventSamplerConfig,
    SamplingPlan,
    analyze_event_h5,
    build_sampling_plan,
    extract_rgb_samples,
    get_video_metadata,
)
from event_guided_timestamp_sampling import (  # noqa: E402
    QWEN_TEMPORAL_PATCH_SIZE,
    build_sampling_prompt,
    build_timestamped_video,
)
from preparation_utils import (  # noqa: E402
    VIDEO_TAG,
    atomic_json_dump,
    count_values,
    file_fingerprint,
    load_records,
    stable_video_key,
    task_name,
    unique_video_paths,
    video_path_from_record,
)


CACHE_VERSION = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-json",
        default=str(PROJECT_ROOT / "sft/data/all_tasks_mixed.json"),
    )
    parser.add_argument(
        "--event-root",
        default=str(PROJECT_ROOT / "data/events/train"),
    )
    parser.add_argument(
        "--cache-root",
        default=str(PROJECT_ROOT / "sft/cache/event_guided_timestamp_train"),
    )
    parser.add_argument(
        "--output-json",
        default=str(PROJECT_ROOT / "sft/data/all_tasks_mixed_event_guided_timestamp.json"),
    )
    parser.add_argument("--analysis-bins", type=int, default=96)
    parser.add_argument("--min-frames", type=int, default=16)
    parser.add_argument("--max-frames", type=int, default=24)
    parser.add_argument(
        "--max-edge",
        type=int,
        default=640,
        help="Maximum edge before full frames/crops are placed on a common canvas.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=min(2, os.cpu_count() or 1),
        help="Concurrent CPU cache builders; keep small to bound RAM usage.",
    )
    parser.add_argument("--require-events", action="store_true")
    parser.add_argument("--overwrite-output", action="store_true")
    parser.add_argument("--overwrite-cache", action="store_true")
    parser.add_argument("--limit-items", type=int, default=0)
    return parser.parse_args()


def config_signature(config: EventSamplerConfig, max_edge: int) -> str:
    payload = {
        "cache_version": CACHE_VERSION,
        "sampler": asdict(config),
        "max_edge": max_edge,
        "temporal_patch_size": QWEN_TEMPORAL_PATCH_SIZE,
        "pairing": "duplicate_each_selected_rgb_sample",
        "canvas": "aspect_preserving_gray_letterbox",
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def load_valid_cache(
    metadata_path: Path,
    video_fingerprint: Dict,
    event_fingerprint: Optional[Dict],
    signature: str,
) -> Optional[Dict]:
    if not metadata_path.is_file():
        return None
    try:
        with metadata_path.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None
    if metadata.get("config_signature") != signature:
        return None
    if metadata.get("video_fingerprint") != video_fingerprint:
        return None
    if metadata.get("event_fingerprint") != event_fingerprint:
        return None

    archive_path = Path(str(metadata.get("video_array_path", "")))
    if not archive_path.is_file():
        return None
    try:
        array = np.load(archive_path, mmap_mode="r", allow_pickle=False)
        valid_shape = (
            array.ndim == 4
            and array.shape[-1] == 3
            and array.dtype == np.uint8
            and array.shape[0] == len(metadata.get("frames_indices", []))
        )
    except (OSError, ValueError):
        return None
    return metadata if valid_shape else None


def make_plan(
    event_path: Optional[str],
    duration: float,
    config: EventSamplerConfig,
    require_events: bool,
) -> Tuple[SamplingPlan, Optional[str]]:
    if not event_path:
        if require_events:
            raise FileNotFoundError("No matching event H5")
        return build_sampling_plan(None, duration, config), "event_h5_missing"
    try:
        analysis = analyze_event_h5(event_path, duration, config)
        return build_sampling_plan(analysis, duration, config), None
    except Exception as exc:
        if require_events:
            raise
        return build_sampling_plan(None, duration, config), f"event_analysis_failed: {exc}"


def _atomic_numpy_save(array: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp.npy")
    try:
        np.save(temporary, array, allow_pickle=False)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def materialize_video(
    video_path: str,
    event_path: Optional[str],
    cache_directory: Path,
    signature: str,
    config: EventSamplerConfig,
    max_edge: int,
    require_events: bool,
    overwrite_cache: bool,
) -> Tuple[Dict, bool]:
    video_fingerprint = file_fingerprint(video_path)
    if video_fingerprint is None:
        raise FileNotFoundError(video_path)
    event_fingerprint = file_fingerprint(event_path)
    metadata_path = cache_directory / "metadata.json"
    if not overwrite_cache:
        cached = load_valid_cache(
            metadata_path,
            video_fingerprint,
            event_fingerprint,
            signature,
        )
        if cached is not None:
            return cached, True

    total_frames, fps, duration = get_video_metadata(video_path)
    plan, fallback_reason = make_plan(event_path, duration, config, require_events)
    samples = extract_rgb_samples(video_path, plan, max_edge=max_edge)
    timestamped_video = build_timestamped_video(
        samples,
        total_frames=total_frames,
        fps=fps,
        duration=duration,
    )

    cache_directory.mkdir(parents=True, exist_ok=True)
    video_array_path = cache_directory / "video.npy"
    _atomic_numpy_save(timestamped_video.frames, video_array_path)

    metadata = {
        "cache_version": CACHE_VERSION,
        "config_signature": signature,
        "video_path": video_path,
        "event_path": event_path,
        "video_fingerprint": video_fingerprint,
        "event_fingerprint": event_fingerprint,
        "fallback_reason": fallback_reason,
        "video_array_path": str(video_array_path.resolve()),
        "total_num_frames": int(total_frames),
        "fps": float(fps),
        "duration": float(duration),
        "width": int(timestamped_video.canvas_width),
        "height": int(timestamped_video.canvas_height),
        "frames_indices": list(timestamped_video.metadata["frames_indices"]),
        "do_sample_frames": False,
        "num_rgb_samples": len(samples),
        "num_raw_video_frames": int(timestamped_video.frames.shape[0]),
        "num_qwen_temporal_patches": len(timestamped_video.patch_timestamps),
        "qwen_patch_timestamps": timestamped_video.patch_timestamps,
        "prompt": build_sampling_prompt(duration, plan, samples),
        "sampling": plan.as_dict(),
        "rgb_samples": [sample.metadata() for sample in samples],
    }
    atomic_json_dump(metadata, metadata_path)
    return metadata, False


def replace_video_with_timestamp_cache(record: Dict, cache: Dict) -> Dict:
    output = copy.deepcopy(record)
    if output.get("image"):
        raise ValueError(
            f"Record {record.get('id', '<unknown>')} already contains images"
        )

    replacement = "<video>\n" + cache["prompt"]
    replaced = 0
    for turn in output.get("conversations", []):
        if turn.get("from") not in ("user", "human"):
            continue
        value, count = VIDEO_TAG.subn(replacement, str(turn.get("value", "")), count=1)
        if count:
            turn["value"] = value
            replaced += count
    if replaced != 1:
        raise ValueError(
            f"Record {record.get('id', '<unknown>')} expected one <video> replacement; got {replaced}"
        )

    output["video"] = [cache["video_array_path"]]
    output.pop("image", None)
    output["event_guided_timestamp_sampling"] = {
        "source_video_path": cache["video_path"],
        "event_path": cache["event_path"],
        "mode": cache["sampling"]["mode"],
        "num_rgb_samples": cache["num_rgb_samples"],
        "num_raw_video_frames": cache["num_raw_video_frames"],
        "metadata_path": str(
            (Path(cache["video_array_path"]).parent / "metadata.json").resolve()
        ),
    }
    return output


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_json).expanduser().resolve()
    output_path = Path(args.output_json).expanduser().resolve()
    cache_root = Path(args.cache_root).expanduser().resolve()
    if output_path.exists() and not args.overwrite_output:
        raise FileExistsError(
            f"Output already exists: {output_path}; pass --overwrite-output to replace it"
        )
    if not 1 <= args.min_frames <= args.max_frames:
        raise ValueError("expected 1 <= min-frames <= max-frames")
    if args.max_edge <= 0 or args.workers <= 0:
        raise ValueError("max-edge and workers must be positive")

    records = load_records(input_path)
    if args.limit_items:
        records = records[: args.limit_items]
    if not records:
        raise ValueError("No SFT records to process")

    sampler_config = EventSamplerConfig(
        analysis_bins=args.analysis_bins,
        min_frames=args.min_frames,
        max_frames=args.max_frames,
    )
    signature = config_signature(sampler_config, args.max_edge)
    signature_root = cache_root / signature
    event_index = EventH5Index(args.event_root)
    videos = unique_video_paths(records)
    event_by_video = {
        video_path: event_index.resolve(video_path) for video_path in videos
    }

    print(f"SFT records: {len(records)}")
    print(f"Unique videos: {len(videos)}")
    print(f"Indexed event H5 files: {event_index.indexed_count}")
    print("Sampling: event-guided RGB -> duplicated temporal pairs -> one native video")
    print(f"Timestamp-video cache: {signature_root}")

    cache_by_video: Dict[str, Dict] = {}
    reused = 0
    generated = 0

    def collect(video_path: str, result: Tuple[Dict, bool]) -> None:
        nonlocal reused, generated
        metadata, was_reused = result
        cache_by_video[video_path] = metadata
        reused += int(was_reused)
        generated += int(not was_reused)

    arguments = dict(
        signature=signature,
        config=sampler_config,
        max_edge=args.max_edge,
        require_events=args.require_events,
        overwrite_cache=args.overwrite_cache,
    )
    if args.workers == 1:
        for video_path in tqdm(videos, desc="timestamp video caches"):
            collect(
                video_path,
                materialize_video(
                    video_path,
                    event_by_video[video_path],
                    signature_root / stable_video_key(video_path),
                    **arguments,
                ),
            )
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(
                    materialize_video,
                    video_path,
                    event_by_video[video_path],
                    signature_root / stable_video_key(video_path),
                    **arguments,
                ): video_path
                for video_path in videos
            }
            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="timestamp video caches",
            ):
                collect(futures[future], future.result())

    converted = [
        replace_video_with_timestamp_cache(
            record,
            cache_by_video[video_path_from_record(record)],
        )
        for record in tqdm(records, desc="building timestamp SFT JSON")
    ]
    atomic_json_dump(converted, output_path)

    metadata_values = list(cache_by_video.values())
    manifest = {
        "input_json": str(input_path),
        "output_json": str(output_path),
        "event_root": str(Path(args.event_root).expanduser().resolve()),
        "cache_root": str(signature_root),
        "config_signature": signature,
        "event_sampler_config": asdict(sampler_config),
        "max_edge": args.max_edge,
        "records": len(converted),
        "unique_videos": len(videos),
        "task_counts": count_values(task_name(record) for record in converted),
        "temporal_patch_size": QWEN_TEMPORAL_PATCH_SIZE,
        "min_rgb_samples_per_video": min(
            metadata["num_rgb_samples"] for metadata in metadata_values
        ),
        "max_rgb_samples_per_video": max(
            metadata["num_rgb_samples"] for metadata in metadata_values
        ),
        "event_mode_videos": sum(
            metadata["sampling"]["mode"] == "event_guided_temporal_spatial"
            for metadata in metadata_values
        ),
        "fallback_mode_videos": sum(
            metadata["sampling"]["mode"] == "uniform_fallback"
            for metadata in metadata_values
        ),
        "generated_video_caches": generated,
        "reused_video_caches": reused,
    }
    atomic_json_dump(manifest, output_path.with_suffix(".manifest.json"))
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
