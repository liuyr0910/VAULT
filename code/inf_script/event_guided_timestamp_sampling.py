"""Shared packing logic for timestamp-aware event-guided Qwen3-VL video.

Training and inference must use this module together.  It converts the RGB
samples produced by :mod:`event_guided_sampling` into one non-uniform native
video whose original ``fps`` and ``frames_indices`` let Qwen3-VL construct
text timestamps automatically.
"""

from __future__ import annotations
from dataclasses import dataclass
import math
from typing import Any, Dict, List, Sequence
import numpy as np
from PIL import Image
from event_guided_sampling import RGBSample, SamplingPlan
from event_guided_video_resolution import (
    preserve_video_resolution,
    build_roi_block_prompt,
)

QWEN_TEMPORAL_PATCH_SIZE = 2


@dataclass
class TimestampedVideoInput:
    """One pre-sampled native video and its original-video time metadata."""

    frames: np.ndarray
    metadata: Dict[str, Any]
    sample_frame_indices: List[int]
    patch_timestamps: List[float]
    canvas_width: int
    canvas_height: int


def _letterbox(image: Image.Image, width: int, height: int) -> Image.Image:
    """Fit a full frame or ROI crop onto a common canvas without distortion."""
    image = image.convert("RGB")
    scale = min(width / image.width, height / image.height)
    resized_width = max(1, min(width, int(round(image.width * scale))))
    resized_height = max(1, min(height, int(round(image.height * scale))))
    if image.size != (resized_width, resized_height):
        image = image.resize((resized_width, resized_height), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (width, height), color=(127, 127, 127))
    offset = ((width - resized_width) // 2, (height - resized_height) // 2)
    canvas.paste(image, offset)
    return canvas


def _render_rgb_block(sample: RGBSample, width: int, height: int) -> np.ndarray:
    if sample.spatial_view == "crop":
        return np.asarray(_letterbox(sample.image, width, height), dtype=np.uint8)
    full = sample.full_image
    array = np.array(_letterbox(full, width, height), dtype=np.uint8, copy=True)
    if sample.pixel_box is None:
        return array
    source_width, source_height = sample.source_size
    scale = min(width / full.width, height / full.height)
    rw = max(1, min(width, int(round(full.width * scale))))
    rh = max(1, min(height, int(round(full.height * scale))))
    ox, oy = ((width - rw) // 2, (height - rh) // 2)
    box = sample.pixel_box
    left = max(ox, ox + math.floor(box[0] * rw / source_width))
    top = max(oy, oy + math.floor(box[1] * rh / source_height))
    right = min(ox + rw, ox + math.ceil(box[2] * rw / source_width))
    bottom = min(oy + rh, oy + math.ceil(box[3] * rh / source_height))
    array[:top] = 127
    array[bottom:] = 127
    array[top:bottom, :left] = 127
    array[top:bottom, right:] = 127
    return array


def build_timestamped_video(
    samples: Sequence[RGBSample], total_frames: int, fps: float, duration: float
) -> TimestampedVideoInput:
    """Pack event-guided RGB samples into timestamp-safe temporal pairs.

    Every selected image is repeated twice so Qwen's temporal patch size of
    two never pairs unrelated non-uniform samples.  Repeating the original
    frame index makes the processor timestamp exactly ``frame_index / fps``.
    """
    sample_indices = [int(sample.frame_index) for sample in samples]
    canvas_width = max((sample.image.width for sample in samples))
    canvas_height = max((sample.image.height for sample in samples))
    raw_frame_count = len(samples) * QWEN_TEMPORAL_PATCH_SIZE
    frames = np.empty((raw_frame_count, canvas_height, canvas_width, 3), dtype=np.uint8)
    packed_indices: List[int] = []
    patch_timestamps: List[float] = []
    for sample_index, sample in enumerate(samples):
        array = _render_rgb_block(sample, canvas_width, canvas_height)
        start = sample_index * QWEN_TEMPORAL_PATCH_SIZE
        stop = start + QWEN_TEMPORAL_PATCH_SIZE
        frames[start:stop] = array
        packed_indices.extend([int(sample.frame_index)] * QWEN_TEMPORAL_PATCH_SIZE)
        patch_timestamps.append(float(sample.frame_index) / fps)
    metadata = {
        "total_num_frames": int(total_frames),
        "fps": float(fps),
        "width": int(canvas_width),
        "height": int(canvas_height),
        "duration": float(duration),
        "video_backend": "event_guided_presampled_rgb",
        "frames_indices": packed_indices,
        "do_sample_frames": False,
    }
    frames, metadata = preserve_video_resolution(frames, metadata)
    return TimestampedVideoInput(
        frames=np.ascontiguousarray(frames),
        metadata=metadata,
        sample_frame_indices=sample_indices,
        patch_timestamps=patch_timestamps,
        canvas_width=metadata["width"],
        canvas_height=metadata["height"],
    )


def build_sampling_prompt(
    duration: float, plan: SamplingPlan, samples: Sequence[RGBSample]
) -> str:
    """Describe sampling and per-block ROIs; Qwen inserts actual timestamps."""
    full_count = sum((sample.pixel_box is None for sample in samples))
    roi_count = len(samples) - full_count
    spatial_note = (
        f" {roi_count} visual block(s) are motion-focused RGB crops; the other {full_count} block(s) are full RGB frames."
        if roi_count
        else f" All {full_count} visual block(s) are full RGB frames."
    )
    if roi_count and samples[0].spatial_view == "masked_roi":
        spatial_note = f" {roi_count} visual block(s) show motion-focused RGB regions at their full-frame scale and position, with areas outside the regions filled in gray; the other {full_count} block(s) are full RGB frames. Gray areas contain no visual evidence."
    if plan.mode == "tea_sel":
        from dataclasses import replace

        return build_sampling_prompt(
            duration,
            replace(plan, mode=plan.analysis_summary["temporal_mode"]),
            samples,
        )
    if plan.mode in ("tea_only", "sel_only"):
        selection = {
            "tea_only": "RGB observations selected by temporal event activity, without spatial cropping",
            "sel_only": "RGB observations sampled at the centers of equal temporal bins, with event-guided spatial crops",
        }[plan.mode]
        return (
            f"The source video duration is {duration:.2f} seconds. The following native video contains {selection}, in chronological order. The timestamp before each visual block is its true time in the source video.{spatial_note} Use all provided RGB observations and their timestamps.\n"
            + build_roi_block_prompt(samples)
        )
    if plan.mode.startswith("sel_region_"):
        from dataclasses import replace

        return build_sampling_prompt(
            duration,
            replace(plan, mode=plan.analysis_summary["reference_mode"]),
            samples,
        )
    if plan.mode.startswith("tea_context_"):
        from dataclasses import replace

        setting = plan.analysis_summary["tea_context"]["setting"]
        reference_mode = plan.analysis_summary["reference_mode"]
        text = build_sampling_prompt(
            duration, replace(plan, mode=reference_mode), samples
        )
        if setting == "global_only" and (
            not plan.analysis_summary["no_activity_fallback"]
        ):
            text = text.replace(
                "selected non-uniformly in chronological order by an event activity controller",
                "selected uniformly in chronological order, with event-guided spatial regions",
            )
        return text
    return (
        f"The source video duration is {duration:.2f} seconds. The following native video contains RGB visual blocks selected non-uniformly in chronological order by an event activity controller. The timestamp immediately before each visual block is its true time in the source video.{spatial_note} Use the timestamps and all provided RGB evidence; do not assume equal time spacing between adjacent blocks. Sampling mode: {plan.mode}.\n"
        + build_roi_block_prompt(samples)
    )
