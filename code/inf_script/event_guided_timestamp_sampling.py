"""Shared packing logic for timestamp-aware event-guided Qwen3-VL video.

Training and inference must use this module together.  It converts the RGB
samples produced by :mod:`event_guided_sampling` into one non-uniform native
video whose original ``fps`` and ``frames_indices`` let Qwen3-VL construct
text timestamps automatically.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence

import numpy as np
from PIL import Image

from event_guided_sampling import RGBSample, SamplingPlan
from event_guided_video_resolution import preserve_video_resolution, build_roi_block_prompt


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


def build_timestamped_video(
    samples: Sequence[RGBSample],
    total_frames: int,
    fps: float,
    duration: float,
) -> TimestampedVideoInput:
    """Pack event-guided RGB samples into timestamp-safe temporal pairs.

    Every selected image is repeated twice so Qwen's temporal patch size of
    two never pairs unrelated non-uniform samples.  Repeating the original
    frame index makes the processor timestamp exactly ``frame_index / fps``.
    """
    if not samples:
        raise ValueError("Cannot build a timestamped video without RGB samples")
    if total_frames <= 0 or fps <= 0 or duration <= 0:
        raise ValueError("Invalid original video metadata")

    sample_indices = [int(sample.frame_index) for sample in samples]
    if any(index < 0 or index >= total_frames for index in sample_indices):
        raise ValueError("A sampled frame index is outside the original video")
    if any(right < left for left, right in zip(sample_indices, sample_indices[1:])):
        raise ValueError("Event-guided samples must be in chronological order")

    canvas_width = max(sample.image.width for sample in samples)
    canvas_height = max(sample.image.height for sample in samples)
    raw_frame_count = len(samples) * QWEN_TEMPORAL_PATCH_SIZE
    frames = np.empty(
        (raw_frame_count, canvas_height, canvas_width, 3),
        dtype=np.uint8,
    )
    packed_indices: List[int] = []
    patch_timestamps: List[float] = []

    for sample_index, sample in enumerate(samples):
        array = np.asarray(
            _letterbox(sample.image, canvas_width, canvas_height),
            dtype=np.uint8,
        )
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
        # vLLM reads this before constructing transformers.VideoMetadata.
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
    duration: float,
    plan: SamplingPlan,
    samples: Sequence[RGBSample],
) -> str:
    """Describe sampling and per-block ROIs; Qwen inserts actual timestamps."""
    full_count = sum(sample.pixel_box is None for sample in samples)
    roi_count = len(samples) - full_count
    spatial_note = (
        f" {roi_count} visual block(s) are motion-focused RGB crops; the other "
        f"{full_count} block(s) are full RGB frames."
        if roi_count
        else f" All {full_count} visual block(s) are full RGB frames."
    )
    return (
        f"The source video duration is {duration:.2f} seconds. The following native video "
        "contains RGB visual blocks selected non-uniformly in chronological order by an event "
        "activity controller. The timestamp immediately before each visual block is its true "
        "time in the source video."
        f"{spatial_note} Use the timestamps and all provided RGB evidence; do not assume equal "
        f"time spacing between adjacent blocks. Sampling mode: {plan.mode}.\n"
        + build_roi_block_prompt(samples)
    )
