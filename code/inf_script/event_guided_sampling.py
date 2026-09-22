"""Use raw events to choose RGB timestamps and spatial crops.

The event modality is a controller only: this module never renders or returns
an event frame.  It converts an H5 stream with rows
``[timestamp, x, y, polarity]`` into a compact sampling plan for an RGB video.
"""

from __future__ import annotations

import math
import os
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

try:
    from decord import VideoReader, cpu
except ImportError:  # pragma: no cover - exercised only in the inference env
    VideoReader = None
    cpu = None


NormalizedBox = Tuple[float, float, float, float]


def _import_h5py():
    try:
        import h5py
        return h5py
    except ImportError as exc:
        raise ImportError("Install h5py in the active environment for event sampling") from exc


@dataclass(frozen=True)
class EventSamplerConfig:
    analysis_bins: int = 96
    spatial_grid_width: int = 16
    spatial_grid_height: int = 12
    sensor_width: int = 240
    sensor_height: int = 180
    timestamp_divisor: float = 1_000_000.0
    activity_threshold: float = 0.35
    min_temporal_contrast: float = 0.08
    merge_gap_bins: int = 1
    min_frames: int = 8
    max_frames: int = 24
    min_global_frames: int = 3
    max_global_frames: int = 6
    global_stride_sec: float = 30.0
    focus_stride_sec: float = 2.5
    fallback_stride_sec: float = 6.0
    roi_mass_fraction: float = 0.80
    roi_margin: float = 0.15
    min_crop_side: float = 0.25
    max_crop_area: float = 0.80
    chunk_events: int = 500_000

    def __post_init__(self) -> None:
        if self.analysis_bins < 4:
            raise ValueError("analysis_bins must be at least 4")
        if self.spatial_grid_width <= 0 or self.spatial_grid_height <= 0:
            raise ValueError("spatial grid dimensions must be positive")
        if self.sensor_width <= 0 or self.sensor_height <= 0:
            raise ValueError("sensor dimensions must be positive")
        if self.timestamp_divisor < 0:
            raise ValueError("timestamp_divisor must be non-negative")
        if not 1 <= self.min_frames <= self.max_frames:
            raise ValueError("expected 1 <= min_frames <= max_frames")
        if not 0.0 < self.roi_mass_fraction <= 1.0:
            raise ValueError("roi_mass_fraction must be in (0, 1]")


@dataclass(frozen=True)
class ActiveSegment:
    start_sec: float
    end_sec: float
    start_bin: int
    end_bin: int
    score_mass: float

    @property
    def duration(self) -> float:
        return max(0.0, self.end_sec - self.start_sec)


@dataclass(frozen=True)
class EventAnalysis:
    duration: float
    event_count: int
    timestamp_divisor: float
    temporal_counts: np.ndarray = field(repr=False)
    spatial_counts: np.ndarray = field(repr=False)
    scores: np.ndarray = field(repr=False)
    active_mask: np.ndarray = field(repr=False)
    segments: Tuple[ActiveSegment, ...]
    temporal_contrast: float

    def summary(self) -> Dict:
        return {
            "duration": round(float(self.duration), 4),
            "event_count": int(self.event_count),
            "timestamp_divisor": float(self.timestamp_divisor),
            "analysis_bins": int(len(self.scores)),
            "temporal_contrast": round(float(self.temporal_contrast), 6),
            "active_bin_ratio": round(float(np.mean(self.active_mask)), 6),
            "active_segments": [asdict(segment) for segment in self.segments],
            "score_min": round(float(np.min(self.scores)) if self.scores.size else 0.0, 6),
            "score_max": round(float(np.max(self.scores)) if self.scores.size else 0.0, 6),
        }


@dataclass(frozen=True)
class FrameRequest:
    timestamp: float
    role: str
    roi: Optional[NormalizedBox] = None
    event_bin: Optional[int] = None

    def as_dict(self) -> Dict:
        data = asdict(self)
        if self.roi is not None:
            data["roi"] = [round(float(value), 6) for value in self.roi]
        data["timestamp"] = round(float(self.timestamp), 4)
        return data


@dataclass(frozen=True)
class SamplingPlan:
    duration: float
    mode: str
    requests: Tuple[FrameRequest, ...]
    analysis_summary: Dict

    def as_dict(self) -> Dict:
        return {
            "duration": round(float(self.duration), 4),
            "mode": self.mode,
            "requested_rgb_frames": len(self.requests),
            "event_frames_sent_to_model": 0,
            "requests": [request.as_dict() for request in self.requests],
            "event_analysis": self.analysis_summary,
        }


@dataclass
class RGBSample:
    image: Image.Image
    timestamp: float
    frame_index: int
    role: str
    roi: Optional[NormalizedBox]
    pixel_box: Optional[Tuple[int, int, int, int]]

    def metadata(self) -> Dict:
        return {
            "timestamp": round(float(self.timestamp), 4),
            "frame_index": int(self.frame_index),
            "role": self.role,
            "roi": list(self.roi) if self.roi is not None else None,
            "pixel_box": list(self.pixel_box) if self.pixel_box is not None else None,
        }


class EventH5Index:
    """Resolve an event H5 by explicit annotation or the RGB filename stem."""

    def __init__(self, root: str):
        self.root = Path(root).expanduser().resolve()
        self.by_stem: Dict[str, List[Path]] = {}
        if self.root.exists():
            for path in self.root.rglob("*"):
                if path.is_file() and path.suffix.lower() in {".h5", ".hdf5"}:
                    self.by_stem.setdefault(path.stem, []).append(path)

    @property
    def indexed_count(self) -> int:
        return sum(len(paths) for paths in self.by_stem.values())

    def resolve(self, rgb_path: str, source: Optional[Dict] = None) -> Optional[str]:
        source = source or {}
        explicit = source.get("event") or source.get("event_path") or source.get("event_h5")
        if isinstance(explicit, list):
            explicit = explicit[0] if explicit else None
        if explicit:
            candidate = Path(str(explicit)).expanduser()
            if not candidate.is_absolute():
                candidate = self.root / candidate
            if candidate.is_file() and candidate.suffix.lower() in {".h5", ".hdf5"}:
                return str(candidate.resolve())

        stem = Path(rgb_path).stem
        candidates = self.by_stem.get(stem, [])
        if not candidates:
            return None
        candidates = sorted(candidates, key=lambda path: (path.parent.name != stem, str(path)))
        return str(candidates[0])


def get_video_metadata(video_path: str) -> Tuple[int, float, float]:
    if VideoReader is None:
        raise ImportError("decord is required for RGB video sampling")
    if not os.path.exists(video_path):
        raise FileNotFoundError(video_path)
    reader = VideoReader(video_path, ctx=cpu(0))
    total_frames = len(reader)
    fps = float(reader.get_avg_fps())
    if total_frames <= 0 or fps <= 0:
        raise ValueError(f"Invalid video metadata: {video_path}")
    return total_frames, fps, total_frames / fps


def _timestamp_divisor(last_timestamp: float, duration: float) -> float:
    if duration <= 0 or last_timestamp <= 0:
        return 1_000_000.0
    candidates = (1.0, 1_000.0, 1_000_000.0, 1_000_000_000.0)
    return min(
        candidates,
        key=lambda divisor: abs(math.log(max(last_timestamp / divisor, 1e-9) / duration)),
    )


def _smooth_1d(values: np.ndarray) -> np.ndarray:
    if len(values) < 3:
        return values.astype(np.float64, copy=True)
    padded = np.pad(values.astype(np.float64), (1, 1), mode="edge")
    return 0.25 * padded[:-2] + 0.50 * padded[1:-1] + 0.25 * padded[2:]


def _fill_short_gaps(mask: np.ndarray, max_gap: int) -> np.ndarray:
    result = mask.astype(bool, copy=True)
    if max_gap <= 0:
        return result
    index = 0
    while index < len(result):
        if result[index]:
            index += 1
            continue
        start = index
        while index < len(result) and not result[index]:
            index += 1
        if start > 0 and index < len(result) and index - start <= max_gap:
            result[start:index] = True
    return result


def _score_histogram(
    spatial_counts: np.ndarray,
    config: EventSamplerConfig,
) -> Tuple[np.ndarray, np.ndarray, float]:
    counts = spatial_counts.sum(axis=(1, 2)).astype(np.float64)
    activity = _smooth_1d(np.log1p(counts))
    low = float(np.percentile(activity, 20))
    high = float(np.percentile(activity, 90))
    spread = max(high - low, 0.0)
    temporal_contrast = spread / max(abs(high), 1e-6)

    if spread <= 1e-8 or temporal_contrast < config.min_temporal_contrast:
        scores = np.zeros_like(activity)
        return scores, np.zeros_like(activity, dtype=bool), temporal_contrast

    temporal = np.clip((activity - low) / spread, 0.0, 1.0)
    flat = spatial_counts.reshape(len(spatial_counts), -1).astype(np.float64)
    top_k = max(1, int(math.ceil(flat.shape[1] * 0.10)))
    top_share = np.zeros(len(flat), dtype=np.float64)
    totals = flat.sum(axis=1)
    valid = totals > 0
    if np.any(valid):
        partitioned = np.partition(flat[valid], flat.shape[1] - top_k, axis=1)
        top_share[valid] = partitioned[:, -top_k:].sum(axis=1) / totals[valid]
    focus = np.clip((top_share - 0.10) / 0.90, 0.0, 1.0)

    # Temporal prominence defines the boundary; spatial concentration is only
    # a small confidence boost so large fires or crowds are not suppressed.
    scores = np.clip(temporal * (0.80 + 0.20 * focus), 0.0, 1.0)
    active = (scores >= config.activity_threshold) & (counts > 0)
    active = _fill_short_gaps(active, config.merge_gap_bins)
    return scores, active, temporal_contrast


def _segments_from_mask(
    active: np.ndarray,
    scores: np.ndarray,
    duration: float,
) -> Tuple[ActiveSegment, ...]:
    bin_duration = duration / max(len(active), 1)
    segments: List[ActiveSegment] = []
    index = 0
    while index < len(active):
        if not active[index]:
            index += 1
            continue
        start = index
        while index + 1 < len(active) and active[index + 1]:
            index += 1
        end = index
        segments.append(
            ActiveSegment(
                start_sec=start * bin_duration,
                end_sec=min(duration, (end + 1) * bin_duration),
                start_bin=start,
                end_bin=end,
                score_mass=float(scores[start : end + 1].sum()),
            )
        )
        index += 1
    return tuple(segments)


def analyze_event_h5(
    path: str,
    duration: float,
    config: EventSamplerConfig,
) -> EventAnalysis:
    """Aggregate a raw H5 stream into temporal scores and a coarse spatial grid."""
    h5py = _import_h5py()
    spatial = np.zeros(
        (config.analysis_bins, config.spatial_grid_height, config.spatial_grid_width),
        dtype=np.int64,
    )
    with h5py.File(path, "r") as handle:
        if "events" not in handle:
            raise KeyError(f"H5 file does not contain an 'events' dataset: {path}")
        events = handle["events"]
        event_count = int(events.shape[0])
        if event_count <= 0:
            scores, active, contrast = _score_histogram(spatial, config)
            return EventAnalysis(
                duration=duration,
                event_count=0,
                timestamp_divisor=config.timestamp_divisor or 1_000_000.0,
                temporal_counts=spatial.sum(axis=(1, 2)),
                spatial_counts=spatial,
                scores=scores,
                active_mask=active,
                segments=(),
                temporal_contrast=contrast,
            )

        if len(events.shape) != 2 or events.shape[1] < 3:
            raise ValueError(f"Expected events [N,4], got {events.shape} in {path}")
        last_timestamp = float(events[event_count - 1, 0])
        divisor = config.timestamp_divisor or _timestamp_divisor(last_timestamp, duration)

        for start in range(0, event_count, config.chunk_events):
            stop = min(event_count, start + config.chunk_events)
            chunk = np.asarray(events[start:stop, :3])
            seconds = chunk[:, 0].astype(np.float64) / divisor
            valid = (seconds >= 0.0) & (seconds <= duration + 1e-6)
            if not np.any(valid):
                continue
            seconds = seconds[valid]
            x = chunk[valid, 1].astype(np.float64)
            y = chunk[valid, 2].astype(np.float64)
            temporal_bin = np.floor(seconds * config.analysis_bins / max(duration, 1e-9)).astype(np.int64)
            temporal_bin = np.clip(temporal_bin, 0, config.analysis_bins - 1)
            grid_x = np.floor(x * config.spatial_grid_width / config.sensor_width).astype(np.int64)
            grid_y = np.floor(y * config.spatial_grid_height / config.sensor_height).astype(np.int64)
            grid_x = np.clip(grid_x, 0, config.spatial_grid_width - 1)
            grid_y = np.clip(grid_y, 0, config.spatial_grid_height - 1)
            flat_index = (
                (temporal_bin * config.spatial_grid_height + grid_y)
                * config.spatial_grid_width
                + grid_x
            )
            spatial += np.bincount(flat_index, minlength=spatial.size).reshape(spatial.shape)

    scores, active, contrast = _score_histogram(spatial, config)
    segments = _segments_from_mask(active, scores, duration)
    return EventAnalysis(
        duration=duration,
        event_count=event_count,
        timestamp_divisor=divisor,
        temporal_counts=spatial.sum(axis=(1, 2)),
        spatial_counts=spatial,
        scores=scores,
        active_mask=active,
        segments=segments,
        temporal_contrast=contrast,
    )


def _bounded_count(value: int, low: int, high: int) -> int:
    return max(low, min(high, int(value)))


def _allocate_counts(weights: Sequence[float], total: int, minimum: int = 1) -> List[int]:
    if not weights or total <= 0:
        return [0] * len(weights)
    counts = [0] * len(weights)
    if total < len(weights) * minimum:
        order = sorted(range(len(weights)), key=lambda idx: weights[idx], reverse=True)
        for idx in order[:total]:
            counts[idx] = 1
        return counts

    counts = [minimum] * len(weights)
    remaining = total - sum(counts)
    weight_array = np.asarray(weights, dtype=np.float64)
    if weight_array.sum() <= 0:
        weight_array[:] = 1.0
    raw = remaining * weight_array / weight_array.sum()
    floors = np.floor(raw).astype(int)
    counts = [count + int(extra) for count, extra in zip(counts, floors)]
    leftover = total - sum(counts)
    order = np.argsort(-(raw - floors))
    for idx in order[:leftover]:
        counts[int(idx)] += 1
    return counts


def _expand_interval(center: float, width: float, minimum: float) -> Tuple[float, float]:
    width = max(width, minimum)
    low = center - width / 2.0
    high = center + width / 2.0
    if low < 0.0:
        high -= low
        low = 0.0
    if high > 1.0:
        low -= high - 1.0
        high = 1.0
    return max(0.0, low), min(1.0, high)


def spatial_roi(
    spatial_counts: np.ndarray,
    event_bin: int,
    config: EventSamplerConfig,
) -> Optional[NormalizedBox]:
    """Return the smallest stable box covering most event mass near one bin."""
    start = max(0, event_bin - 1)
    stop = min(len(spatial_counts), event_bin + 2)
    cells = spatial_counts[start:stop].sum(axis=0).astype(np.float64)
    total = float(cells.sum())
    if total <= 0:
        return None

    flat = cells.ravel()
    order = np.argsort(-flat)
    cumulative = np.cumsum(flat[order])
    keep_count = int(np.searchsorted(cumulative, total * config.roi_mass_fraction) + 1)
    kept = order[:keep_count]
    kept = kept[flat[kept] > 0]
    if len(kept) < 2:
        return None

    rows, cols = np.unravel_index(kept, cells.shape)
    left = float(cols.min()) / config.spatial_grid_width
    right = float(cols.max() + 1) / config.spatial_grid_width
    top = float(rows.min()) / config.spatial_grid_height
    bottom = float(rows.max() + 1) / config.spatial_grid_height

    width = right - left
    height = bottom - top
    left, right = _expand_interval(
        (left + right) / 2.0,
        width * (1.0 + 2.0 * config.roi_margin),
        config.min_crop_side,
    )
    top, bottom = _expand_interval(
        (top + bottom) / 2.0,
        height * (1.0 + 2.0 * config.roi_margin),
        config.min_crop_side,
    )
    if (right - left) * (bottom - top) >= config.max_crop_area:
        return None
    return (left, top, right, bottom)


def _segment_sample_times(
    segment: ActiveSegment,
    scores: np.ndarray,
    duration: float,
    count: int,
) -> List[float]:
    if count <= 0:
        return []
    bins = np.arange(segment.start_bin, segment.end_bin + 1)
    bin_duration = duration / len(scores)
    centers = (bins.astype(np.float64) + 0.5) * bin_duration
    if count == 1:
        return [float(centers[int(np.argmax(scores[bins]))])]

    weights = np.maximum(scores[bins], 1e-6)
    cdf = np.cumsum(weights) / weights.sum()
    quantiles = (np.arange(count, dtype=np.float64) + 0.5) / count
    weighted = np.array([centers[min(int(np.searchsorted(cdf, q)), len(centers) - 1)] for q in quantiles])
    uniform = np.linspace(segment.start_sec, segment.end_sec, count + 2)[1:-1]
    blended = np.sort(0.5 * weighted + 0.5 * uniform)
    return [float(np.clip(value, 0.0, duration)) for value in blended]


def build_sampling_plan(
    analysis: Optional[EventAnalysis],
    duration: float,
    config: EventSamplerConfig,
) -> SamplingPlan:
    """Build a variable-length mixture of global RGB frames and focused crops."""
    if duration <= 0:
        raise ValueError("duration must be positive")

    segments = analysis.segments if analysis is not None else ()
    if not segments:
        count = _bounded_count(
            int(math.ceil(duration / config.fallback_stride_sec)) + 2,
            config.min_frames,
            config.max_frames,
        )
        timestamps = np.linspace(0.0, duration, count)
        requests = tuple(FrameRequest(float(ts), "uniform_fallback") for ts in timestamps)
        summary = analysis.summary() if analysis is not None else {"reason": "event_h5_missing"}
        return SamplingPlan(duration, "uniform_fallback", requests, summary)

    global_count = _bounded_count(
        int(math.ceil(duration / config.global_stride_sec)) + 2,
        config.min_global_frames,
        config.max_global_frames,
    )
    desired_focus = sum(
        max(2, int(math.ceil(segment.duration / config.focus_stride_sec)))
        for segment in segments
    )
    total = _bounded_count(global_count + desired_focus, config.min_frames, config.max_frames)
    focus_count = max(0, total - global_count)

    global_times = np.linspace(0.0, duration, global_count)
    requests: List[FrameRequest] = [
        FrameRequest(float(timestamp), "global_context") for timestamp in global_times
    ]

    weights = [segment.duration * (0.5 + segment.score_mass) for segment in segments]
    allocations = _allocate_counts(weights, focus_count, minimum=2)
    for segment, count in zip(segments, allocations):
        for timestamp in _segment_sample_times(segment, analysis.scores, duration, count):
            event_bin = min(
                len(analysis.scores) - 1,
                max(0, int(timestamp * len(analysis.scores) / duration)),
            )
            requests.append(
                FrameRequest(
                    timestamp=timestamp,
                    role="event_guided_focus",
                    roi=spatial_roi(analysis.spatial_counts, event_bin, config),
                    event_bin=event_bin,
                )
            )

    role_order = {"global_context": 0, "event_guided_focus": 1}
    requests.sort(key=lambda request: (request.timestamp, role_order.get(request.role, 2)))
    return SamplingPlan(
        duration=duration,
        mode="event_guided_temporal_spatial",
        requests=tuple(requests),
        analysis_summary=analysis.summary(),
    )


def _resize_rgb(image: Image.Image, max_edge: int) -> Image.Image:
    image = image.convert("RGB")
    if max(image.size) <= max_edge:
        return image
    scale = max_edge / max(image.size)
    size = (max(1, int(round(image.width * scale))), max(1, int(round(image.height * scale))))
    return image.resize(size, Image.Resampling.LANCZOS)


def extract_rgb_samples(
    video_path: str,
    plan: SamplingPlan,
    max_edge: int = 640,
) -> List[RGBSample]:
    """Materialize only RGB frames/crops requested by a sampling plan."""
    if VideoReader is None:
        raise ImportError("decord is required for RGB video sampling")
    reader = VideoReader(video_path, ctx=cpu(0))
    total_frames = len(reader)
    fps = float(reader.get_avg_fps())
    if total_frames <= 0 or fps <= 0:
        return []

    prepared = []
    seen = set()
    for request in plan.requests:
        frame_index = max(0, min(int(round(request.timestamp * fps)), total_frames - 1))
        roi_key = tuple(round(value, 4) for value in request.roi) if request.roi else None
        key = (frame_index, roi_key)
        if key in seen:
            continue
        seen.add(key)
        prepared.append((request, frame_index))
    if not prepared:
        return []

    unique_indices = sorted({frame_index for _, frame_index in prepared})
    arrays = reader.get_batch(unique_indices).asnumpy()
    frames_by_index = {index: array for index, array in zip(unique_indices, arrays)}

    samples: List[RGBSample] = []
    for request, frame_index in prepared:
        image = Image.fromarray(frames_by_index[frame_index]).convert("RGB")
        pixel_box = None
        role = request.role
        if request.roi is not None:
            left, top, right, bottom = request.roi
            pixel_box = (
                max(0, min(image.width - 1, int(math.floor(left * image.width)))),
                max(0, min(image.height - 1, int(math.floor(top * image.height)))),
                max(1, min(image.width, int(math.ceil(right * image.width)))),
                max(1, min(image.height, int(math.ceil(bottom * image.height)))),
            )
            if pixel_box[2] > pixel_box[0] and pixel_box[3] > pixel_box[1]:
                image = image.crop(pixel_box)
            else:
                pixel_box = None
                role = "event_guided_full_frame"
        elif role == "event_guided_focus":
            role = "event_guided_full_frame"

        samples.append(
            RGBSample(
                image=_resize_rgb(image, max_edge=max_edge),
                timestamp=frame_index / fps,
                frame_index=frame_index,
                role=role,
                roi=request.roi if pixel_box is not None else None,
                pixel_box=pixel_box,
            )
        )
    return samples
