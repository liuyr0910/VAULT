"""Adaptive event images supplementing timestamp RGB.

H5 contract: v2e events=[timestamp_us,x,y,polarity], source-aligned 240x180.
Window candidates are source-frame periods, not assumed sensor microseconds.
"""

from dataclasses import dataclass, asdict, field, replace
from pathlib import Path
import json
import math
import numpy as np
from PIL import Image
from event_guided_sampling import _import_h5py, EventSamplerConfig, analyze_event_h5

POLICY = "eventvault_event_images_score_ranked"
SCORE_FIELDS = (
    "analysis_bins",
    "spatial_grid_width",
    "spatial_grid_height",
    "activity_threshold",
    "min_temporal_contrast",
    "merge_gap_bins",
)


@dataclass(frozen=True)
class EventImageConfig:
    max_groups: int = 3
    score_config: dict = field(
        default_factory=lambda: {
            k: getattr(EventSamplerConfig(), k) for k in SCORE_FIELDS
        }
    )
    frame_windows: tuple = (3, 6, 12, 24, 48)
    target_events_per_pixel: float = 1.0
    representation: str = "gray_pair"
    sensor_width: int = 240
    sensor_height: int = 180
    timestamp_divisor: float = 1000000.0
    max_read_events: int = 4000000


def load_config(path):
    data = json.loads(Path(path).read_text())
    data = data.get("config", data)
    return EventImageConfig(**data)


def bind_sampler_config(config, sampler):
    """Use the RGB sampler's scoring settings for event anchor selection."""
    return replace(config, score_config={k: getattr(sampler, k) for k in SCORE_FIELDS})


def anchor_indices(samples, max_groups, scores, duration, fps):
    """Select score-ranked distinct time bins and retain chronological order."""
    if not max_groups or not samples:
        return []
    pool = []
    for i, sample in enumerate(samples):
        time = int(sample["frame_index"]) / fps
        if not 0 < time <= duration:
            continue
        event_bin = min(len(scores) - 1, int(time * len(scores) / duration))
        pool.append((i, event_bin, time, float(scores[event_bin])))
    pool.sort(key=lambda item: (-item[3], item[2], item[0]))
    selected = []
    seen = set()
    for i, event_bin, time, score in pool:
        if event_bin in seen:
            continue
        selected.append(i)
        seen.add(event_bin)
        if len(selected) == max_groups:
            break
    return sorted(selected, key=lambda i: (samples[i]["frame_index"], i))


class EventReader:
    def __init__(self, path, config, duration):
        self.config = config
        self.handle = _import_h5py().File(path, "r")
        self.events = self.handle["events"]

    def close(self):
        self.handle.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def bound(self, time):
        lo, hi = (0, len(self.events))
        value = time * self.config.timestamp_divisor
        while lo < hi:
            mid = (lo + hi) // 2
            if float(self.events[mid, 0]) < value:
                lo = mid + 1
            else:
                hi = mid
        return lo

    def chunks(self, start, end, roi):
        lo, hi = (self.bound(start), self.bound(end))
        c = self.config
        l, t, r, b = roi
        for offset in range(lo, hi, c.max_read_events):
            e = np.asarray(
                self.events[offset : min(hi, offset + c.max_read_events)],
                dtype=np.float64,
            )
            mask = (e[:, 1] >= l) & (e[:, 1] < r) & (e[:, 2] >= t) & (e[:, 2] < b)
            e = e[mask]
            e[:, 0] /= c.timestamp_divisor
            e[:, 1] -= l
            e[:, 2] -= t
            yield e

    def read(self, start, end, roi):
        pieces = list(self.chunks(start, end, roi))
        return np.concatenate(pieces) if pieces else np.empty((0, 4))


def sensor_roi(sample, c):
    box = sample.get("roi") if sample.get("pixel_box") is not None else None
    box = box or [0, 0, 1, 1]
    l, t, r, b = box
    return [
        math.floor(l * c.sensor_width),
        math.floor(t * c.sensor_height),
        math.ceil(r * c.sensor_width),
        math.ceil(b * c.sensor_height),
    ]


def histogram(e, start, end, shape):
    h, w = shape
    out = np.zeros((3, 2, h, w), np.float32)
    if len(e):
        bins = np.minimum(2, ((e[:, 0] - start) / (end - start) * 3).astype(int))
        np.add.at(
            out,
            (bins, (e[:, 3] > 0).astype(int), e[:, 2].astype(int), e[:, 1].astype(int)),
            1,
        )
    return out


def stats(hist):
    counts = hist.sum(axis=(1, 2, 3))
    occupancy = (hist.sum(axis=1) > 0).mean(axis=(1, 2))
    return dict(
        counts=counts.astype(int).tolist(),
        occupancy=occupancy.tolist(),
        empty_bins=int((counts == 0).sum()),
        events_per_pixel=float(counts.sum() / np.prod(hist.shape[-2:])),
    )


def candidates(reader, sample, fps, c):
    end = int(sample["frame_index"]) / fps
    roi = sensor_roi(sample, c)
    h, w = (roi[3] - roi[1], roi[2] - roi[0])
    if end <= 0:
        return []
    starts = [max(0, end - n / fps) for n in c.frame_windows]
    histograms = [np.zeros((3, 2, h, w), np.float32) for _ in starts]
    for events in reader.chunks(min(starts), end, roi):
        for start, hist in zip(starts, histograms):
            part = events[events[:, 0] >= start]
            hist += histogram(part, start, end, (h, w))
    return [
        (
            dict(frame_periods=n, start=start, end=end, roi_sensor=roi, **stats(hist)),
            hist,
        )
        for n, start, hist in zip(c.frame_windows, starts, histograms)
    ]


def choose(options, c):
    for meta, hist in options:
        if meta["events_per_pixel"] >= c.target_events_per_pixel and (
            not meta["empty_bins"]
        ):
            return (meta, hist)
    return options[-1]


def render(hist, representation):
    positive = hist[hist > 0]
    scale = max(1.0, float(np.percentile(positive, 99))) if len(positive) else 1.0
    v = np.minimum(1, np.log1p(hist) / np.log1p(scale))
    images = []
    for plane in v:
        if representation == "gray_pair":
            neg, pos = plane
            h, w = pos.shape
            gray = np.full((h, 2 * w + 16), 255, np.uint8)
            gray[:, :w] = np.rint(255 * (1 - pos)).astype(np.uint8)
            gray[:, w + 16 :] = np.rint(255 * (1 - neg)).astype(np.uint8)
            rgb = np.repeat(gray[:, :, None], 3, axis=2)
            bg = 255
        else:
            h, w = plane.shape[1:]
            rgb = np.zeros((h, w, 3), np.uint8)
            rgb[:, :, 0] = np.rint(plane[1] * 255).astype(np.uint8)
            rgb[:, :, 2] = np.rint(plane[0] * 255).astype(np.uint8)
            bg = 0
        h, w = rgb.shape[:2]
        rgb = np.pad(rgb, ((0, -h % 32), (0, -w % 32), (0, 0)), constant_values=bg)
        images.append(Image.fromarray(rgb))
    return (images, scale)


def build_bundle(event_path, samples, fps, duration, c):
    meta = dict(
        policy=POLICY,
        config=asdict(c),
        groups=[],
        status="disabled" if c.max_groups == 0 else "missing_event",
    )
    if not c.max_groups:
        return ([], meta)
    images = []
    reader = EventReader(event_path, c, duration)
    with reader:
        sampler = EventSamplerConfig(
            **c.score_config,
            sensor_width=c.sensor_width,
            sensor_height=c.sensor_height,
            timestamp_divisor=c.timestamp_divisor,
        )
        analysis = analyze_event_h5(event_path, duration, sampler)
        meta["anchor_selection"] = dict(
            method="top_score_distinct_bins",
            scores=analysis.scores.tolist(),
            analysis=analysis.summary(),
            score_config=c.score_config,
        )
        for i in anchor_indices(samples, c.max_groups, analysis.scores, duration, fps):
            options = candidates(reader, samples[i], fps, c)
            if not options:
                continue
            chosen, hist = choose(options, c)
            group_images, scale = render(hist, c.representation)
            group = {
                **chosen,
                "rgb_block": i + 1,
                "rgb_frame_index": int(samples[i]["frame_index"]),
                "roi_normalized": samples[i].get("roi")
                if samples[i].get("pixel_box") is not None
                else None,
                "scale_count_p99": scale,
                "image_sizes": [list(im.size) for im in group_images],
                "target_reached": chosen["events_per_pixel"]
                >= c.target_events_per_pixel,
                "candidate_statistics": [m for m, _ in options],
            }
            event_bin = min(
                len(analysis.scores) - 1,
                int(
                    int(samples[i]["frame_index"])
                    / fps
                    * len(analysis.scores)
                    / duration
                ),
            )
            group.update(
                anchor_event_bin=event_bin,
                anchor_score=float(analysis.scores[event_bin]),
            )
            meta["groups"].append(group)
            images.extend(group_images)
    meta.update(
        status="ok" if images else "no_eligible_event_anchor",
        image_count=len(images),
        visual_tokens=sum((im.width * im.height // 1024 for im in images)),
    )
    return (images, meta)


def event_content(meta, images):
    if not images:
        return []
    rep = meta["config"]["representation"]
    legend = (
        "Each image has two grayscale panels: LEFT=positive brightness-change events, RIGHT=negative events. Darker means more events; white means no events. Panels show the SAME region, not different places."
        if rep == "gray_pair"
        else "Red=positive brightness-change events, blue=negative events; brighter means more events; black means no events."
    )
    content = [
        dict(
            type="text",
            text="Supplementary event observations. "
            + legend
            + " These are event count maps, not natural photographs or object colors. Brightness change can also come from camera motion or lighting; it does not by itself establish object motion or anomaly. Each group uses a shared intensity scale. Read actual intervals; do not assume equal time gaps between groups. RGB remains the source of appearance and scene context.",
        )
    ]
    k = 0
    for g in meta["groups"]:
        region = g["roi_normalized"] or [0, 0, 1, 1]
        edges = np.linspace(g["start"], g["end"], 4)
        for j, name in enumerate(("early", "middle", "late")):
            text = f"Event for RGB block {g['rgb_block']}, region={','.join((f'{x:.4f}' for x in region))}, {name} interval=[{edges[j]:.6f},{edges[j + 1]:.6f}) seconds; events={g['counts'][j]}."
            content.extend(
                [dict(type="text", text=text), dict(type="image", image=images[k])]
            )
            k += 1
    return content


def augment_messages(messages, meta, images):
    """Insert after existing video/sampling+ROI note, before task; preserve text."""
    import copy
    from event_guided_video_resolution import ROI_BLOCK_MAP_HEADER

    result = copy.deepcopy(messages)
    extra = event_content(meta, images)
    if not extra:
        return result
    users = [m for m in result if m["role"] == "user"]
    content = users[0]["content"]
    new = []
    for item in content:
        if item["type"] == "text" and ROI_BLOCK_MAP_HEADER in item["text"]:
            text = item["text"]
            idx = text.find("\n", text.index(ROI_BLOCK_MAP_HEADER))
            if idx < 0:
                idx = len(text)
            else:
                idx += 1
            new.append(dict(type="text", text=text[:idx]))
            new.extend(extra)
            if text[idx:]:
                new.append(dict(type="text", text=text[idx:]))
        else:
            new.append(item)
    users[0]["content"] = new
    return result


def processor_inputs(processor, messages, frames, metadata, images, training=False):
    from event_guided_video_resolution import preserve_video_resolution

    frames, metadata = preserve_video_resolution(frames, metadata)
    metadata = dict(metadata)
    metadata.pop("do_sample_frames", None)
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=not training
    )
    result = processor(
        text=[text],
        videos=[frames],
        video_metadata=[metadata],
        images=images or None,
        do_resize=False,
        do_sample_frames=False,
        return_tensors="pt",
    )
    return result
