"""Redistribute only full RGB frames; preserve crops and all event anchors.

Operate AFTER baseline RGB extraction/deduplication and event rendering.
Never rescore events or regenerate their images from the redistributed samples.
"""

import copy
import numpy as np
from PIL import Image
from event_guided_sampling import (
    RGBSample,
    SamplingPlan,
    FrameRequest,
    VideoReader,
    cpu,
    _resize_rgb,
)
from event_guided_timestamp_sampling import _letterbox, TimestampedVideoInput

POLICY = "eventvault_rgb_full_coverage"


def selection(samples, event_meta, total_frames):
    """Return (old block index, new frame index), stably sorted by source time.

    Duplicated frame positions remain distinct blocks. Pinned full-frame anchors
    replace their nearest uniform target. All other full-frame slots use the
    remaining targets; no crop is removed or synthesized.
    """
    pinned = {int(g["rgb_block"]) - 1 for g in event_meta["groups"]}
    full = [i for i, s in enumerate(samples) if s.get("pixel_box") is None]
    targets = np.linspace(0, total_frames - 1, len(full)).tolist()
    for i in full:
        if i in pinned:
            j = min(
                range(len(targets)),
                key=lambda j: abs(targets[j] - samples[i]["frame_index"]),
            )
            targets.pop(j)
    free = iter((int(round(t)) for t in targets))
    chosen = [
        (i, int(s["frame_index"]) if i not in full or i in pinned else next(free))
        for i, s in enumerate(samples)
    ]
    return sorted(chosen, key=lambda pair: (pair[1], pair[0]))


def redistribute(
    video_path, samples, frames, metadata, event_meta, canvas_size, max_edge
):
    """Keep retained pixels exactly; render new full frames on baseline canvas."""
    chosen = selection(samples, event_meta, int(metadata["total_num_frames"]))
    changed = [(i, frame) for i, frame in chosen if frame != samples[i]["frame_index"]]
    arrays = {}
    if changed:
        reader = VideoReader(str(video_path), ctx=cpu(0), num_threads=1)
        indices = sorted({frame for _, frame in changed})
        arrays = dict(zip(indices, reader.get_batch(indices).asnumpy()))
    output = np.empty_like(frames)
    result = []
    for block, (old, frame) in enumerate(chosen):
        sample = copy.deepcopy(samples[old])
        if frame == sample["frame_index"]:
            output[2 * block : 2 * block + 2] = frames[2 * old : 2 * old + 2]
        else:
            image = _letterbox(
                _resize_rgb(Image.fromarray(arrays[frame]), max_edge), *canvas_size
            )
            a = np.asarray(image)
            output[2 * block : 2 * block + 2] = 127
            output[2 * block : 2 * block + 2, : a.shape[0], : a.shape[1]] = a
            sample.update(
                frame_index=frame,
                timestamp=round(frame / metadata["fps"], 4),
                role="coverage_full_frame",
                roi=None,
                pixel_box=None,
            )
        result.append(sample)
    remap = {old + 1: new + 1 for new, (old, _) in enumerate(chosen)}
    event = copy.deepcopy(event_meta)
    for group in event["groups"]:
        group["rgb_block"] = remap[group["rgb_block"]]
    meta = copy.deepcopy(metadata)
    meta["frames_indices"] = [int(s["frame_index"]) for s in result for _ in range(2)]
    provenance = dict(
        policy=POLICY,
        old_to_new_rgb_blocks=remap,
        moved_full_frames=len(changed),
        full_frames=sum((s.get("pixel_box") is None for s in samples)),
        crop_frames=sum((s.get("pixel_box") is not None for s in samples)),
    )
    return (result, output, meta, event, provenance)


def make_plan(duration, samples, baseline_analysis):
    return SamplingPlan(
        duration,
        POLICY,
        tuple((FrameRequest(s["timestamp"], s["role"], s.get("roi")) for s in samples)),
        baseline_analysis,
    )


def apply_inference(video_path, samples, video, plan, event_meta, max_edge):
    original = [s.metadata() for s in samples]
    canvas = (
        max((s.image.width for s in samples)),
        max((s.image.height for s in samples)),
    )
    rows, frames, meta, event, provenance = redistribute(
        video_path, original, video.frames, video.metadata, event_meta, canvas, max_edge
    )
    rgb = [
        RGBSample(
            Image.fromarray(frames[2 * i]),
            s["timestamp"],
            s["frame_index"],
            s["role"],
            s.get("roi"),
            s.get("pixel_box"),
        )
        for i, s in enumerate(rows)
    ]
    updated = TimestampedVideoInput(
        frames,
        meta,
        [s["frame_index"] for s in rows],
        [s["frame_index"] / meta["fps"] for s in rows],
        video.canvas_width,
        video.canvas_height,
    )
    new_plan = make_plan(meta["duration"], rows, plan.as_dict()["event_analysis"])
    return (rgb, updated, new_plan, event, provenance)
