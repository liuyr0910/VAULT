"""Lossless spatial alignment for pre-sampled Qwen3-VL videos.

The duplicated temporal frames must never share a fixed total pixel budget.
Pad spatial dimensions to the patch/merge grid, then disable processor resize.
This also works with the existing uncompressed numpy training caches.
"""

import numpy as np

ROI_BLOCK_MAP_HEADER = "RGB block map (source coordinates):"


def build_roi_block_prompt(samples):
    """Match each native temporal block to its full frame or source ROI."""
    entries = []
    masked = False
    for index, sample in enumerate(samples, start=1):
        sample = sample if isinstance(sample, dict) else sample.metadata()
        if sample.get("pixel_box") is not None:
            left, top, right, bottom = sample["roi"]
            is_masked = sample.get("spatial_view") == "masked_roi"
            masked = masked or is_masked
            view = (
                "RGB motion-focused region at full-frame scale and position"
                if is_masked
                else "RGB motion-focused crop"
            )
            entries.append(
                f"Visual block {index}={view} (normalized region {left:.2f},{top:.2f},{right:.2f},{bottom:.2f})"
            )
        else:
            entries.append(f"Visual block {index}=full RGB frame")
    return (
        ROI_BLOCK_MAP_HEADER
        + " "
        + "; ".join(entries)
        + ". Block numbers follow the timestamped visual blocks in order. Regions are left,top,right,bottom in the original full frame, normalized to [0,1]. "
        + (
            "Full frames provide scene context; focused regions retain full-frame coordinates. Gray areas contain no visual evidence.\n"
            if masked
            else "Full frames provide scene context; motion-focused crops provide additional spatial detail.\n"
        )
    )


def preserve_video_resolution(frames, metadata, spatial_factor=32):
    count, height, width, _ = frames.shape
    aligned_height = (height + spatial_factor - 1) // spatial_factor * spatial_factor
    aligned_width = (width + spatial_factor - 1) // spatial_factor * spatial_factor
    if (height, width) != (aligned_height, aligned_width):
        frames = np.pad(
            frames,
            ((0, 0), (0, aligned_height - height), (0, aligned_width - width), (0, 0)),
            mode="constant",
            constant_values=127,
        )
    frames = np.ascontiguousarray(frames)
    metadata = {**metadata, "height": aligned_height, "width": aligned_width}
    return (frames, metadata)


def resolution_contract(frames):
    count, height, width, _ = frames.shape
    return {
        "policy": "preserve_pixels_pad_to_grid_v1",
        "processor_kwargs": {"do_resize": False, "do_sample_frames": False},
        "video_grid_thw": [count // 2, height // 16, width // 16],
        "visual_tokens": count // 2 * (height // 32) * (width // 32),
        "processor_frame_size": [width, height],
    }
