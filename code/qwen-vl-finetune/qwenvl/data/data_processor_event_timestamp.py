"""SFT dataset processor for pre-sampled Qwen3-VL timestamp videos.

Unlike the ordinary video data path, this loader never calls
``qwen_vl_utils.fetch_video`` and never samples by ``fps``.  It loads the
cached RGB tensor and the original source video's ``fps/frames_indices`` and
passes both directly to ``Qwen3VLProcessor`` with ``do_sample_frames=False``.
"""

from __future__ import annotations

import json
import copy
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch

from . import data_processor_event_lora as base

_inference_dir = str(base.REPO_ROOT / "inf_script")
if _inference_dir not in sys.path:
    sys.path.insert(0, _inference_dir)
from event_guided_video_resolution import (
    preserve_video_resolution, resolution_contract,
    build_roi_block_prompt, ROI_BLOCK_MAP_HEADER,
)


def with_roi_block_prompt(source):
    """Upgrade old cached training prompts without rebuilding RGB arrays."""
    _, metadata_path = _timestamp_cache_paths(source)
    cache = json.loads(metadata_path.read_text())
    note = build_roi_block_prompt(cache["rgb_samples"])
    source = copy.deepcopy(source)
    for turn in source.get("conversations", []):
        if turn.get("from") not in ("user", "human"):
            continue
        text = turn["value"]
        if ROI_BLOCK_MAP_HEADER in text:
            continue
        prefix = cache["prompt"]
        if prefix not in text:
            raise ValueError(f"{source.get('id')}: cached sampling prompt not found")
        turn["value"] = text.replace(prefix, prefix + note, 1)
    return source


def _timestamp_cache_paths(source: Dict[str, Any]) -> tuple[Path, Path]:
    cache_info = source.get("event_guided_timestamp_sampling")
    if not isinstance(cache_info, dict):
        raise ValueError(
            f"Sample {source.get('id', '<unknown>')} lacks event_guided_timestamp_sampling"
        )

    videos = source.get("video") or []
    if isinstance(videos, str):
        videos = [videos]
    if len(videos) != 1:
        raise ValueError(
            f"Timestamp sample {source.get('id', '<unknown>')} must contain exactly one video cache"
        )
    base_path = Path(source.get("data_path") or base.REPO_ROOT)
    array_path = Path(base.resolve_media_path(videos[0], base_path))
    metadata_path = Path(
        base.resolve_media_path(cache_info.get("metadata_path", ""), base_path)
    )
    return array_path, metadata_path


def _load_timestamp_video(source: Dict[str, Any]) -> tuple[np.ndarray, Dict[str, Any]]:
    array_path, metadata_path = _timestamp_cache_paths(source)
    if not array_path.is_file():
        raise FileNotFoundError(array_path)
    if not metadata_path.is_file():
        raise FileNotFoundError(metadata_path)

    with metadata_path.open("r", encoding="utf-8") as handle:
        cache = json.load(handle)
    memory_mapped = np.load(array_path, mmap_mode="r", allow_pickle=False)
    if memory_mapped.ndim != 4 or memory_mapped.shape[-1] != 3:
        raise ValueError(f"Expected NHWC uint8 video, got {memory_mapped.shape}")
    if memory_mapped.dtype != np.uint8:
        raise ValueError(f"Expected uint8 video cache, got {memory_mapped.dtype}")

    frame_indices = [int(index) for index in cache.get("frames_indices", [])]
    if len(frame_indices) != memory_mapped.shape[0]:
        raise ValueError(
            f"frames_indices length {len(frame_indices)} != video frames {memory_mapped.shape[0]}"
        )
    if cache.get("do_sample_frames") is not False:
        raise ValueError("Timestamp video cache must set do_sample_frames=false")

    # Materialize one writable contiguous array for torch.from_numpy. The mmap
    # keeps dataset construction cheap and bounds each worker to one sample.
    frames = np.array(memory_mapped, dtype=np.uint8, copy=True, order="C")
    metadata = {
        "total_num_frames": int(cache["total_num_frames"]),
        "fps": float(cache["fps"]),
        "width": int(cache["width"]),
        "height": int(cache["height"]),
        "duration": float(cache["duration"]),
        "video_backend": "event_guided_presampled_rgb",
        "frames_indices": frame_indices,
    }
    return frames, metadata


def preprocess_qwen_visual(sources, processor) -> Dict[str, torch.Tensor]:
    if len(sources) != 1:
        raise ValueError(f"Expected 1 source, got {len(sources)}")
    source = with_roi_block_prompt(sources[0])
    if source.get("image"):
        raise ValueError("Timestamp SFT records must not contain independent images")

    base_path = Path(source.get("data_path") or base.REPO_ROOT)
    messages = base._build_messages(source, base_path)
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    frames, metadata = _load_timestamp_video(source)
    frames, metadata = preserve_video_resolution(frames, metadata)
    contract = resolution_contract(frames)

    result = processor(
        text=[text],
        videos=[frames],
        video_metadata=[metadata],
        **contract["processor_kwargs"],
        return_tensors="pt",
    )
    if result["video_grid_thw"].tolist() != [contract["video_grid_thw"]]:
        raise ValueError("Processor changed the native video resolution or temporal grid")
    input_ids = result["input_ids"]
    if isinstance(input_ids, list):
        input_ids = torch.tensor(input_ids).unsqueeze(0)
    if input_ids.ndim == 1:
        input_ids = input_ids.unsqueeze(0)
    result["input_ids"] = input_ids
    result["labels"] = base._make_labels(input_ids)
    return result


class LazySupervisedDataset(base.LazySupervisedDataset):
    """Native-resolution timestamp inputs with strict sample validation."""

    def __init__(self, processor, data_args):
        if data_args.data_flatten or data_args.data_packing:
            raise ValueError("Timestamp SFT requires one video per sample; disable data_flatten/data_packing")
        super().__init__(processor, data_args)
        base.rank0_print("Timestamp spatial policy: preserve pixels, pad to 32, no processor resize")

    def __getitem__(self, index):
        # Never replace a long example with a different sample on failure.
        source = self.list_data_dict[index]
        return self.item_fn([source] if isinstance(source, dict) else source)

    def _get_item(self, sources) -> Dict[str, torch.Tensor]:
        data_dict = preprocess_qwen_visual(sources, self.processor)
        seq_len = data_dict["input_ids"][0].size(0)
        if not (data_dict["labels"] != -100).any():
            raise ValueError(f"{sources[0].get('id')}: no supervised assistant tokens")
        if seq_len > self.tokenizer.model_max_length:
            raise ValueError(
                f"{sources[0].get('id')}: {seq_len} tokens exceed model_max_length="
                f"{self.tokenizer.model_max_length}; increase MODEL_MAX_LENGTH. "
                "Refusing to shrink visual inputs or truncate supervision."
            )
        image_grid_thw = data_dict.get("image_grid_thw")
        if image_grid_thw is not None and not isinstance(image_grid_thw, Sequence):
            image_grid_thw = [image_grid_thw]
        video_grid_thw = data_dict.get("video_grid_thw")
        if video_grid_thw is not None and not isinstance(video_grid_thw, Sequence):
            video_grid_thw = [video_grid_thw]

        second_per_grid_ts = None
        if video_grid_thw:
            video_processor = self.processor.video_processor
            second_per_grid_ts = [
                video_processor.temporal_patch_size / video_processor.fps
            ] * len(video_grid_thw)
        position_ids, _ = self.get_rope_index(
            self.merge_size,
            data_dict["input_ids"],
            image_grid_thw=torch.cat(image_grid_thw, dim=0) if image_grid_thw else None,
            video_grid_thw=torch.cat(video_grid_thw, dim=0) if video_grid_thw else None,
            second_per_grid_ts=second_per_grid_ts,
        )
        data_dict["position_ids"] = position_ids
        data_dict["attention_mask"] = [seq_len]
        return data_dict


def make_supervised_data_module(processor, data_args) -> Dict[str, Any]:
    train_dataset = LazySupervisedDataset(processor, data_args=data_args)
    collator = base.DataCollatorForSupervisedDataset(processor.tokenizer)
    return {
        "train_dataset": train_dataset,
        "eval_dataset": None,
        "data_collator": collator,
    }
