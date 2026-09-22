"""Image SFT data processing for event-guided LoRA training.

The event sampler has already materialized the selected full frames and
spatial crops as image files.  This module intentionally exposes only the
ordinary Qwen-VL SFT fields; event metadata is provenance, not an auxiliary
training target.
"""

import json
import logging
import os
import random
import re
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

import transformers


# Always prefer the qwen-vl-utils copy shipped with this repository.
os.environ.setdefault("FORCE_QWENVL_VIDEO_READER", "decord")
QWEN_UTILS_SRC = Path(__file__).resolve().parents[3] / "qwen-vl-utils" / "src"
if QWEN_UTILS_SRC.exists() and str(QWEN_UTILS_SRC) not in sys.path:
    sys.path.insert(0, str(QWEN_UTILS_SRC))
from qwen_vl_utils import process_vision_info

from . import data_list
from .rope2d import get_rope_index_2, get_rope_index_25, get_rope_index_3


IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = 151655
VIDEO_TOKEN_INDEX = 151656
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_VIDEO_TOKEN = "<video>"

REPO_ROOT = Path(__file__).resolve().parents[3]
LEGACY_REPO_ROOTS = ()
local_rank = None


def rank0_print(*args):
    if local_rank in (None, 0):
        print(*args)


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def resolve_media_path(path: Any, base_path: Path = REPO_ROOT) -> str:
    """Resolve media paths while transparently remapping the old repo root."""

    raw = Path(str(path)).expanduser()
    candidates: List[Path] = []
    if raw.is_absolute():
        for legacy_root in LEGACY_REPO_ROOTS:
            try:
                # Never let an event run silently read the old repository.
                # Return the active-tree spelling even if the file is absent;
                # the subsequent image loader will then report the right path.
                mapped = REPO_ROOT / raw.relative_to(legacy_root)
                return str(mapped.resolve())
            except ValueError:
                continue
        candidates.append(raw)
    else:
        candidates.extend((base_path / raw, REPO_ROOT / raw))

    for candidate in candidates:
        if candidate.exists():
            return str(candidate.resolve())

    # Return the repository-local spelling for legacy paths even when a file
    # is missing, so the resulting error points at the active data tree.
    return str(candidates[0 if raw.is_absolute() and len(candidates) > 1 else -1].resolve())


def _extract_inline_media_tags(
    text: str, base_path: Path, media_type: str
) -> Tuple[str, List[Dict[str, str]]]:
    media_items: List[Dict[str, str]] = []
    pattern = re.compile(rf"<{media_type}>(.*?)</{media_type}>", flags=re.DOTALL)

    def replace_match(match: re.Match) -> str:
        media_path = match.group(1).strip()
        if media_path:
            media_items.append(
                {
                    "type": media_type,
                    media_type: resolve_media_path(media_path, base_path),
                }
            )
        return f"<{media_type}>"

    return pattern.sub(replace_match, text), media_items


def _make_abs_paths(base: Path, files: Any) -> str:
    return resolve_media_path(files, base)


def update_processor_pixels(processor, data_args):
    image_processor = processor.image_processor
    rank0_print("Image min_pixels:", getattr(image_processor, "min_pixels", None))
    rank0_print("Image max_pixels:", getattr(image_processor, "max_pixels", None))

    if hasattr(image_processor, "min_pixels"):
        image_processor.min_pixels = data_args.min_pixels
    if hasattr(image_processor, "max_pixels"):
        image_processor.max_pixels = data_args.max_pixels
    if isinstance(getattr(image_processor, "size", None), dict):
        image_processor.size["shortest_edge"] = data_args.min_pixels
        image_processor.size["longest_edge"] = data_args.max_pixels

    video_processor = getattr(processor, "video_processor", None)
    if video_processor is not None:
        if hasattr(video_processor, "min_pixels"):
            video_processor.min_pixels = data_args.video_min_pixels
        if hasattr(video_processor, "max_pixels"):
            video_processor.max_pixels = data_args.video_max_pixels
        if hasattr(video_processor, "min_frames"):
            video_processor.min_frames = data_args.video_min_frames
        if hasattr(video_processor, "max_frames"):
            video_processor.max_frames = data_args.video_max_frames
        if hasattr(video_processor, "fps"):
            video_processor.fps = data_args.video_fps
    return processor


def _build_messages(item: Dict[str, Any], base_path: Path) -> List[Dict[str, Any]]:
    images = item.get("image") or []
    videos = item.get("video") or []
    if isinstance(images, str):
        images = [images]
    if isinstance(videos, str):
        videos = [videos]

    image_pool = [
        {"type": "image", "image": _make_abs_paths(base_path, image)}
        for image in images
    ]
    video_pool = [
        {"type": "video", "video": _make_abs_paths(base_path, video)}
        for video in videos
    ]

    messages: List[Dict[str, Any]] = []
    for turn in item.get("conversations", []):
        role = "user" if turn.get("from") in ("human", "user") else "assistant"
        text = str(turn.get("value", ""))
        if role == "user":
            content: List[Dict[str, Any]] = []
            text, inline_images = _extract_inline_media_tags(text, base_path, "image")
            text, inline_videos = _extract_inline_media_tags(text, base_path, "video")
            image_pool = inline_images + image_pool
            video_pool = inline_videos + video_pool

            for segment in re.split(r"(<image>|<video>)", text):
                if segment == "<image>":
                    if not image_pool:
                        raise ValueError(
                            f"Too few images for sample {item.get('id', 'unknown')}"
                        )
                    content.append(image_pool.pop(0))
                elif segment == "<video>":
                    if not video_pool:
                        raise ValueError(
                            f"Too few videos for sample {item.get('id', 'unknown')}"
                        )
                    content.append(video_pool.pop(0))
                elif segment.strip():
                    content.append({"type": "text", "text": segment.strip()})
            messages.append({"role": "user", "content": content})
        else:
            messages.append(
                {"role": "assistant", "content": [{"type": "text", "text": text}]}
            )
    return messages


def _apply_video_processor_options(messages: List[Dict[str, Any]], processor) -> None:
    video_processor = getattr(processor, "video_processor", None)
    if video_processor is None:
        return
    options = {
        key: getattr(video_processor, key, None)
        for key in ("min_pixels", "max_pixels", "min_frames", "max_frames", "fps")
    }
    options = {key: value for key, value in options.items() if value is not None}
    for message in messages:
        for content in message.get("content", []):
            if content.get("type") == "video":
                for key, value in options.items():
                    content.setdefault(key, value)


def _make_labels(input_ids: torch.Tensor) -> torch.Tensor:
    labels = torch.full_like(input_ids, IGNORE_INDEX)
    token_ids = input_ids[0].tolist()
    position = 0
    while position < len(token_ids):
        # Qwen chat templates use 77091 as the assistant header and 151645 as
        # the end-of-turn token.  Keep the original repository convention.
        if token_ids[position] == 77091:
            answer_start = position + 2
            answer_end = answer_start
            while answer_end < len(token_ids) and token_ids[answer_end] != 151645:
                answer_end += 1
            if answer_end < len(token_ids):
                labels[0, answer_start : answer_end + 2] = input_ids[
                    0, answer_start : answer_end + 2
                ]
                position = answer_end
        position += 1
    return labels


def preprocess_qwen_visual(sources, processor) -> Dict[str, torch.Tensor]:
    if len(sources) != 1:
        raise ValueError(f"Expected 1 source, got {len(sources)}")
    source = sources[0]
    base_path = Path(source.get("data_path") or REPO_ROOT)
    messages = _build_messages(source, base_path)
    _apply_video_processor_options(messages, processor)
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )

    image_inputs, video_inputs, video_kwargs = process_vision_info(
        messages,
        image_patch_size=16,
        return_video_kwargs=True,
        return_video_metadata=True,
    )
    video_metadatas = None
    if video_inputs is not None:
        video_inputs, video_metadatas = zip(*video_inputs)
        video_inputs = list(video_inputs)
        video_metadatas = list(video_metadatas)

    result = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        video_metadata=video_metadatas,
        do_resize=False,
        return_tensors="pt",
        **video_kwargs,
    )
    input_ids = result["input_ids"]
    if isinstance(input_ids, list):
        input_ids = torch.tensor(input_ids).unsqueeze(0)
    if input_ids.ndim == 1:
        input_ids = input_ids.unsqueeze(0)
    result["input_ids"] = input_ids
    result["labels"] = _make_labels(input_ids)
    return result


class LazySupervisedDataset(Dataset):
    """Lazy image/video dataset matching the original SFT data contract."""

    def __init__(self, processor, data_args):
        super().__init__()
        self.model_type = data_args.model_type
        if self.model_type == "qwen3vl":
            self.get_rope_index = get_rope_index_3
        elif self.model_type == "qwen2.5vl":
            self.get_rope_index = get_rope_index_25
        elif self.model_type == "qwen2vl":
            self.get_rope_index = get_rope_index_2
        else:
            raise ValueError(f"model_type: {self.model_type} not supported")

        if data_args.data_path:
            if str(data_args.data_path).endswith(".jsonl"):
                annotations = read_jsonl(data_args.data_path)
            else:
                with open(data_args.data_path, "r", encoding="utf-8") as handle:
                    annotations = json.load(handle)
            base_path = Path(getattr(data_args, "image_folder", "") or REPO_ROOT)
            for annotation in annotations:
                annotation["data_path"] = str(base_path)
            list_data_dict = annotations
        elif data_args.dataset_use:
            list_data_dict = []
            for data in data_list(data_args.dataset_use.split(",")):
                if data["annotation_path"].endswith(".jsonl"):
                    annotations = read_jsonl(data["annotation_path"])
                else:
                    with open(data["annotation_path"], "r", encoding="utf-8") as handle:
                        annotations = json.load(handle)
                sampling_rate = data.get("sampling_rate", 1.0)
                if sampling_rate < 1.0:
                    annotations = random.sample(annotations, int(len(annotations) * sampling_rate))
                for annotation in annotations:
                    if isinstance(annotation, list):
                        for item in annotation:
                            item["data_path"] = data["data_path"]
                    else:
                        annotation["data_path"] = data["data_path"]
                list_data_dict.extend(annotations)
        else:
            raise ValueError("Provide either --data_path or --dataset_use")

        rank0_print(f"Total training samples: {len(list_data_dict)}")
        random.shuffle(list_data_dict)
        self.processor = update_processor_pixels(processor, data_args)
        self.tokenizer = processor.tokenizer
        self.data_args = data_args
        self.merge_size = getattr(processor.image_processor, "merge_size", 2)
        self.list_data_dict = list_data_dict
        self.item_fn = self._get_packed_item if data_args.data_packing else self._get_item

    def __len__(self):
        return len(self.list_data_dict)

    @property
    def lengths(self):
        return [
            sum(len(conv.get("value", "").split()) for conv in sample.get("conversations", []))
            + (128 if "image" in sample else 0)
            for sample in self.list_data_dict
        ]

    @property
    def modality_lengths(self):
        result = []
        for sample in self.list_data_dict:
            length = sum(len(conv.get("value", "").split()) for conv in sample.get("conversations", []))
            result.append(length if ("image" in sample or "video" in sample) else -length)
        return result

    @property
    def pre_calculated_length(self):
        if self.list_data_dict and "num_tokens" in self.list_data_dict[0]:
            return np.array([sample["num_tokens"] for sample in self.list_data_dict])
        return np.ones(len(self.list_data_dict), dtype=np.int64)

    def __getitem__(self, index) -> Dict[str, torch.Tensor]:
        for attempt in range(3):
            try:
                source = self.list_data_dict[index]
                return self.item_fn([source] if isinstance(source, dict) else source)
            except Exception as exc:
                print(f"[Try #{attempt}] Failed to fetch sample {index}: {exc}")
                if attempt < 2:
                    time.sleep(1)

        # Preserve the original retry behavior for transient/corrupt media.
        next_index = min(index + 1, len(self.list_data_dict) - 1)
        for _ in range(3):
            try:
                source = self.list_data_dict[next_index]
                return self.item_fn([source] if isinstance(source, dict) else source)
            except Exception:
                pass

        source = self.list_data_dict[index]
        return self.item_fn([source] if isinstance(source, dict) else source)

    def _get_item(self, sources) -> Dict[str, torch.Tensor]:
        data_dict = preprocess_qwen_visual(sources, self.processor)
        seq_len = data_dict["input_ids"][0].size(0)
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

    def _get_packed_item(self, sources) -> Dict[str, torch.Tensor]:
        if isinstance(sources, dict):
            sources = [sources]
        if len(sources) == 1:
            return self._get_item(sources)
        items = [self._get_item([source]) for source in sources]
        result = {
            "input_ids": torch.cat([item["input_ids"] for item in items], dim=1),
            "labels": torch.cat([item["labels"] for item in items], dim=1),
            "position_ids": torch.cat([item["position_ids"] for item in items], dim=2),
            "attention_mask": [item["attention_mask"][0] for item in items],
        }
        if any("pixel_values" in item for item in items):
            result["pixel_values"] = torch.cat(
                [item["pixel_values"] for item in items if "pixel_values" in item], dim=0
            )
            result["image_grid_thw"] = torch.cat(
                [item["image_grid_thw"] for item in items if "image_grid_thw" in item], dim=0
            )
        if any("pixel_values_videos" in item for item in items):
            result["pixel_values_videos"] = torch.cat(
                [item["pixel_values_videos"] for item in items if "pixel_values_videos" in item], dim=0
            )
            result["video_grid_thw"] = torch.cat(
                [item["video_grid_thw"] for item in items if "video_grid_thw" in item], dim=0
            )
        return result


def pad_and_cat(tensor_list):
    if not tensor_list:
        return None
    max_length = max(tensor.shape[2] for tensor in tensor_list)
    return torch.cat(
        [
            torch.nn.functional.pad(tensor, (0, max_length - tensor.shape[2]), value=1)
            for tensor in tensor_list
        ],
        dim=1,
    )


@dataclass
class DataCollatorForSupervisedDataset:
    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids = [instance["input_ids"].squeeze(0) for instance in instances]
        labels = [instance["labels"].squeeze(0) for instance in instances]
        position_ids = pad_and_cat([instance["position_ids"] for instance in instances])
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX
        )
        input_ids = input_ids[:, : self.tokenizer.model_max_length]
        labels = labels[:, : self.tokenizer.model_max_length]
        if position_ids is not None:
            position_ids = position_ids[:, :, : self.tokenizer.model_max_length]

        batch: Dict[str, Any] = {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": input_ids.ne(self.tokenizer.pad_token_id),
            "position_ids": position_ids,
        }
        images = [instance["pixel_values"] for instance in instances if "pixel_values" in instance]
        if images:
            batch["pixel_values"] = torch.cat(images, dim=0)
            batch["image_grid_thw"] = torch.cat(
                [instance["image_grid_thw"] for instance in instances if "image_grid_thw" in instance],
                dim=0,
            )
        else:
            batch["pixel_values"] = None
            batch["image_grid_thw"] = None

        videos = [
            instance["pixel_values_videos"]
            for instance in instances
            if "pixel_values_videos" in instance
        ]
        if videos:
            batch["pixel_values_videos"] = torch.cat(videos, dim=0)
            batch["video_grid_thw"] = torch.cat(
                [instance["video_grid_thw"] for instance in instances if "video_grid_thw" in instance],
                dim=0,
            )
        else:
            batch["pixel_values_videos"] = None
            batch["video_grid_thw"] = None
        return batch


@dataclass
class FlattenedDataCollatorForSupervisedDataset(DataCollatorForSupervisedDataset):
    pass


def make_supervised_data_module(processor, data_args) -> Dict[str, Any]:
    train_dataset = LazySupervisedDataset(processor, data_args=data_args)
    collator = (
        FlattenedDataCollatorForSupervisedDataset(processor.tokenizer)
        if data_args.data_flatten or data_args.data_packing
        else DataCollatorForSupervisedDataset(processor.tokenizer)
    )
    return {"train_dataset": train_dataset, "eval_dataset": None, "data_collator": collator}
