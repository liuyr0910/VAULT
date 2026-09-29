# Adapted for EventVAULT training and multimodal data inputs.
"""Load EventVAULT RGB caches, event images, and supervised chat answers."""

import json
import random
import re
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict
import numpy as np
import torch
import transformers
from torch.utils.data import Dataset
from PIL import Image
from .rope2d import get_rope_index_3

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "inf_script"))
from event_representation import augment_messages, processor_inputs

IGNORE_INDEX = -100


def rank0_print(*args):
    print(*args)


def _build_messages(source):
    video = source["video"][0]
    messages = []
    for turn in source["conversations"]:
        role = "user" if turn["from"] in ("user", "human") else "assistant"
        content = []
        for part in re.split(r"(<video>)", turn["value"]):
            if part == "<video>":
                content.append(dict(type="video", video=video))
            elif part.strip():
                content.append(dict(type="text", text=part.strip()))
        messages.append(dict(role=role, content=content))
    return messages


def load_source(source):
    rgb = json.loads(
        Path(source["event_guided_timestamp_sampling"]["metadata_path"]).read_text()
    )
    event = json.loads(
        Path(source["event_image_supplement"]["metadata_path"]).read_text()
    )
    frames = np.array(
        np.load(source["video"][0], mmap_mode="r"), dtype=np.uint8, copy=True, order="C"
    )
    metadata = {
        key: rgb[key]
        for key in (
            "total_num_frames",
            "fps",
            "width",
            "height",
            "duration",
            "frames_indices",
        )
    }
    metadata["video_backend"] = "event_guided_presampled_rgb"
    images = []
    for path in event["image_paths"]:
        with Image.open(path) as image:
            images.append(image.convert("RGB"))
    messages = augment_messages(_build_messages(source), event, images)
    return messages, frames, metadata, images


def preprocess_qwen_visual(sources, processor):
    messages, frames, metadata, images = load_source(sources[0])
    result = processor_inputs(
        processor, messages, frames, metadata, images, training=True
    )
    result["labels"] = _make_labels(result["input_ids"])
    return result


class LazySupervisedDataset(Dataset):
    def __init__(self, processor, data_args):
        self.list_data_dict = json.loads(Path(data_args.data_path).read_text())
        random.shuffle(self.list_data_dict)
        self.processor = update_processor_pixels(processor, data_args)
        self.tokenizer = processor.tokenizer
        self.merge_size = processor.image_processor.merge_size
        self.get_rope_index = get_rope_index_3

    def __len__(self):
        return len(self.list_data_dict)

    @property
    def lengths(self):
        return [
            sum(len(turn["value"].split()) for turn in sample["conversations"])
            for sample in self.list_data_dict
        ]

    @property
    def modality_lengths(self):
        return self.lengths

    @property
    def pre_calculated_length(self):
        return np.ones(len(self), dtype=np.int64)

    def __getitem__(self, index):
        return self._get_item([self.list_data_dict[index]])

    def _get_item(self, sources):
        data = preprocess_qwen_visual(sources, self.processor)
        length = data["input_ids"].shape[-1]
        seconds = [
            self.processor.video_processor.temporal_patch_size
            / self.processor.video_processor.fps
        ] * len(data["video_grid_thw"])
        positions, _ = self.get_rope_index(
            self.merge_size,
            data["input_ids"],
            image_grid_thw=data.get("image_grid_thw"),
            video_grid_thw=data["video_grid_thw"],
            second_per_grid_ts=seconds,
        )
        data["position_ids"] = positions
        data["attention_mask"] = [length]
        return data


def make_supervised_data_module(processor, data_args):
    return dict(
        train_dataset=LazySupervisedDataset(processor, data_args),
        eval_dataset=None,
        data_collator=DataCollatorForSupervisedDataset(processor.tokenizer),
    )


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


def _make_labels(input_ids: torch.Tensor) -> torch.Tensor:
    labels = torch.full_like(input_ids, IGNORE_INDEX)
    token_ids = input_ids[0].tolist()
    position = 0
    while position < len(token_ids):
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


def pad_and_cat(tensor_list):
    if not tensor_list:
        return None
    max_length = max((tensor.shape[2] for tensor in tensor_list))
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
        images = [
            instance["pixel_values"]
            for instance in instances
            if "pixel_values" in instance
        ]
        if images:
            batch["pixel_values"] = torch.cat(images, dim=0)
            batch["image_grid_thw"] = torch.cat(
                [
                    instance["image_grid_thw"]
                    for instance in instances
                    if "image_grid_thw" in instance
                ],
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
                [
                    instance["video_grid_thw"]
                    for instance in instances
                    if "video_grid_thw" in instance
                ],
                dim=0,
            )
        else:
            batch["pixel_values_videos"] = None
            batch["video_grid_thw"] = None
        return batch
