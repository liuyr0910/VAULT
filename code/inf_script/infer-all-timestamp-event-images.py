#!/usr/bin/env python3
"""Timestamp RGB baseline plus aligned, adaptive-window event image groups.
RGB sampler, resolution policy, task prompts, decoding and evaluation schema
are inherited from infer-all-event-guided-sampling-timestamp.py. No arrows.
Use --event-image-config to select the train-calibrated rendering contract.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from tqdm import tqdm
from event_rgb_visualization import (load_config, build_bundle, augment_messages, processor_inputs, bind_sampler_config, POLICY)
EVENT_IMAGE_CONFIG = None
ACTIVE_EVENT_IMAGES = []
ACTIVE_EVENT_META = None
from event_guided_video_resolution import preserve_video_resolution, resolution_contract

from event_guided_sampling import (
    EventH5Index,
    EventSamplerConfig,
    RGBSample,
    SamplingPlan,
    analyze_event_h5,
    build_sampling_plan,
    extract_rgb_samples,
    get_video_metadata,
)
from event_guided_timestamp_sampling import (
    QWEN_TEMPORAL_PATCH_SIZE,
    TimestampedVideoInput,
    build_sampling_prompt,
    build_timestamped_video,
)


CATEGORY_LIST = [
    "traffic accident", "building collapse", "fighting", "hijack", "environmental pollution",
    "fire", "burglary", "marine accident", "vandalism", "facility malfunction",
    "illegal hunting", "robbery", "riot", "shooting", "natural hazard", "normal",
    "theft", "traffic violation", "object falling", "people falling", "explosion",
    "abuse", "assault", "arrest", "drowning", "animal aggression",
]


def _gpu_memory_snapshot() -> List[Dict[str, float]]:
    """Read GPU memory before torch/vLLM creates a CUDA context."""
    command = [
        "nvidia-smi",
        "--query-gpu=index,memory.total,memory.free",
        "--format=csv,noheader,nounits",
    ]
    try:
        output = subprocess.check_output(command, text=True, stderr=subprocess.STDOUT)
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(f"Unable to query GPU memory with nvidia-smi: {exc}") from exc

    snapshot: List[Dict[str, float]] = []
    for line in output.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 3:
            continue
        snapshot.append(
            {
                "index": int(fields[0]),
                "total_mib": float(fields[1]),
                "free_mib": float(fields[2]),
            }
        )
    if not snapshot:
        raise RuntimeError("nvidia-smi returned no usable GPU memory rows")
    return snapshot


def _resolve_visible_devices(
    requested: str,
    tensor_parallel_size: int,
    gpu_memory_utilization: float,
) -> tuple[List[str], List[Dict[str, float]]]:
    snapshot = _gpu_memory_snapshot()
    by_index = {int(item["index"]): item for item in snapshot}

    if requested.strip().lower() == "auto":
        eligible = [
            item
            for item in snapshot
            if item["free_mib"] >= item["total_mib"] * gpu_memory_utilization
        ]
        eligible.sort(key=lambda item: item["free_mib"], reverse=True)
        if len(eligible) < tensor_parallel_size:
            status = ", ".join(
                f"GPU{int(item['index'])}: {item['free_mib']:.0f}/{item['total_mib']:.0f} MiB free"
                for item in snapshot
            )
            raise RuntimeError(
                f"Need {tensor_parallel_size} GPU(s) with at least "
                f"gpu_memory_utilization={gpu_memory_utilization:.2f}, but found "
                f"{len(eligible)}. Current memory: {status}"
            )
        selected = [str(int(item["index"])) for item in eligible[:tensor_parallel_size]]
        return selected, snapshot

    selected = [device.strip() for device in requested.split(",") if device.strip()]
    if not selected:
        raise ValueError("cuda-visible-devices must be 'auto' or contain at least one GPU id")
    if tensor_parallel_size > len(selected):
        raise ValueError(
            "tensor-parallel-size cannot exceed the number of cuda-visible-devices: "
            f"{tensor_parallel_size} > {len(selected)}"
        )
    for device in selected:
        if not device.isdigit() or int(device) not in by_index:
            raise ValueError(f"Unknown physical GPU id: {device}")
    for device in selected[:tensor_parallel_size]:
        item = by_index[int(device)]
        required = item["total_mib"] * gpu_memory_utilization
        if item["free_mib"] < required:
            raise RuntimeError(
                f"GPU{device} has {item['free_mib']:.0f} MiB free, but "
                f"gpu_memory_utilization={gpu_memory_utilization:.2f} requires "
                f"about {required:.0f} MiB. Choose another GPU or lower the utilization."
            )
    return selected, snapshot

@dataclass(frozen=True)
class OfflineVLLMConfig:
    model: str
    tensor_parallel_size: int
    gpu_memory_utilization: float
    max_model_len: int
    enforce_eager: bool
    video_min_pixels: int
    video_max_pixels: int
    max_tokens: int
    temperature: float
    max_frames: int = 24
    max_edge: int = 640


def build_engine_kwargs(config: OfflineVLLMConfig) -> Dict:
    """Profile the actual single-video workload, without resizing its pixels."""
    edge = (config.max_edge + 31) // 32 * 32
    raw_frames = config.max_frames * QWEN_TEMPORAL_PATCH_SIZE
    return {
        "model": config.model,
        "tensor_parallel_size": config.tensor_parallel_size,
        "gpu_memory_utilization": config.gpu_memory_utilization,
        "max_model_len": config.max_model_len,
        "max_num_seqs": 1,
        "max_num_batched_tokens": config.max_model_len,
        "limit_mm_per_prompt": {
            "image": {"count": EVENT_IMAGE_CONFIG.max_groups * 3, "width": ((EVENT_IMAGE_CONFIG.sensor_width*2+16+31)//32)*32, "height": ((EVENT_IMAGE_CONFIG.sensor_height+31)//32)*32},
            "video": {"count": 1, "num_frames": raw_frames, "width": edge, "height": edge},
        },
        # The size bound is for vLLM's dummy-video construction. Real inputs
        # are already padded and must never be resized or sampled again.
        "mm_processor_kwargs": {
            "do_resize": False,
            "do_sample_frames": False,
            "size": {"shortest_edge": 32 * 32, "longest_edge": raw_frames * edge * edge},
        },
        "enforce_eager": config.enforce_eager,
        "seed": 0,
    }

class PromptFactory:
    @staticmethod
    def task_prompt(task_type: str, data: Dict, vqa_item: Optional[Dict] = None) -> str:
        if task_type == "d":
            return (
                "Please give a concise description of the anomaly in the video. "
                "If there is no anomaly, just describe the normal event."
            )
        if task_type == "c":
            categories = ", ".join(f'"{category}"' for category in CATEGORY_LIST)
            return (
                "Classify the video content into one or multiple of the following categories:\n"
                f"[{categories}]\n"
                "Respond with the category name(s) only, separated by commas if multiple."
            )
        if task_type == "t":
            return (
                "Identify the start and end timestamps (in seconds) for all distinct anomalous "
                "events in the video. If multiple anomalies occur, separate the time intervals "
                "with commas. Format strictly as numbers: start_sec - end_sec, start_sec - end_sec. "
                "If the video contains no anomalies (is normal), output -1 - -1."
            )
        if task_type == "vqa":
            if not vqa_item:
                return "Error: no VQA item provided."
            options = "\n".join(
                f"{key}: {value}" for key, value in sorted(vqa_item.get("options", {}).items())
            )
            return (
                f"Question: {vqa_item.get('question', '')}\n"
                f"Options:\n{options}\n"
                "Respond with the correct option letter (e.g., A)."
            )
        return "Describe the video content concisely."


class CircularManager:
    @staticmethod
    def get_variants(options: Dict, correct_key: str) -> List[Dict]:
        keys = sorted(options)
        if correct_key not in options:
            return []
        values = [options[key] for key in keys]
        correct_value = options[correct_key]
        variants = []
        for shift_id in range(len(keys)):
            shifted = values[shift_id:] + values[:shift_id]
            variants.append(
                {
                    "options": {keys[index]: shifted[index] for index in range(len(keys))},
                    "correct_key": keys[shifted.index(correct_value)],
                    "shift_id": shift_id,
                }
            )
        return variants


class OfflineVLLMInferer:
    """Run vLLM with a per-sample ``(frames, VideoMetadata)`` input."""

    def __init__(self, config: OfflineVLLMConfig, dry_run: bool = False):
        self.config = config
        self.dry_run = dry_run
        self.processor = None
        self.llm = None
        self.sampling_params = None
        if dry_run:
            return

        os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
        from transformers import AutoProcessor
        from vllm import LLM, SamplingParams

        self.processor = AutoProcessor.from_pretrained(config.model)
        self.llm = LLM(**build_engine_kwargs(config))
        self.sampling_params = SamplingParams(
            temperature=config.temperature,
            max_tokens=config.max_tokens,
        )

    def __enter__(self) -> "OfflineVLLMInferer":
        return self

    def close(self) -> None:
        if self.llm is not None:
            self.llm.llm_engine.engine_core.shutdown()
            self.llm = None

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def infer(self, query: str, video: TimestampedVideoInput) -> str:
        if self.dry_run:
            return "[DRY_RUN] skipped model call"
        assert self.processor is not None
        assert self.llm is not None
        assert self.sampling_params is not None

        frames, metadata = preserve_video_resolution(video.frames, video.metadata)
        contract = resolution_contract(frames)

        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": "presampled_event_guided_rgb"},
                    {"type": "text", "text": query},
                ],
            },
        ]
        messages = augment_messages(messages, ACTIVE_EVENT_META, ACTIVE_EVENT_IMAGES)
        checked = processor_inputs(self.processor, messages, frames, metadata, ACTIVE_EVENT_IMAGES)
        if checked["input_ids"].shape[-1] + self.config.max_tokens > self.config.max_model_len:
            raise ValueError("RGB+event input plus generation budget exceeds context; refusing truncation")
        prompt = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        model_input = {
            "prompt": prompt,
            "multi_modal_data": {
                "video": (frames, metadata),
            },
            # The frames have already been selected. Re-sampling here would
            # destroy the original frame-index/timestamp correspondence.
            "mm_processor_kwargs": contract["processor_kwargs"],
        }
        if ACTIVE_EVENT_IMAGES:
            model_input["multi_modal_data"]["image"] = ACTIVE_EVENT_IMAGES
        outputs = self.llm.generate([model_input], self.sampling_params, use_tqdm=False)
        return outputs[0].outputs[0].text


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--event-image-config", default=None)
    parser.add_argument("--disable-event-images", action="store_true", help="Exact original RGB input ablation")
    parser.add_argument(
        "--task",
        choices=("all", "d", "c", "t", "vqa"),
        default="all",
        help="Run one task, or all four tasks in d -> c -> t -> vqa order (default: all).",
    )
    parser.add_argument(
        "--input-jsonl",
        default=os.getenv("INPUT_JSONL_PATH", str(Path(__file__).resolve().parents[1]/"data/annotations/test.jsonl")),
    )
    parser.add_argument(
        "--event-root",
        default=os.getenv("EVENT_ROOT", str(Path(__file__).resolve().parents[1]/"data/events/test")),
    )
    parser.add_argument(
        "--output-jsonl",
        default=os.getenv("OUTPUT_JSONL_PATH", ""),
        help=(
            "Single-task output path. In --task all mode, include the literal "
            "'{task}' placeholder, or leave empty to use the default four paths."
        ),
    )
    parser.add_argument("--max-samples", type=int, default=int(os.getenv("MAX_SAMPLES", "0")))
    parser.add_argument("--min-frames", type=int, default=int(os.getenv("MIN_FRAMES", "16")))
    parser.add_argument("--max-frames", type=int, default=int(os.getenv("MAX_FRAMES", "24")))
    parser.add_argument(
        "--analysis-bins",
        type=int,
        default=int(os.getenv("EVENT_ANALYSIS_BINS", "96")),
    )
    parser.add_argument("--max-edge", type=int, default=int(os.getenv("MAX_EDGE", "640")),
                        help="Match the original multi-image inference edge limit; no later budget downscale.")
    parser.add_argument("--dry-run", action="store_true", default=os.getenv("DRY_RUN", "0") == "1")
    parser.add_argument(
        "--cuda-visible-devices",
        default="auto",
        help=(
            "Physical GPU ids made visible before importing torch/vLLM, or 'auto' "
            "to select GPU(s) with enough free memory (default: auto)."
        ),
    )
    parser.add_argument(
        "--model",
        default=os.getenv(
            "VLLM_MODEL",
            str(Path(__file__).resolve().parents[1]/"outputs/merged-model")
        ),
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.80,
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=16384,
    )
    parser.add_argument(
        "--enforce-eager",
        action="store_true",
        default=os.getenv("ENFORCE_EAGER", "1") == "1",
        help="Use eager execution by default to avoid compilation/CUDA-graph memory overhead.",
    )
    parser.add_argument(
        "--video-min-total-pixel-units",
        type=int,
        default=int(os.getenv("VIDEO_MIN_TOTAL_PIXEL_UNITS", "256")),
        help="Deprecated compatibility option; native pixels are preserved without budget resizing.",
    )
    parser.add_argument(
        "--video-max-total-pixel-units",
        type=int,
        default=int(os.getenv("VIDEO_MAX_TOTAL_PIXEL_UNITS", "2304")),
        help="Deprecated compatibility option; native pixels are preserved without budget resizing.",
    )
    parser.add_argument("--max-tokens", type=int, default=int(os.getenv("MAX_TOKENS", "2048")))
    parser.add_argument(
        "--temperature",
        type=float,
        default=float(os.getenv("TEMPERATURE", "0.1")),
    )
    return parser


def _load_processed(path: Path, task: str) -> set:
    processed = set()
    if not path.exists():
        return processed
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if item.get("sample_meta", {}).get("resolution_contract", {}).get("policy") != "preserve_pixels_pad_to_grid_v1":
                raise ValueError(f"{path} contains results from the old spatial policy; use a new output path")
            if item.get("sample_meta", {}).get("prompt_policy") != "timestamp_with_roi_blocks_v1":
                raise ValueError(f"{path} lacks per-block ROI prompts; use a new output path")
            if task == "vqa":
                processed.add(
                    f"{item.get('video_path')}_{item.get('question_id')}_{item.get('shift_id')}"
                )
            else:
                processed.add(item.get("video_path"))
    return processed


def _make_plan(
    event_path: Optional[str],
    duration: float,
    config: EventSamplerConfig,
) -> SamplingPlan:
    if not event_path:
        return build_sampling_plan(None, duration, config)
    try:
        analysis = analyze_event_h5(event_path, duration, config)
        return build_sampling_plan(analysis, duration, config)
    except Exception as exc:
        print(f"[Event warning] {event_path}: {exc}; falling back to uniform RGB sampling")
        return build_sampling_plan(None, duration, config)


def _make_output_entry(
    task: str,
    data: Dict,
    prediction: str,
    video_path: str,
    event_path: Optional[str],
    plan: SamplingPlan,
    samples: Sequence[RGBSample],
    timestamped_video: TimestampedVideoInput,
) -> Dict:
    aligned_frames, _ = preserve_video_resolution(timestamped_video.frames, timestamped_video.metadata)
    entry = {
        "video_path": video_path,
        "event_info": {
            "event_h5_path": event_path or "",
            "event_used_as_model_input": bool(ACTIVE_EVENT_IMAGES),
            "event_image_supplement": ACTIVE_EVENT_META,
            "event_used_as_sampling_controller": bool(event_path),
        },
        "sample_meta": {
            "sampling": plan.as_dict(),
            "input_format": "native_timestamp_video_plus_independent_event_images",
            "prompt_policy": "timestamp_with_roi_blocks_v1",
            "resolution_contract": resolution_contract(aligned_frames),
            "num_rgb_samples": len(samples),
            "num_raw_video_frames": int(timestamped_video.frames.shape[0]),
            "num_qwen_temporal_patches": len(samples),
            "qwen_temporal_patch_size": QWEN_TEMPORAL_PATCH_SIZE,
            "rgb_timestamps": [round(float(sample.timestamp), 4) for sample in samples],
            "rgb_frame_indices": [int(sample.frame_index) for sample in samples],
            "rgb_samples": [sample.metadata() for sample in samples],
            "video_metadata": {
                **timestamped_video.metadata,
                "fps": round(float(timestamped_video.metadata["fps"]), 8),
                "duration": round(float(timestamped_video.metadata["duration"]), 4),
            },
            "qwen_patch_timestamps": [
                round(float(value), 4) for value in timestamped_video.patch_timestamps
            ],
            "timestamp_formula": "mean(frames_indices_in_temporal_patch / original_fps)",
            "canvas_size": [
                timestamped_video.canvas_width,
                timestamped_video.canvas_height,
            ],
        },
        "prediction": prediction,
    }
    if task == "d":
        entry["gt_info"] = data.get("description", "")
    elif task == "c":
        entry["gt_info"] = {"category": data.get("category", [])}
    elif task == "t":
        entry["gt_info"] = {"temporal_segments": data.get("temporal_segments", [])}
    return entry


TASK_ORDER = ("d", "c", "t", "vqa")


def _resolve_tasks_and_outputs(args) -> tuple[tuple[str, ...], Dict[str, Path]]:
    tasks = TASK_ORDER if args.task == "all" else (args.task,)
    default_root = Path(__file__).resolve().parents[1] / "outputs/predictions"

    if args.output_jsonl:
        if len(tasks) > 1 and "{task}" not in args.output_jsonl:
            raise ValueError(
                "--task all needs an --output-jsonl containing '{task}', for example "
                "'/path/305-{task}.jsonl'; otherwise leave --output-jsonl empty"
            )
        outputs = {
            task: Path(args.output_jsonl.format(task=task))
            for task in tasks
        }
    else:
        outputs = {task: default_root / f"{task}.jsonl" for task in tasks}

    for output in outputs.values():
        output.parent.mkdir(parents=True, exist_ok=True)
    return tasks, outputs


def main() -> None:
    global EVENT_IMAGE_CONFIG, ACTIVE_EVENT_IMAGES, ACTIVE_EVENT_META
    args = _build_arg_parser().parse_args()
    sampler_config = EventSamplerConfig(analysis_bins=args.analysis_bins, min_frames=args.min_frames, max_frames=args.max_frames)
    policy_path=Path(args.model)/"timestamp_event_images_policy.json"
    if args.event_image_config is None:
        if policy_path.exists():
            from event_rgb_visualization import EventImageConfig
            EVENT_IMAGE_CONFIG=EventImageConfig(**json.loads(policy_path.read_text())["event_config"])
        else:
            EVENT_IMAGE_CONFIG=load_config(Path(__file__).resolve().parents[1]/"sft/event_rgb/configs/adaptive_gray_pair.json")
    else:
        EVENT_IMAGE_CONFIG = load_config(args.event_image_config)
        if policy_path.exists() and json.loads(policy_path.read_text())["config_signature"]!=EVENT_IMAGE_CONFIG.signature():
            print("[Event config] Explicit representation ablation differs from checkpoint training policy")
    EVENT_IMAGE_CONFIG = bind_sampler_config(EVENT_IMAGE_CONFIG, sampler_config)
    if policy_path.exists():
        from event_rgb_visualization import EventImageConfig
        saved_policy=json.loads(policy_path.read_text())
        if EventImageConfig(**saved_policy["event_config"]).signature()!=saved_policy["config_signature"]:
            raise ValueError("Checkpoint uses an older event selection policy; use a checkpoint trained with regenerated score-ranked event data")
        if args.event_image_config is None and saved_policy["config_signature"] != EVENT_IMAGE_CONFIG.signature():
            raise ValueError("RGB sampler scoring settings differ from checkpoint event policy; match --analysis-bins or explicitly configure an ablation")
    if (Path(args.model)/"adapter_config.json").exists():
        raise ValueError("Pass a merged model to --model; use sft/merge_timestamp_event_images_lora.py first")
    if args.disable_event_images:
        from dataclasses import replace
        EVENT_IMAGE_CONFIG = replace(EVENT_IMAGE_CONFIG, max_groups=0)
    if args.tensor_parallel_size <= 0:
        raise ValueError("tensor-parallel-size must be positive")
    if not 0.0 < args.gpu_memory_utilization < 1.0:
        raise ValueError("gpu-memory-utilization must be in (0, 1)")
    if args.dry_run:
        visible_devices: List[str] = []
        gpu_snapshot: List[Dict[str, float]] = []
    else:
        visible_devices, gpu_snapshot = _resolve_visible_devices(
            args.cuda_visible_devices,
            args.tensor_parallel_size,
            args.gpu_memory_utilization,
        )

    # This must happen before importing transformers/vLLM (done lazily in
    # OfflineVLLMInferer).  On this server vLLM 0.13's separate V1 EngineCore
    # process exits before reporting its root exception.  Keeping the V1
    # engine in the launcher process is verified to load Qwen3-VL and run
    # timestamped video inference successfully.
    if visible_devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(visible_devices)
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

    if args.max_edge <= 0:
        raise ValueError("max-edge must be positive")
    if not 0 < args.video_min_total_pixel_units <= args.video_max_total_pixel_units:
        raise ValueError(
            "expected 0 < video-min-total-pixel-units <= video-max-total-pixel-units"
        )
    tasks, outputs = _resolve_tasks_and_outputs(args)

    # Prevent resuming incompatible model/representation/decoding runs into one file.
    run = {"event_config": __import__("dataclasses").asdict(EVENT_IMAGE_CONFIG), "arguments": vars(args)}
    for output in outputs.values():
        marker = output.with_suffix(output.suffix + ".run.json")
        if marker.exists() and json.loads(marker.read_text()) != run:
            raise ValueError(f"Run settings changed; use a new output path: {output}")
        if output.exists() and output.stat().st_size and not marker.exists():
            raise ValueError(f"Refusing untracked existing results: {output}")
        marker.write_text(json.dumps(run, indent=2))
    event_index = EventH5Index(args.event_root)
    sampler_config = EventSamplerConfig(
        analysis_bins=args.analysis_bins,
        min_frames=args.min_frames,
        max_frames=args.max_frames,
    )
    inference_config = OfflineVLLMConfig(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        enforce_eager=args.enforce_eager,
        video_min_pixels=args.video_min_total_pixel_units * 32 * 32,
        video_max_pixels=args.video_max_total_pixel_units * 32 * 32,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        max_frames=args.max_frames,
        max_edge=args.max_edge,
    )
    processed_by_task = {
        task: _load_processed(outputs[task], task)
        for task in tasks
    }

    print(f"Event H5 files indexed: {event_index.indexed_count}")
    print(f"Tasks: {' -> '.join(tasks)}")
    for task in tasks:
        print(f"Output[{task}]: {outputs[task]}")
    print("Input format: one non-uniform native video with original fps/frames_indices")
    print("Spatial policy: preserve all input pixels, pad to 32, do_resize=False; total-pixel limits are not applied")
    print(f"Event policy: {POLICY}; activity_filter={EVENT_IMAGE_CONFIG.filter_inactive_anchors}, one_per_segment={EVENT_IMAGE_CONFIG.one_anchor_per_segment}, balanced_note={EVENT_IMAGE_CONFIG.balanced_event_note}")
    print(f"Engine workload: max_num_seqs=1, image<={EVENT_IMAGE_CONFIG.max_groups * 3}, video=1, "
          f"profile_raw_frames={args.max_frames * QWEN_TEMPORAL_PATCH_SIZE}, "
          f"profile_edge={(args.max_edge + 31) // 32 * 32}, eager={args.enforce_eager}")
    if gpu_snapshot:
        print(
            "GPU memory before model load: "
            + ", ".join(
                f"GPU{int(item['index'])}={item['free_mib']:.0f}/{item['total_mib']:.0f} MiB free"
                for item in gpu_snapshot
            )
        )
    print(
        "Runtime: "
        f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'not-needed')}, "
        f"tensor_parallel_size={args.tensor_parallel_size}, "
        "VLLM_ENABLE_V1_MULTIPROCESSING=0"
    )
    if args.dry_run:
        print("DRY_RUN=1: model loading and generation will be skipped")

    processed_count = 0
    with OfflineVLLMInferer(
        inference_config, dry_run=args.dry_run
    ) as inferer, open(
        args.input_jsonl, "r", encoding="utf-8"
    ) as input_handle, ExitStack() as output_stack:
        output_handles = {
            task: output_stack.enter_context(outputs[task].open("a", encoding="utf-8"))
            for task in tasks
        }
        for row_index, line in enumerate(
            tqdm(input_handle, desc=f"event-guided-timestamp {args.task}")
        ):
            if args.max_samples and processed_count >= args.max_samples:
                break
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue

            video_path = str(data.get("video_path", ""))
            if not video_path:
                raise ValueError(f"Missing video_path in annotation row {row_index}")
            if not Path(video_path).is_absolute():
                video_path = str(Path(__file__).resolve().parents[1] / video_path)
            if not os.path.isfile(video_path):
                raise FileNotFoundError(video_path)

            pending_non_vqa = any(
                task != "vqa" and video_path not in processed_by_task[task]
                for task in tasks
            )
            pending_vqa = False
            if "vqa" in tasks:
                for question_index, item in enumerate(
                    data.get("vqa_pairs", []) or data.get("vqa", [])
                ):
                    question_id = f"v{row_index}_q{question_index}"
                    variants = CircularManager.get_variants(
                        item.get("options", {}), item.get("correct_key")
                    )
                    if any(
                        f"{video_path}_{question_id}_{variant['shift_id']}"
                        not in processed_by_task["vqa"]
                        for variant in variants
                    ):
                        pending_vqa = True
                        break
            if not pending_non_vqa and not pending_vqa:
                continue

            try:
                total_frames, fps, duration = get_video_metadata(video_path)
                event_path = event_index.resolve(video_path, data)
                plan = _make_plan(event_path, duration, sampler_config)
                samples = extract_rgb_samples(video_path, plan, max_edge=args.max_edge)
                timestamped_video = build_timestamped_video(
                    samples,
                    total_frames=total_frames,
                    fps=fps,
                    duration=duration,
                )
            except Exception as exc:
                print(f"[RGB warning] {video_path}: {exc}")
                continue

            ACTIVE_EVENT_IMAGES, ACTIVE_EVENT_META = build_bundle(
                event_path, [sample.metadata() for sample in samples], fps, duration, EVENT_IMAGE_CONFIG
            )
            if ACTIVE_EVENT_META['status'] not in ('ok', 'disabled', 'no_eligible_event_anchor'):
                print(f"[Event supplement] {video_path}: {ACTIVE_EVENT_META['status']} {ACTIVE_EVENT_META.get('error', '')}")
            base_prompt = build_sampling_prompt(duration, plan, samples)
            wrote_video = False
            for task in tasks:
                output_handle = output_handles[task]
                processed = processed_by_task[task]

                if task != "vqa":
                    if video_path in processed:
                        continue
                    task_prompt = PromptFactory.task_prompt(task, data)
                    prediction = inferer.infer(
                        base_prompt + task_prompt,
                        timestamped_video,
                    )
                    entry = _make_output_entry(
                        task,
                        data,
                        prediction,
                        video_path,
                        event_path,
                        plan,
                        samples,
                        timestamped_video,
                    )
                    output_handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
                    output_handle.flush()
                    processed.add(video_path)
                    wrote_video = True
                    continue

                pairs = data.get("vqa_pairs", []) or data.get("vqa", [])
                for question_index, item in enumerate(pairs):
                    variants = CircularManager.get_variants(
                        item.get("options", {}), item.get("correct_key")
                    )
                    question_id = f"v{row_index}_q{question_index}"
                    for variant in variants:
                        run_key = f"{video_path}_{question_id}_{variant['shift_id']}"
                        if run_key in processed:
                            continue
                        shifted_item = {
                            "question": item.get("question", ""),
                            "options": variant["options"],
                        }
                        task_prompt = PromptFactory.task_prompt(
                            "vqa",
                            data,
                            vqa_item=shifted_item,
                        )
                        prediction = inferer.infer(
                            base_prompt + task_prompt,
                            timestamped_video,
                        )
                        entry = _make_output_entry(
                            task,
                            data,
                            prediction,
                            video_path,
                            event_path,
                            plan,
                            samples,
                            timestamped_video,
                        )
                        entry.update(
                            {
                                "question_id": question_id,
                                "shift_id": variant["shift_id"],
                                "question": item.get("question"),
                                "options_at_shift": variant["options"],
                                "gt_correct_key": variant["correct_key"],
                            }
                        )
                        output_handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
                        output_handle.flush()
                        processed.add(run_key)
                        wrote_video = True

            if wrote_video:
                processed_count += 1

    print("Finished. Results saved to:")
    for task in tasks:
        print(f"  {task}: {outputs[task]}")


if __name__ == "__main__":
    main()
