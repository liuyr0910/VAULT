"""EventVAULT four-task inference with timestamped RGB and event observations.
Redistribute only non-anchor full RGB frames at a fixed per-video frame budget.
Event inputs are rendered on the ORIGINAL samples, before redistribution.
"""

from __future__ import annotations
import argparse
import json
import os
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
from eventvault import POLICY as RGB_POLICY, apply_inference

ACTIVE_COVERAGE = None
from event_representation import (
    load_config,
    build_bundle,
    augment_messages,
    bind_sampler_config,
    POLICY,
)

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


@dataclass(frozen=True)
class OfflineVLLMConfig:
    model: str
    tensor_parallel_size: int
    gpu_memory_utilization: float
    max_model_len: int
    enforce_eager: bool
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
            "image": {
                "count": EVENT_IMAGE_CONFIG.max_groups * 3,
                "width": (EVENT_IMAGE_CONFIG.sensor_width * 2 + 16 + 31) // 32 * 32,
                "height": (EVENT_IMAGE_CONFIG.sensor_height + 31) // 32 * 32,
            },
            "video": {
                "count": 1,
                "num_frames": raw_frames,
                "width": edge,
                "height": edge,
            },
        },
        "mm_processor_kwargs": {
            "do_resize": False,
            "do_sample_frames": False,
            "size": {
                "shortest_edge": 32 * 32,
                "longest_edge": raw_frames * edge * edge,
            },
        },
        "enforce_eager": config.enforce_eager,
        "seed": 0,
    }


from task_prompts import TASK_PROMPT_POLICY, PromptFactory as SharedPromptFactory


class PromptFactory(SharedPromptFactory):
    pass


class CircularManager:
    @staticmethod
    def get_variants(options: Dict, correct_key: str) -> List[Dict]:
        keys = sorted(options)
        values = [options[key] for key in keys]
        variants = []
        for shift_id in range(len(keys)):
            shifted = values[shift_id:] + values[:shift_id]
            variants.append(
                {
                    "options": {
                        keys[index]: shifted[index] for index in range(len(keys))
                    },
                    "correct_key": keys[
                        (keys.index(correct_key) - shift_id) % len(keys)
                    ],
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
            temperature=config.temperature, max_tokens=config.max_tokens
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
        prompt = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        model_input = {
            "prompt": prompt,
            "multi_modal_data": {"video": (frames, metadata)},
            "mm_processor_kwargs": contract["processor_kwargs"],
        }
        if ACTIVE_EVENT_IMAGES:
            model_input["multi_modal_data"]["image"] = ACTIVE_EVENT_IMAGES
        outputs = self.llm.generate([model_input], self.sampling_params, use_tqdm=False)
        return outputs[0].outputs[0].text


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=ROOT,
        help="Root for relative video_path values in annotations",
    )
    parser.add_argument(
        "--task",
        choices=("all", "d", "c", "t", "vqa"),
        default="all",
        help="Run one task, or all four tasks in d -> c -> t -> vqa order (default: all).",
    )
    parser.add_argument(
        "--input-jsonl",
        default=os.getenv(
            "INPUT_JSONL_PATH", str(ROOT / "data/annotations/test.jsonl")
        ),
    )
    parser.add_argument(
        "--event-root", default=os.getenv("EVENT_ROOT", str(ROOT / "data/events/test"))
    )
    parser.add_argument(
        "--output-jsonl",
        default=os.getenv("OUTPUT_JSONL_PATH", ""),
        help="Single-task output path. In --task all mode, include the literal '{task}' placeholder, or leave empty to use the default four paths.",
    )
    parser.add_argument(
        "--max-samples", type=int, default=int(os.getenv("MAX_SAMPLES", "0"))
    )
    parser.add_argument(
        "--min-frames", type=int, default=int(os.getenv("MIN_FRAMES", "16"))
    )
    parser.add_argument(
        "--max-frames", type=int, default=int(os.getenv("MAX_FRAMES", "24"))
    )
    parser.add_argument(
        "--analysis-bins", type=int, default=int(os.getenv("EVENT_ANALYSIS_BINS", "96"))
    )
    parser.add_argument(
        "--max-edge",
        type=int,
        default=int(os.getenv("MAX_EDGE", "640")),
        help="Match the original multi-image inference edge limit; no later budget downscale.",
    )
    parser.add_argument(
        "--dry-run", action="store_true", default=os.getenv("DRY_RUN", "0") == "1"
    )
    parser.add_argument(
        "--cuda-visible-devices",
        default=os.getenv("CUDA_VISIBLE_DEVICES", "0"),
        help="GPU ids to use for inference.",
    )
    parser.add_argument(
        "--model", default=os.getenv("VLLM_MODEL", str(ROOT / "outputs/merged-model"))
    )
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument("--max-model-len", type=int, default=16384)
    parser.add_argument(
        "--enforce-eager",
        action="store_true",
        default=os.getenv("ENFORCE_EAGER", "1") == "1",
        help="Use eager execution by default to avoid compilation/CUDA-graph memory overhead.",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=int(os.getenv("MAX_TOKENS", "2048"))
    )
    parser.add_argument(
        "--temperature", type=float, default=float(os.getenv("TEMPERATURE", "0.1"))
    )
    return parser


def _load_processed(path: Path, task: str) -> set:
    processed = set()
    if not path.exists():
        return processed
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            item = json.loads(line)
            if task == "vqa":
                processed.add(
                    f"{item.get('video_path')}_{item.get('question_id')}_{item.get('shift_id')}"
                )
            else:
                processed.add(item.get("video_path"))
    return processed


def _make_plan(
    event_path: Optional[str], duration: float, config: EventSamplerConfig
) -> SamplingPlan:
    analysis = analyze_event_h5(event_path, duration, config)
    return build_sampling_plan(analysis, duration, config)


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
    aligned_frames, _ = preserve_video_resolution(
        timestamped_video.frames, timestamped_video.metadata
    )
    entry = {
        "video_path": video_path,
        "event_info": {
            "event_h5_path": event_path or "",
            "event_used_as_model_input": bool(ACTIVE_EVENT_IMAGES),
            "event_image_supplement": ACTIVE_EVENT_META,
            "event_used_as_sampling_controller": bool(event_path),
        },
        "sample_meta": {
            "rgb_policy": RGB_POLICY,
            "coverage": ACTIVE_COVERAGE,
            "sampling": {
                **plan.as_dict(),
                "event_frames_sent_to_model": len(ACTIVE_EVENT_IMAGES),
            },
            "input_format": "native_timestamp_video_plus_independent_event_images",
            "prompt_policy": "timestamp_with_roi_blocks_v1",
            "task_prompt_policy": TASK_PROMPT_POLICY,
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
    default_root = ROOT / "outputs/predictions"
    if args.output_jsonl:
        outputs = {task: Path(args.output_jsonl.format(task=task)) for task in tasks}
    else:
        outputs = {task: default_root / f"{task}.jsonl" for task in tasks}
    for output in outputs.values():
        output.parent.mkdir(parents=True, exist_ok=True)
    return (tasks, outputs)


def main() -> None:
    global EVENT_IMAGE_CONFIG, ACTIVE_EVENT_IMAGES, ACTIVE_EVENT_META, ACTIVE_COVERAGE
    args = _build_arg_parser().parse_args()
    for field in ("input_jsonl", "event_root", "model"):
        path = Path(getattr(args, field)).expanduser()
        setattr(
            args,
            field,
            str((ROOT / path).resolve() if not path.is_absolute() else path.resolve()),
        )
    sampler_config = EventSamplerConfig(
        analysis_bins=args.analysis_bins,
        min_frames=args.min_frames,
        max_frames=args.max_frames,
    )
    EVENT_IMAGE_CONFIG = bind_sampler_config(
        load_config(ROOT / "configs/event_config.json"), sampler_config
    )
    if not args.dry_run:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    tasks, outputs = _resolve_tasks_and_outputs(args)
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
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        max_frames=args.max_frames,
        max_edge=args.max_edge,
    )
    processed_by_task = {task: _load_processed(outputs[task], task) for task in tasks}
    print(f"Event H5 files indexed: {event_index.indexed_count}")
    print(f"Tasks: {' -> '.join(tasks)}")
    for task in tasks:
        print(f"Output[{task}]: {outputs[task]}")
    print("Input format: one non-uniform native video with original fps/frames_indices")
    print(
        "Spatial policy: preserve all input pixels, pad to 32, do_resize=False; total-pixel limits are not applied"
    )
    print(f"Event policy: {POLICY}; score-ranked distinct-bin anchors")
    print(
        f"Engine workload: max_num_seqs=1, image<={EVENT_IMAGE_CONFIG.max_groups * 3}, video=1, profile_raw_frames={args.max_frames * QWEN_TEMPORAL_PATCH_SIZE}, profile_edge={(args.max_edge + 31) // 32 * 32}, eager={args.enforce_eager}"
    )
    print(
        f"Runtime: CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'not-needed')}, tensor_parallel_size={args.tensor_parallel_size}, VLLM_ENABLE_V1_MULTIPROCESSING=0"
    )
    if args.dry_run:
        print("DRY_RUN=1: model loading and generation will be skipped")
    processed_count = 0
    with (
        OfflineVLLMInferer(inference_config, dry_run=args.dry_run) as inferer,
        open(args.input_jsonl, "r", encoding="utf-8") as input_handle,
        ExitStack() as output_stack,
    ):
        output_handles = {
            task: output_stack.enter_context(outputs[task].open("a", encoding="utf-8"))
            for task in tasks
        }
        for row_index, line in enumerate(
            tqdm(input_handle, desc=f"EventVAULT {args.task}")
        ):
            if args.max_samples and processed_count >= args.max_samples:
                break
            if not line.strip():
                continue
            data = json.loads(line)
            if not line.strip():
                continue
            raw_path = data.get("video_path", "")
            video = Path(raw_path).expanduser()
            video_path = str(
                (args.data_root / video).resolve()
                if not video.is_absolute()
                else video.resolve()
            )
            pending_non_vqa = any(
                (
                    task != "vqa" and video_path not in processed_by_task[task]
                    for task in tasks
                )
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
                        (
                            f"{video_path}_{question_id}_{variant['shift_id']}"
                            not in processed_by_task["vqa"]
                            for variant in variants
                        )
                    ):
                        pending_vqa = True
                        break
            if not pending_non_vqa and (not pending_vqa):
                continue
            total_frames, fps, duration = get_video_metadata(video_path)
            event_path = event_index.resolve(video_path, data)
            plan = _make_plan(event_path, duration, sampler_config)
            samples = extract_rgb_samples(video_path, plan, max_edge=args.max_edge)
            timestamped_video = build_timestamped_video(
                samples, total_frames=total_frames, fps=fps, duration=duration
            )
            ACTIVE_EVENT_IMAGES, ACTIVE_EVENT_META = build_bundle(
                event_path,
                [sample.metadata() for sample in samples],
                fps,
                duration,
                EVENT_IMAGE_CONFIG,
            )
            samples, timestamped_video, plan, ACTIVE_EVENT_META, ACTIVE_COVERAGE = (
                apply_inference(
                    video_path,
                    samples,
                    timestamped_video,
                    plan,
                    ACTIVE_EVENT_META,
                    args.max_edge,
                )
            )
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
                        base_prompt + task_prompt, timestamped_video
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
                            "vqa", data, vqa_item=shifted_item
                        )
                        prediction = inferer.infer(
                            base_prompt + task_prompt, timestamped_video
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
                        output_handle.write(
                            json.dumps(entry, ensure_ascii=False) + "\n"
                        )
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
