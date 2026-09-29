#!/usr/bin/env python3
"""Merge EventVAULT LoRA weights into Qwen3-VL on CPU."""

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--adapter", type=Path, default=ROOT / "outputs/lora/checkpoint-420"
    )
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/merged-model")
    parser.add_argument(
        "--base-model", type=Path, default=ROOT / "models/Qwen3-VL-8B-Instruct"
    )
    args = parser.parse_args()
    import torch
    from peft import PeftModel
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.base_model, dtype=torch.bfloat16, device_map="cpu"
    )
    model = PeftModel.from_pretrained(model, args.adapter).merge_and_unload()
    model.config.use_cache = True
    model.save_pretrained(args.output, safe_serialization=True)
    AutoProcessor.from_pretrained(args.base_model).save_pretrained(args.output)
    print(args.output)


if __name__ == "__main__":
    main()
