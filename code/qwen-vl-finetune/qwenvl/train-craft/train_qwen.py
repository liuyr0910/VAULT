"""Original-style Qwen-VL LoRA SFT entry point for event-guided images."""

import logging
import os
import pathlib
import sys
from pathlib import Path

import torch
import transformers


TRAIN_CRAFT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TRAIN_CRAFT_DIR.parents[1]
if str(TRAIN_CRAFT_DIR) not in sys.path:
    sys.path.insert(0, str(TRAIN_CRAFT_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from argument import (  # noqa: E402
    DataArguments,
    LoraArguments,
    ModelArguments,
    TrainingArguments,
)
from trainer import replace_qwen2_vl_attention_class  # noqa: E402
from qwenvl.data.data_processor_event_lora import (  # noqa: E402
    make_supervised_data_module,
)

from peft import LoraConfig, TaskType, get_peft_model  # noqa: E402
from transformers import (  # noqa: E402
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
    Qwen2VLForConditionalGeneration,
    Qwen3VLMoeForConditionalGeneration,
    Qwen3VLForConditionalGeneration,
    Trainer,
)


logger = logging.getLogger(__name__)
local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def safe_save_model_for_hf_trainer(
    trainer: transformers.Trainer, output_dir: str
) -> None:
    """Collect and save the trainer state in the original SFT format."""
    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)


def set_model(model_args, model):
    if model_args.tune_mm_vision:
        for _, parameter in model.visual.named_parameters():
            parameter.requires_grad = True
    else:
        for _, parameter in model.visual.named_parameters():
            parameter.requires_grad = False

    if model_args.tune_mm_mlp:
        if hasattr(model.visual, "merger"):
            for _, parameter in model.visual.merger.named_parameters():
                parameter.requires_grad = True
    else:
        if hasattr(model.visual, "merger"):
            for _, parameter in model.visual.merger.named_parameters():
                parameter.requires_grad = False

    if model_args.tune_mm_llm:
        for _, parameter in model.language_model.named_parameters():
            parameter.requires_grad = True
        model.lm_head.requires_grad = True
    else:
        for _, parameter in model.language_model.named_parameters():
            parameter.requires_grad = False
        model.lm_head.requires_grad = False


def train(attn_implementation="flash_attention_2", patch_embed_linear=False):
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments, LoraArguments)
    )
    model_args, data_args, training_args, lora_args = parser.parse_args_into_dataclasses()
    local_rank = training_args.local_rank
    os.makedirs(training_args.output_dir, exist_ok=True)

    model_name = model_args.model_name_or_path.lower()
    if "qwen3" in model_name and "moe" in model_name:
        model_class = Qwen3VLMoeForConditionalGeneration
        data_args.model_type = "qwen3vl"
    elif "qwen3" in model_name:
        model_class = Qwen3VLForConditionalGeneration
        data_args.model_type = "qwen3vl"
    elif "qwen2.5" in model_name:
        model_class = Qwen2_5_VLForConditionalGeneration
        data_args.model_type = "qwen2.5vl"
    else:
        model_class = Qwen2VLForConditionalGeneration
        data_args.model_type = "qwen2vl"

    model = model_class.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        attn_implementation=attn_implementation,
        dtype=(torch.bfloat16 if training_args.bf16 else None),
    )
    if patch_embed_linear:
        from qwenvl.patch_embed_linear import enable_patch_embed_linear
        enable_patch_embed_linear(model)
    rank0_print(
        f"The initialized model is {model_args.model_name_or_path}, "
        f"class is {model.__class__.__name__}"
    )

    processor = AutoProcessor.from_pretrained(model_args.model_name_or_path)
    # The dataset/collator use this tokenizer, not the separate Trainer tokenizer.
    processor.tokenizer.model_max_length = training_args.model_max_length
    if data_args.data_flatten or data_args.data_packing:
        replace_qwen2_vl_attention_class()
    model.config.use_cache = False

    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, inputs, output):
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )

    # This is the original LM-only LoRA path: freeze the base model, inject
    # adapters, and let the regular Hugging Face Trainer optimize them.
    if lora_args.use_lora:
        rank0_print("LoRA enabled")
        for parameter in model.parameters():
            parameter.requires_grad = False

        lora_config = LoraConfig(
            r=lora_args.lora_r,
            lora_alpha=lora_args.lora_alpha,
            lora_dropout=lora_args.lora_dropout,
            target_modules=lora_args.lora_target_modules,
            bias=lora_args.lora_bias,
            task_type=TaskType.CAUSAL_LM,
            modules_to_save=lora_args.lora_modules_to_save,
        )
        model = get_peft_model(model, lora_config)
        if local_rank in (None, 0):
            model.print_trainable_parameters()
    else:
        set_model(model_args, model)
        if local_rank in (None, 0):
            trainable = sum(
                parameter.numel()
                for parameter in model.parameters()
                if parameter.requires_grad
            )
            print(f"Trainable parameters: {trainable}")

    data_module = make_supervised_data_module(processor, data_args=data_args)
    trainer = Trainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        **data_module,
    )

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        logging.info("checkpoint found, resume training")
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    trainer.save_state()
    model.config.use_cache = True
    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)
    processor.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    train(attn_implementation="flash_attention_2")
