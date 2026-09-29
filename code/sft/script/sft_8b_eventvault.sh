#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

# Distributed training configuration.
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}
export PYTORCH_ALLOC_CONF=${PYTORCH_ALLOC_CONF:-expandable_segments:True}
export TOKENIZERS_PARALLELISM=false
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-$(shuf -i 20001-29999 -n 1)}
NPROC_PER_NODE=${NPROC_PER_NODE:-2}

# Model, data, and output paths.
deepspeed="$ROOT/qwen-vl-finetune/scripts/zero2_eventvault.json"
llm="$ROOT/models/Qwen3-VL-8B-Instruct"
data_path="$ROOT/sft/data/all_tasks_mixed_eventvault.json"
run_name="eventvault"
output_dir="$ROOT/outputs/lora"
entry_file="$ROOT/qwen-vl-finetune/qwenvl/train/train_eventvault.py"

# Training parameters. The default effective batch size is 1 * 16 * 2 = 32.
args=(
    --deepspeed "$deepspeed"
    --model_name_or_path "$llm"
    --data_path "$data_path"
    --min_pixels $((64 * 32 * 32))
    --max_pixels $((256 * 32 * 32))
    --video_min_frames 32
    --video_max_frames 48
    --bf16 True
    --tune_mm_vision False
    --tune_mm_mlp False
    --tune_mm_llm False
    --output_dir "$output_dir"
    --num_train_epochs 2
    --per_device_train_batch_size 1
    --per_device_eval_batch_size 1
    --gradient_accumulation_steps 16
    --eval_strategy no
    --save_strategy steps
    --save_steps 30
    --save_total_limit 30
    --learning_rate 2e-5
    --optim adamw_torch
    --weight_decay 0.1
    --warmup_ratio 0.1
    --lr_scheduler_type cosine
    --logging_steps 1
    --model_max_length 6144
    --gradient_checkpointing True
    --dataloader_num_workers 2
    --seed 42
    --run_name "$run_name"
    --report_to tensorboard
    --use_lora True
    --lora_r 32
    --lora_alpha 64
    --lora_dropout 0.05
    --lora_target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj
    --lora_bias none
    --lora_modules_to_save mermaid_mlp
)

# Print the command without starting training when explicitly requested.
if [[ "${DRY_RUN:-0}" == "1" ]]; then
    printf '%q ' torchrun --nproc_per_node="$NPROC_PER_NODE" \
        --master_addr="$MASTER_ADDR" --master_port="$MASTER_PORT" "$entry_file" "${args[@]}"
    printf '\n'
    exit 0
fi

# Start LoRA training.
exec torchrun --nproc_per_node="$NPROC_PER_NODE" \
    --master_addr="$MASTER_ADDR" --master_port="$MASTER_PORT" \
    "$entry_file" "${args[@]}"
