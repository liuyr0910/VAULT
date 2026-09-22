#!/usr/bin/env bash
set -euo pipefail

# LoRA SFT for event-guided non-uniform native videos with true Qwen3-VL
# timestamps. Build DATA_PATH and the .npy caches first with:
#   python sft/prepare_timestamp_event_images_sft.py

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON=${PYTHON:-python}
TORCHRUN=${TORCHRUN:-torchrun}
ENTRY=${ROOT}/qwen-vl-finetune/qwenvl/train-craft/train_qwen_timestamp_event_images.py
BASE_MODEL=${BASE_MODEL:?Set BASE_MODEL to the local Qwen3-VL-8B-Instruct directory}
DATA_PATH=${DATA_PATH:-${ROOT}/sft/data/all_tasks_mixed_timestamp_event_images.json}
OUTPUT_DIR=${OUTPUT_DIR:-${ROOT}/outputs/lora}
DEEPSPEED=${DEEPSPEED:-${ROOT}/qwen-vl-finetune/scripts/zero2_timestamp.json}

# Set CUDA_VISIBLE_DEVICES externally to select available GPUs.
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
export PATH=${CUDA_HOME}/bin:${PATH}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
export NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS:-1}
export PYTORCH_ALLOC_CONF=${PYTORCH_ALLOC_CONF:-expandable_segments:True}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1

NPROC_PER_NODE=${NPROC_PER_NODE:-2}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-$(shuf -i 20001-29999 -n 1)}
DATALOADER_NUM_WORKERS=${DATALOADER_NUM_WORKERS:-2}
BATCH_SIZE=${BATCH_SIZE:-1}
EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE:-1}
# Preserve the previous effective batch of 32 when changing GPU count.
EFFECTIVE_BATCH_SIZE=${EFFECTIVE_BATCH_SIZE:-32}
if [[ -z "${GRAD_ACCUM:-}" ]]; then
    if (( EFFECTIVE_BATCH_SIZE % (NPROC_PER_NODE * BATCH_SIZE) != 0 )); then
        printf 'EFFECTIVE_BATCH_SIZE must be divisible by NPROC_PER_NODE * BATCH_SIZE.\n' >&2
        exit 1
    fi
    GRAD_ACCUM=$((EFFECTIVE_BATCH_SIZE / (NPROC_PER_NODE * BATCH_SIZE)))
fi
NUM_TRAIN_EPOCHS=${NUM_TRAIN_EPOCHS:-3}
MAX_STEPS=${MAX_STEPS:-}

# Timestamp videos preserve cached pixels and only pad to the 32-pixel grid.
# Existing .npy caches remain valid. Use a new output directory to train afresh.
IMAGE_MIN_PIXEL_UNITS=${IMAGE_MIN_PIXEL_UNITS:-64}
IMAGE_MAX_PIXEL_UNITS=${IMAGE_MAX_PIXEL_UNITS:-256}

if [[ ! -f "${DATA_PATH}" ]]; then
    printf 'Timestamp event-guided SFT data not found: %s\n' "${DATA_PATH}" >&2
    printf 'Run sft/prepare_timestamp_event_images_sft.py first.\n' >&2
    exit 1
fi

if [[ -e "${OUTPUT_DIR}" ]] && [[ ! -d "${OUTPUT_DIR}" ]]; then
    printf 'Output path exists but is not a directory: %s\n' "${OUTPUT_DIR}" >&2
    exit 1
fi

if [[ "${DRY_RUN:-0}" != "1" ]]; then
    "${PYTHON}" \
        "${ROOT}/sft/check_timestamp_event_images.py" \
        --data-path "${DATA_PATH}" --model "${BASE_MODEL}" \
        --model-max-length "${MODEL_MAX_LENGTH:-6144}" --output-dir "${OUTPUT_DIR}"
fi

cd "${ROOT}/qwen-vl-finetune"

printf 'Timestamp SFT: micro_batch=%s, GPUs=%s, grad_accum=%s, effective_batch=%s, context=%s\n' \
    "${BATCH_SIZE}" "${NPROC_PER_NODE}" "${GRAD_ACCUM}" \
    "$((BATCH_SIZE * NPROC_PER_NODE * GRAD_ACCUM))" "${MODEL_MAX_LENGTH:-6144}"
printf 'Inputs: native RGB pixels + per-block ROI text; plus independent event images; no RGB resampling, resizing or packing.\n'

COMMAND=(
    "${TORCHRUN}"
    --nproc_per_node="${NPROC_PER_NODE}"
    --master_addr="${MASTER_ADDR}"
    --master_port="${MASTER_PORT}"
    "${ENTRY}"
    --deepspeed "${DEEPSPEED}"
    --model_name_or_path "${BASE_MODEL}"
    --data_path "${DATA_PATH}"
    --data_flatten False
    --data_packing False
    --min_pixels "$((IMAGE_MIN_PIXEL_UNITS * 32 * 32))"
    --max_pixels "$((IMAGE_MAX_PIXEL_UNITS * 32 * 32))"
    --video_min_frames 32
    --video_max_frames 48
    # This config value is not used to re-sample cached videos. Their original
    # fps and frames_indices are supplied separately for every record.
    --video_fps 2
    --bf16 True
    --tune_mm_vision False
    --tune_mm_mlp False
    --tune_mm_llm False
    --output_dir "${OUTPUT_DIR}"
    --num_train_epochs "${NUM_TRAIN_EPOCHS}"
    --per_device_train_batch_size "${BATCH_SIZE}"
    --per_device_eval_batch_size "${EVAL_BATCH_SIZE}"
    --gradient_accumulation_steps "${GRAD_ACCUM}"
    --eval_strategy no
    --save_strategy "${SAVE_STRATEGY:-steps}"
    --save_steps "${SAVE_STEPS:-30}"
    --save_total_limit 30
    --learning_rate 2e-5
    --weight_decay 0.1
    --warmup_ratio 0.1
    --lr_scheduler_type cosine
    --logging_steps 1
    --model_max_length "${MODEL_MAX_LENGTH:-6144}"
    --gradient_checkpointing True
    --torch_compile False
    --auto_find_batch_size False
    --dataloader_num_workers "${DATALOADER_NUM_WORKERS}"
    --dataloader_pin_memory True
    --remove_unused_columns False
    --seed 42
    --run_name qwen3vl-8b-timestamp-event-images
    --report_to "${REPORT_TO:-tensorboard}"
    --use_lora True
    --lora_r 32
    --lora_alpha 64
    --lora_dropout 0.05
    --lora_target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj
    --lora_bias none
)

if [[ -n "${MAX_STEPS}" ]]; then
    COMMAND+=(--max_steps "${MAX_STEPS}")
fi

if [[ "${DRY_RUN:-0}" == "1" ]]; then
    printf '%q ' "${COMMAND[@]}"
    printf '\n'
    exit 0
fi

exec "${COMMAND[@]}"
