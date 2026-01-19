#!/usr/bin/env bash
# FSDP Full Fine-tuning for OpenVLA-7B on NPU
# Note: Uses VLA-Adapter model loading (not HF AutoClasses)

# Resolve repo root based on script location
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

mkdir -p logs
mkdir -p hf_cache

# Redirect HF cache to project disk
export HF_HOME="${ROOT_DIR}/hf_cache"

# Use EGL for consistency with eval
export MUJOCO_GL=egl

# Set model type for ACTION_TOKEN_BEGIN_IDX detection
export VLA_MODEL_TYPE=OPENVLA
export PYOPENGL_PLATFORM=egl

# Auto-activate local virtualenv if present
if [ -f "${ROOT_DIR}/env/bin/activate" ]; then
  source "${ROOT_DIR}/env/bin/activate"
fi

# Prefer the project-local torchrun if available
TORCHRUN_BIN="${ROOT_DIR}/env/bin/torchrun"
if [ ! -x "${TORCHRUN_BIN}" ]; then
  TORCHRUN_BIN="torchrun"
fi

# Configuration
data_name=libero_object_no_noops
current_time=$(date "+%Y-%m-%d_%H-%M-%S")

echo "[INFO] ============================================"
echo "[INFO] Launching FSDP Full Fine-tuning"
echo "[INFO] ============================================"
echo "[INFO] Using VLA-Adapter model loading (not HF AutoClasses)"
echo "[INFO] FSDP will shard model across 4 NPUs"
echo "[INFO] Expected memory per NPU: ~15-20 GB"
echo "[INFO] ============================================"
echo ""

# Full Fine-tuning with FSDP (using HF AutoClasses with added FSDP support)
"${TORCHRUN_BIN}" --standalone --nnodes 1 --nproc-per-node 4 scripts/finetune_openvla.py \
  --vla_path /models/zhangguoxi/openvla-7b \
  --data_root_dir /datasets/zhangguoxi/modified_libero_rlds \
  --dataset_name "${data_name}" \
  --use_hf_model True \
  --training_mode full \
  --use_fsdp True \
  --train_strategy fsdp-shard-grad-op \
  --freeze_vision_backbone False \
  --freeze_llm_backbone False \
  --unfreeze_last_llm_layer True \
  --enable_gradient_checkpointing False \
  --enable_mixed_precision_training True \
  --reduce_in_full_precision False \
  --run_root_dir runs \
  --batch_size 1 \
  --per_device_batch_size 1 \
  --global_batch_size 4 \
  --grad_accumulation_steps 4 \
  --max_steps 5005 \
  --save_steps 2500 \
  --learning_rate 2e-5 \
  --lr_scheduler_type constant \
  --warmup_ratio 0.03 \
  --weight_decay 0.01 \
  --max_grad_norm 1.0 \
  --seed 7 \
  --wandb_entity "yiyangchen-sylvia-bigai" \
  --wandb_project "openvla-fsdp" \
  --run_id_note "fsdp-${data_name}--${current_time}" \
  > "logs/openvla-fsdp-${data_name}--${current_time}.log" 2>&1

exit_code=$?

if [ $exit_code -eq 0 ]; then
  echo "[SUCCESS] FSDP training completed successfully!"
else
  echo "[ERROR] FSDP training failed with exit code: $exit_code"
  echo "[INFO] Check log file: logs/openvla-fsdp-${data_name}--${current_time}.log"
  echo ""
  echo "[SUGGESTION] If FSDP failed, consider using LoRA instead:"
  echo "  bash run_finetune_openvla_lora_recommended.sh"
fi

echo ""
echo "[INFO] Log file: logs/openvla-fsdp-${data_name}--${current_time}.log"

