#!/usr/bin/env bash
# set -euo pipefail

# Resolve repo root based on script location
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

mkdir -p logs
mkdir -p hf_cache

# Redirect HF cache to project disk (avoid small $HOME quota)
export HF_HOME="${ROOT_DIR}/hf_cache"

# Use EGL for consistency with eval (harmless for training)
export MUJOCO_GL=egl

# Set model type for ACTION_TOKEN_BEGIN_IDX detection
export VLA_MODEL_TYPE=OPENVLA
export PYOPENGL_PLATFORM=egl

# Auto-activate local virtualenv if present
if [ -f "${ROOT_DIR}/env/bin/activate" ]; then
  # shellcheck disable=SC1091
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

# LoRA Fine-tuning with HF Model
"${TORCHRUN_BIN}" --standalone --nnodes 1 --nproc-per-node 4 scripts/finetune_openvla.py \
  --vla_path /models/zhangguoxi/openvla-7b \
  --data_root_dir /datasets/zhangguoxi/modified_libero_rlds \
  --dataset_name "${data_name}" \
  --use_lora True \
  --image_aug True \
  --use_hf_model True \
  --training_mode lora \
  --use_lora True \
  --lora_rank 64 \
  --lora_dropout 0.0 \
  --batch_size 8 \
  --max_steps 5005 \
  --save_steps 5000 \
  --learning_rate 2e-4 \
  --grad_accumulation_steps 2 \
  --image_aug True \
  --shuffle_buffer_size 100000 \
  --save_latest_checkpoint_only True \
  --wandb_entity "yiyangchen-sylvia-bigai" \
  --wandb_project "openvla-lora" \
  --run_id_note "lora-${data_name}--${current_time}" \
  > "logs/openvla-lora-${data_name}--${current_time}.log" 2>&1

echo "[INFO] Launched LoRA fine-tuning for ${data_name}."
echo "[INFO] Log file: logs/openvla-lora-${data_name}--${current_time}.log"

