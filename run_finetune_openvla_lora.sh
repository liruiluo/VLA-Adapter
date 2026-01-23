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
"${TORCHRUN_BIN}" --standalone --nnodes 1 --nproc-per-node 4 scripts/finetune.py \
  --vlm_path /models/zhangguoxi/openvla-7b \
  --config_file_path /root/sylvia/VLA-Adapter/prismatic/extern/hf \
  --data_root_dir /datasets/zhangguoxi/modified_libero_rlds \
  --dataset_name "${data_name}" \
  --run_root_dir outputs \
  --vla_arch openvla \
  --use_l1_regression False \
  --use_film False \
  --num_images_in_input 1 \
  --use_minivlm False \
  --use_wrist_img False \
  --use_proprio False \
  --use_pro_version False \
  --use_fz False \
  --use_lora True \
  --image_aug True \
  --lora_rank 64 \
  --lora_dropout 0.0 \
  --batch_size 12 \
  --max_steps 50005 \
  --save_freq 25000 \
  --learning_rate 5e-4 \
  --grad_accumulation_steps 1 \
  --wandb_entity "yiyangchen-sylvia-bigai" \
  --wandb_project "openvla-lora" \
  --run_id_note "Openvla-lora-${data_name}--${current_time}" \
  > "logs/openvla-lora-${data_name}--${current_time}.log" 2>&1

echo "[INFO] Launched LoRA fine-tuning for ${data_name}."
echo "[INFO] Log file: logs/openvla-lora-${data_name}--${current_time}.log"
