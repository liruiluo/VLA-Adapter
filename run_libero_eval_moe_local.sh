#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

# Optional: load user conda base, then project env via setup_libero_env.sh
if [ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]; then
  . "${HOME}/miniconda3/etc/profile.d/conda.sh"
fi

source "${ROOT_DIR}/setup_libero_env.sh"

export MUJOCO_GL=osmesa
unset PYOPENGL_PLATFORM

# Force use of project conda env python (has draccus, libero, etc.)
PY_BIN="${ROOT_DIR}/env/bin/python"

# === MoE-LoRA LIBERO-Object eval (local smoke test) ===

# MoE-LoRA finetune checkpoint directory (adjust if you run a new job)
CKPT_DIR="outputs/configs+libero_object_no_noops+b16+lr-0.0002+moe-lora-e3-r64+dropout-0.05--image_aug--VLA-Adapter-MoELoRA--object-4GPU--2025-12-13_10-01-16--5000_chkpt"

# Base OpenVLA config used to reconstruct the model skeleton
CONFIG_DIR="pretrained_models/configs"

CUDA_VISIBLE_DEVICES=0 "${PY_BIN}" experiments/robot/libero/run_libero_eval.py \
  --model_family openvla \
  --pretrained_checkpoint "${CKPT_DIR}" \
  --config_file_path "${CONFIG_DIR}" \
  --use_moe_lora True \
  --moe_num_experts 3 \
  --moe_top_k 2 \
  --moe_target_modules "all-linear" \
  --lora_rank 64 \
  --use_l1_regression True \
  --use_minivlm True \
  --use_film False \
  --num_images_in_input 2 \
  --use_proprio True \
  --task_suite_name libero_object \
  --num_trials_per_task 50 \
  --use_pro_version True \
  "$@"
