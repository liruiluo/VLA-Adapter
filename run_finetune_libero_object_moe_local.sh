#!/usr/bin/env bash
set -euo pipefail

# Resolve repo root based on script location
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

mkdir -p logs
mkdir -p hf_cache timm_cache

# Redirect HF / timm cache to project disk (avoid small $HOME quota)
export HF_HOME="${ROOT_DIR}/hf_cache"
export TIMM_CACHE_DIR="${ROOT_DIR}/timm_cache"

# Use EGL for consistency with eval (harmless for training)
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

# Auto-activate local conda/venv at ./env if present
if [ -f "${ROOT_DIR}/env/bin/activate" ]; then
  # shellcheck disable=SC1091
  source "${ROOT_DIR}/env/bin/activate"
fi

# Prefer the project-local torchrun if available
TORCHRUN_BIN="${ROOT_DIR}/env/bin/torchrun"
if [ ! -x "${TORCHRUN_BIN}" ]; then
  TORCHRUN_BIN="torchrun"
fi

data_name=libero_object_no_noops
current_time=$(date "+%Y-%m-%d_%H-%M-%S")

# 单卡本地 MoE-LoRA 微调示例，先跑小步数观察显存和 loss 再放大
CUDA_VISIBLE_DEVICES=0 \
"${TORCHRUN_BIN}" --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/finetune.py \
  --vlm_path pretrained_models/prism-qwen25-extra-dinosiglip-224px-0_5b \
  --config_file_path pretrained_models/configs \
  --data_root_dir data/libero \
  --dataset_name "${data_name}" \
  --run_root_dir outputs \
  --use_film False \
  --num_images_in_input 2 \
  --use_proprio True \
  --use_minivlm True \
  --image_aug True \
  --use_lora False \
  --use_moe_lora True \
  --moe_num_experts 3 \
  --moe_target_modules "all-linear" \
  --moe_top_k 2 \
  --lora_rank 8 \
  --lora_dropout 0.05 \
  --use_fz False \
  --num_steps_before_decay 200000 \
  --max_steps 50 \
  --save_freq 50 \
  --save_latest_checkpoint_only False \
  --merge_lora_during_training False \
  --batch_size 2 \
  --grad_accumulation_steps 16 \
  --learning_rate 2e-4 \
  --use_pro_version True \
  --wandb_entity "YOUR_WANDB_ENTITY" \
  --wandb_project "${data_name}" \
  --run_id_note "VLA-Adapter-MoELoRA--libero_object_no_noops--${current_time}" \
  "$@" \
  > "logs/VLA-Adapter-MoELoRA--libero_object_no_noops--${current_time}.log" 2>&1 &

echo "[INFO] Launched local MoE-LoRA finetune for ${data_name}."
echo "[INFO] Log file: logs/VLA-Adapter-MoELoRA--libero_object_no_noops--${current_time}.log"
