#!/usr/bin/env bash
set -euo pipefail

# Resolve repo root (works both locally and under Slurm)
if [ -n "${SLURM_SUBMIT_DIR-}" ]; then
  ROOT_DIR="${SLURM_SUBMIT_DIR}"
else
  ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
cd "${ROOT_DIR}"

mkdir -p logs

# Redirect HF / timm cache to project disk (avoid small $HOME quota)
export HF_HOME="${ROOT_DIR}/hf_cache"
export TIMM_CACHE_DIR="${ROOT_DIR}/timm_cache"
mkdir -p "${HF_HOME}" "${TIMM_CACHE_DIR}"

# For consistency with eval env; harmless for pure training
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

# Prefer the project-local torchrun if available
TORCHRUN_BIN="${ROOT_DIR}/env/bin/torchrun"
if [ ! -x "${TORCHRUN_BIN}" ]; then
  TORCHRUN_BIN="torchrun"
fi

data_name=libero_object_no_noops
current_time=$(date "+%Y-%m-%d_%H-%M-%S")

echo "[INFO] Starting 4-GPU MoE-LoRA finetune for ${data_name}..."

# 4-GPU (>=80GB total) MoE-LoRA training on LIBERO-Object
CUDA_VISIBLE_DEVICES=0,1,2,3 \
"${TORCHRUN_BIN}" --standalone --nnodes 1 --nproc-per-node 4 vla-scripts/finetune.py \
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
  --lora_rank 64 \
  --use_fz False \
  --num_steps_before_decay 150000 \
  --max_steps 5000 \
  --save_freq 5000 \
  --save_latest_checkpoint_only False \
  --merge_lora_during_training False \
  --batch_size 8 \
  --grad_accumulation_steps 2 \
  --learning_rate 2e-4 \
  --use_pro_version True \
  --wandb_entity "YOUR_WANDB_ENTITY" \
  --wandb_project "${data_name}" \
  --run_id_note "VLA-Adapter-MoELoRA--object-4GPU--${current_time}" \
  "$@" \
  > "logs/VLA-Adapter-MoELoRA--object-4GPU--${current_time}.log" 2>&1

echo "[INFO] Finished 4-GPU MoE-LoRA finetune job for ${data_name}."
echo "[INFO] Log file: logs/VLA-Adapter-MoELoRA--object-4GPU--${current_time}.log"

