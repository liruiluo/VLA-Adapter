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

# Ensure training logs (including debug prints) stream to the log file immediately.
export PYTHONUNBUFFERED=1

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

echo "[INFO] Starting 4-GPU finetune for ${data_name} (1 trajectory per task)..."

# 4-GPU (>=80GB total) training VLA-Adapter-Pro on LIBERO-Object
# This ablation keeps up to 1 trajectory per unique language_instruction (task).
CUDA_VISIBLE_DEVICES=0,1,2,3 \
"${TORCHRUN_BIN}" --standalone --nnodes 1 --nproc-per-node 4 vla-scripts/finetune.py \
  --vlm_path pretrained_models/prism-qwen25-extra-dinosiglip-224px-0_5b \
  --config_file_path pretrained_models/configs \
  --data_root_dir data/libero \
  --dataset_name "${data_name}" \
  --run_root_dir outputs \
  --max_trajectories_per_task 1 \
  --use_film False \
  --num_images_in_input 2 \
  --use_proprio True \
  --use_lora True \
  --use_fz False \
  --use_minivlm True \
  --image_aug True \
  --num_steps_before_decay 5000 \
  --max_steps 5000 \
  --save_freq 5000 \
  --save_latest_checkpoint_only False \
  --merge_lora_during_training True \
  --batch_size 4 \
  --grad_accumulation_steps 4 \
  --learning_rate 2e-4 \
  --lora_rank 64 \
  --use_pro_version True \
  --wandb_entity "YOUR_WANDB_ENTITY" \
  --wandb_project "${data_name}" \
  --run_id_note "VLA-Adapter--object-4GPU--1traj-per-task--${current_time}" \
  "$@" \
  > "logs/VLA-Adapter--object-4GPU--1traj-per-task--${current_time}.log" 2>&1

echo "[INFO] Finished 4-GPU finetune job for ${data_name} (1 trajectory per task)."
echo "[INFO] Log file: logs/VLA-Adapter--object-4GPU--1traj-per-task--${current_time}.log"
