#!/usr/bin/env bash
set -euo pipefail

# Resolve repo root (works locally, from anywhere, and under Slurm)
if [ -n "${SLURM_SUBMIT_DIR-}" ] && [ -d "${SLURM_SUBMIT_DIR}" ]; then
  ROOT_DIR="${SLURM_SUBMIT_DIR}"
elif command -v git >/dev/null 2>&1 && git -C "${PWD}" rev-parse --show-toplevel >/dev/null 2>&1; then
  ROOT_DIR="$(git -C "${PWD}" rev-parse --show-toplevel)"
else
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
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

data_name=libero_10_no_noops
current_time=$(date "+%Y-%m-%d_%H-%M-%S")

echo "[INFO] Starting 4-GPU finetune (tokenized actions) for ${data_name}..."

# 4-GPU (>=80GB total) training VLA-Adapter-Pro on LIBERO-Long (LIBERO-10) (offline subset: 1 traj / task).
CUDA_VISIBLE_DEVICES=0,1,2,3 \
"${TORCHRUN_BIN}" --standalone --nnodes 1 --nproc-per-node 4 vla-scripts/finetune.py \
  --vlm_path pretrained_models/prism-qwen25-extra-dinosiglip-224px-0_5b \
  --config_file_path pretrained_models/configs \
  --data_root_dir data/libero_subsets \
  --dataset_name "${data_name}" \
  --run_root_dir outputs \
  --use_film False \
  --num_images_in_input 2 \
  --use_proprio True \
  --use_l1_regression False \
  --use_diffusion False \
  --use_lora True \
  --use_fz False \
  --use_minivlm True \
  --image_aug True \
  --num_steps_before_decay 30000 \
  --max_steps 30005 \
  --save_freq 30000 \
  --save_latest_checkpoint_only False \
  --merge_lora_during_training True \
  --batch_size 16 \
  --grad_accumulation_steps 1 \
  --learning_rate 2e-4 \
  --lora_rank 64 \
  --use_pro_version True \
  --wandb_entity "YOUR_WANDB_ENTITY" \
  --wandb_project "${data_name}" \
  --run_id_note "VLA-Adapter--token--long-4GPU--${data_name}--${current_time}" \
  "$@" \
  > "logs/VLA-Adapter--token--long-4GPU--${data_name}--${current_time}.log" 2>&1

echo "[INFO] Finished 4-GPU finetune job for ${data_name}."
echo "[INFO] Log file: logs/VLA-Adapter--token--long-4GPU--${data_name}--${current_time}.log"

