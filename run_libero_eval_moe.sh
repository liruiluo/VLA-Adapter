#!/usr/bin/env bash
set -euo pipefail

# In Slurm, the script may be copied to a spool dir, but CWD stays where you submitted the job.
ROOT_DIR="${PWD}"
cd "${ROOT_DIR}"

mkdir -p eval_logs hf_cache timm_cache

# Redirect HF / timm cache to project disk (avoid small $HOME quota)
export HF_HOME="${ROOT_DIR}/hf_cache"
export TIMM_CACHE_DIR="${ROOT_DIR}/timm_cache"

# Optional: load user conda base, then project env via setup_libero_env.sh
if [ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]; then
  . "${HOME}/miniconda3/etc/profile.d/conda.sh"
fi

source "${ROOT_DIR}/setup_libero_env.sh"

export MUJOCO_GL=osmesa
unset PYOPENGL_PLATFORM

# Force use of project conda env python (has draccus, libero, etc.)
PY_BIN="${ROOT_DIR}/env/bin/python"

# === MoE-LoRA LIBERO-Object eval ===

CKPT_DIR="${CKPT_DIR:-${ROOT_DIR}/outputs/configs+libero_object_no_noops+b16+lr-0.0002+moe-lora-e3-r64+dropout-0.0--image_aug--VLA-Adapter-MoELoRA--object-4GPU--2025-12-14_10-19-42--5000_chkpt}"
CONFIG_DIR="pretrained_models/configs"

if [ ! -d "${CKPT_DIR}" ]; then
  echo "[ERROR] CKPT_DIR is not a directory: ${CKPT_DIR}" >&2
  echo "[ERROR] Repo root (PWD): ${ROOT_DIR}" >&2
  echo "[HINT] If running under Slurm, avoid hard-coded /home paths; use repo-relative outputs/ or set CKPT_DIR env var." >&2
  exit 1
fi

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES-0}" "${PY_BIN}" experiments/robot/libero/run_libero_eval.py \
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
  --local_log_dir "./eval_logs" \
  --run_id_note "moe-lora-e3-r64-step5000" \
  "$@"
