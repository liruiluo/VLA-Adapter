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

mkdir -p eval_logs hf_cache timm_cache

# Redirect HF / timm cache to project disk (avoid small $HOME quota)
export HF_HOME="${ROOT_DIR}/hf_cache"
export TIMM_CACHE_DIR="${ROOT_DIR}/timm_cache"

# Optional: load user conda base, then project env via setup_libero_env.sh
if [ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]; then
  . "${HOME}/miniconda3/etc/profile.d/conda.sh"
fi

source "${ROOT_DIR}/scripts/setup_libero_env.sh"

export MUJOCO_GL=osmesa
unset PYOPENGL_PLATFORM

PY_BIN="${ROOT_DIR}/env/bin/python"

# === LoRA (merged) LIBERO-Spatial eval ===

CKPT_DIR="${CKPT_DIR:-}"
if [ -z "${CKPT_DIR}" ]; then
  echo "[ERROR] Please set CKPT_DIR to a merged LoRA checkpoint directory (HF-style dir containing config.json/model files)." >&2
  exit 1
fi

CONFIG_DIR="${CONFIG_DIR:-pretrained_models/configs}"
RUN_ID_NOTE="${RUN_ID_NOTE:-lora-eval}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES-0}" "${PY_BIN}" experiments/robot/libero/run_libero_eval.py \
  --model_family openvla \
  --pretrained_checkpoint "${CKPT_DIR}" \
  --config_file_path "${CONFIG_DIR}" \
  --use_film False \
  --num_images_in_input 2 \
  --use_proprio True \
  --task_suite_name libero_spatial \
  --num_trials_per_task 50 \
  --use_pro_version True \
  --local_log_dir "./eval_logs" \
  --run_id_note "${RUN_ID_NOTE}" \
  "$@"

