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

if [ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]; then
  . "${HOME}/miniconda3/etc/profile.d/conda.sh"
fi

source "${ROOT_DIR}/scripts/setup_libero_env.sh"

export MUJOCO_GL=osmesa
unset PYOPENGL_PLATFORM

PY_BIN="${ROOT_DIR}/env/bin/python"

CUDA_VISIBLE_DEVICES=0 "${PY_BIN}" experiments/robot/libero/run_libero_eval.py \
  --use_proprio True \
  --num_images_in_input 2 \
  --use_film False \
  --pretrained_checkpoint outputs/LIBERO-Spatial-Pro \
  --task_suite_name libero_spatial \
  --use_pro_version True \
  "$@"

