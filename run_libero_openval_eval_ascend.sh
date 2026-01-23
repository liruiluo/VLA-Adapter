#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

# ========================================
# 1. 初始化昇腾NPU环境（关键：必须在最前面）
# ========================================
echo "[INFO] 正在初始化昇腾NPU环境..."
if [ -f "/usr/local/Ascend/ascend-toolkit/set_env.sh" ]; then
  source /usr/local/Ascend/ascend-toolkit/set_env.sh
  echo "[INFO] ✓ 昇腾环境已加载，TBE模块可用"
else
  echo "[WARN] ⚠ 昇腾set_env.sh未找到，可能导致NPU初始化失败"
fi

if [ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]; then
  . "${HOME}/miniconda3/etc/profile.d/conda.sh"
fi

source "${ROOT_DIR}/setup_libero_env.sh"

# ========================================
# 2. 渲染配置：OSMesa（CPU软件渲染）
#    - NPU不支持EGL，必须使用CPU渲染
#    - 性能影响小：渲染仅占5-10ms/帧
# ========================================
export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa # egl不适配 npu
echo "[INFO] 使用OSMesa进行MuJoCo渲染(CPU渲染)"

# ========================================
# 3. NPU推理加速配置
#    - 性能关键：模型推理占50-200ms/帧
#    - 使用NPU加速可提升3-5倍速度
# ========================================
export ASCEND_VISIBLE_DEVICES=1  
echo "[INFO] 使用NPU设备 $ASCEND_VISIBLE_DEVICES 进行模型推理加速"

# Set model type for ACTION_TOKEN_BEGIN_IDX detection
# For original OpenVLA (not VLA-Adapter), use OPENVLA
export VLA_MODEL_TYPE=OPENVLA

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

export PYTHONPATH=/root/sylvia/VLA-Adapter:/root/sylvia/VLA-Adapter/LIBERO:${PYTHONPATH:-}

# 创建 LIBERO 配置文件，避免交互式输入
LIBERO_CONFIG_DIR="$HOME/.libero"
LIBERO_CONFIG_FILE="$LIBERO_CONFIG_DIR/config.yaml"
LIBERO_ROOT="/root/sylvia/VLA-Adapter/LIBERO"

if [ ! -f "$LIBERO_CONFIG_FILE" ]; then
    mkdir -p "$LIBERO_CONFIG_DIR"
    python3 << EOF
import yaml
import os

config = {
    "benchmark_root": "$LIBERO_ROOT/libero/libero",
    "bddl_files": "$LIBERO_ROOT/libero/libero/bddl_files",
    "init_states": "$LIBERO_ROOT/libero/libero/init_files",
    "datasets": "/datasets/zhangguoxi/modified_libero_rlds/",
    "assets": "$LIBERO_ROOT/libero/libero/assets",
}

with open("$LIBERO_CONFIG_FILE", "w") as f:
    yaml.dump(config, f)
print(f"Created LIBERO config file: $LIBERO_CONFIG_FILE")
EOF
fi

data_name=libero_object
current_time=$(date "+%Y-%m-%d_%H-%M-%S")

"${TORCHRUN_BIN}" --standalone --nnodes 1 --nproc-per-node 1 experiments/robot/libero/run_libero_eval.py \
  --model_family openvla \
  --save_version openvla-lora \
  --use_proprio False \
  --num_images_in_input 1 \
  --use_l1_regression False \
  --num_open_loop_steps 1 \
  --use_film False \
  --center_crop True \
  --pretrained_checkpoint /root/sylvia/VLA-Adapter/outputs/openvla-7b+libero_object_no_noops+b16+lr-0.0002+lora-r64+dropout-0.0--lora-libero_object_no_noops--2026-01-20_15-57-22--image_aug--5000_chkpt \
  --task_suite_name ${data_name} \
  > eval_logs/${data_name}-OpenVLA-sft-chkpt-5k--${current_time}.log 2>&1 
