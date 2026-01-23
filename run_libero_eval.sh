#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

if [ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]; then
  . "${HOME}/miniconda3/etc/profile.d/conda.sh"
fi

source "${ROOT_DIR}/setup_libero_env.sh"

# 渲染配置 - 使用OSMesa避免EGL问题
export MUJOCO_GL=egl

# 补充 PYTHONPATH，避免 import 问题
export PYTHONPATH=/root/sylvia/VLA-Adapter:/root/sylvia/VLA-Adapter/LIBERO

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

ASCEND_RT_VISIBLE_DEVICES=0 python experiments/robot/libero/run_libero_eval.py \
  --use_proprio True \
  --num_images_in_input 2 \
  --use_film False \
  --pretrained_checkpoint outputs/hf+libero_object_no_noops+b16+lr-0.0002+lora-r64+dropout-0.0--image_aug--VLA-Adapter--libero_object_no_noops--2026-01-16_17-48-44--5000_chkpt \
  --task_suite_name libero_object \
  --use_pro_version True \
  > eval_logs/Object-VlaAdapter-sft-chkpt-5k.log 2>&1 
