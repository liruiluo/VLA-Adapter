#!/usr/bin/env bash
set -euo pipefail

# Resolve repo root (works locally, from anywhere, and under Slurm)
if [ -n "${SLURM_SUBMIT_DIR-}" ] && [ -d "${SLURM_SUBMIT_DIR}" ]; then
  ROOT_DIR="${SLURM_SUBMIT_DIR}"
elif command -v git >/dev/null 2>&1 && git -C "${PWD}" rev-parse --show-toplevel >/dev/null 2>&1; then
  ROOT_DIR="$(git -C "${PWD}" rev-parse --show-toplevel)"
else
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
fi
cd "${ROOT_DIR}"

echo "[INFO] Repo root: ${ROOT_DIR}"

# Ensure repo root is on PYTHONPATH so imports like `experiments.*` work.
case ":${PYTHONPATH-}:" in
  *":${ROOT_DIR}:"*) ;;
  *) export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH-}" ;;
esac

# 1) 尝试激活本项目自带的 conda 环境 ./env
#    注意：只有在 *source* 这个脚本时（例如：`source scripts/setup_libero_env.sh`），
#    conda activate 对当前 shell 才是有效的；直接 `bash scripts/setup_libero_env.sh`
#    只会在子进程里生效。
if command -v conda >/dev/null 2>&1; then
  # 优先按绝对路径激活，失败就尝试相对路径；若都失败则给出提示但不中断。
  if conda activate "${ROOT_DIR}/env" 2>/dev/null; then
    echo "[INFO] 已通过 conda activate 激活环境: ${ROOT_DIR}/env"
  elif conda activate ./env 2>/dev/null; then
    echo "[INFO] 已通过 conda activate 激活环境: ./env"
  else
    echo "[WARN] 未能通过 conda activate ./env，继续使用当前环境"
  fi
else
  echo "[WARN] 未检测到 conda 命令，跳过 conda activate ./env"
fi

# 2) 选择 python 解释器：优先使用当前 shell 中的 python
PYTHON_BIN="$(command -v python || echo python)"
echo "[INFO] Using python: ${PYTHON_BIN}"

# 3) 把 LIBERO 源码目录挂到 PYTHONPATH（不依赖 pip 是否成功安装 libero）。
case ":${PYTHONPATH-}:" in
  *":${ROOT_DIR}/LIBERO:"*) ;;
  *) export PYTHONPATH="${ROOT_DIR}/LIBERO:${PYTHONPATH-}" ;;
esac
echo "[INFO] 已将 ${ROOT_DIR}/LIBERO 加入 PYTHONPATH"

# 4) 使用 LIBERO 提供的 API 设置 ~/.libero/config.yaml
#    - 若首次使用且 config 不存在，导入时可能会提示交互问题，按 Y/N 回答即可；
#    - 若 config 已存在，本调用会直接覆盖为当前仓库下的 LIBERO 路径。
echo "[INFO] 配置 ~/.libero/config.yaml 中的 LIBERO 路径..."
"${PYTHON_BIN}" - <<'PYCODE'
from pathlib import Path

from libero.libero import set_libero_default_path

root = Path("LIBERO/libero/libero").resolve()
set_libero_default_path(str(root))
PYCODE

echo
echo "[DONE] LIBERO 源码已通过 PYTHONPATH 暴露，且路径配置已更新。"
echo "      之后在当前 shell 中可以直接运行（示例）："
echo
echo "CUDA_VISIBLE_DEVICES=0 ${PYTHON_BIN} experiments/robot/libero/run_libero_eval.py \\"
echo "  --use_proprio True \\"
echo "  --num_images_in_input 2 \\"
echo "  --use_film False \\"
echo "  --pretrained_checkpoint outputs/LIBERO-Object-Pro \\"
echo "  --task_suite_name libero_object \\"
echo "  --use_pro_version True"
