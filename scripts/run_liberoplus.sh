#!/usr/bin/env bash
# 在 conda env `liberoplus` 里跑任意命令（LIBERO-plus 已 `pip install -e`）。
#
#   ./scripts/run_liberoplus.sh python -c "import libero; print(libero.__file__)"
#   ./scripts/run_liberoplus.sh python scripts/smoke_test.py
#   ./scripts/run_liberoplus.sh bash          # 进一个已经准备好的 shell
#
# 也可以 source 进当前 shell（不带命令时只准备环境，不会 exec 掉你的 shell）：
#   source scripts/run_liberoplus.sh
#
# 需要换仓库位置时：LIBERO_PLUS_DIR=/path/to/LIBERO-plus ./scripts/run_liberoplus.sh ...
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export LIBERO_PLUS_DIR="${LIBERO_PLUS_DIR:-$(cd "$HERE/.." && pwd)/benchmarks/LIBERO-plus}"

if [ ! -d "$LIBERO_PLUS_DIR/libero" ]; then
  echo "✗ 找不到 LIBERO-plus: $LIBERO_PLUS_DIR" >&2
  echo "  先跑 ./scripts/fetch.py libero（或设 LIBERO_PLUS_DIR）" >&2
  exit 1
fi

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate liberoplus

export MAGICK_HOME="${MAGICK_HOME:-/opt/homebrew/opt/imagemagick}"   # wand 需要 ImageMagick
export MUJOCO_GL="${MUJOCO_GL:-glfw}"                                # macOS 离屏渲染
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-glfw}"
export LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-$HOME/.libero}"
# 上游是 src 布局（libero/libero/...），pip install -e 在这里不产生映射，直接把仓库根加进路径
export PYTHONPATH="$LIBERO_PLUS_DIR${PYTHONPATH:+:$PYTHONPATH}"

if [ "$#" -eq 0 ]; then
  echo "环境就绪：conda=liberoplus  LIBERO_PLUS_DIR=$LIBERO_PLUS_DIR"
else
  "$@"
fi
