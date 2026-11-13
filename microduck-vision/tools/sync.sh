#!/usr/bin/env bash
# 把宿主上的代码同步进运行中的容器，然后跑一条命令。
#   ./tools/sync.sh                                  → selftest
#   ./tools/sync.sh eval --episodes 20 --backends none,color
#   ./tools/sync.sh python tools/export_3d.py --seed 1   → 直接跑脚本
set -euo pipefail

CONTAINER="${MV_CONTAINER:-microduck}"
HERE="$(cd "$(dirname "$0")/.." && pwd)"
DEST=/data/vision

docker exec -u root "$CONTAINER" mkdir -p "$DEST"
docker cp "$HERE/." "$CONTAINER:$DEST"
docker exec -u root "$CONTAINER" chown -R duck:duck "$DEST"

if [ "$#" -eq 0 ]; then
  set -- selftest
fi

docker exec "$CONTAINER" sh -c '
  export MUJOCO_GL=osmesa
  export MICRODUCK_MOSS_DIR="${MICRODUCK_MOSS_DIR:-/data/moss}"
  export PYTHONPATH=/data/vision:/app/microduck_local/src
  cd /data/vision
  if [ "$1" = "python" ]; then
    shift
    exec /opt/venv/bin/python "$@"
  fi
  exec /opt/venv/bin/python -m mvision.cli "$@"
' sh "$@"
