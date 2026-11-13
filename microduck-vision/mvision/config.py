"""路径、分辨率、约定 —— 全部可由环境变量覆盖，不在代码里硬编码绝对路径。"""

from __future__ import annotations

import os
from pathlib import Path

#: 新代码永远只写这里（NFR-4：/app 只读）。容器里 /data 是 named volume。
#: 两个变量都认：`MV_DATA` 是本包的，`MVISION_DATA` 是 Dockerfile 里钉的。
DATA_ROOT = Path(os.environ.get("MV_DATA")
                 or os.environ.get("MVISION_DATA")
                 or "/data/captures")
#: 采集产物
CAPTURE_ROOT = Path(os.environ.get("MV_CAPTURES", str(DATA_ROOT / "episodes")))
#: 报告 / 指标
REPORT_ROOT = Path(os.environ.get("MV_REPORTS", str(DATA_ROOT / "reports")))

#: microduck_local 的源码目录（容器里是 /app/microduck_local/src）
MICRODUCK_SRC = os.environ.get("MICRODUCK_SRC", "/app/microduck_local/src")

#: 渲染分辨率。320x240 实测 24 ms/帧（osmesa 软件光栅）。
RENDER_W = int(os.environ.get("MV_WIDTH", "320"))
RENDER_H = int(os.environ.get("MV_HEIGHT", "240"))

#: 深度超过它就算背景（osmesa 下背景深度是内部 far≈70.7，不是 zfar）
MAX_RANGE_M = float(os.environ.get("MV_MAX_RANGE", "2.0"))

#: 工作区 ROI（rover 系），用于裁掉地板远端与机器人自身
ROI = tuple(float(v) for v in os.environ.get("MV_ROI", "0.05,0.75,-0.45,0.45,-0.02,0.60").split(","))

#: 相机名（MJCF <camera> 元素），见 scene.build
CAM_FRONT = "front_camera"
CAM_WRIST = "wrist_camera"

#: 场景里 rover 的 body 名 —— "base frame" 就是它
FRAME_BODY = "rover"

#: 目标物体的显式颜色（color 语义后端靠它分割）。
#: **不能是橙色** —— MOSS 自己的身体就是橙红 (0.93, 0.45, 0.38)，实测在
#: 0.12 m 距离上整个前视画面被自家机械臂占满，橙色分割会把机器人当成目标。
TARGET_RGBA = (0.80, 0.10, 0.75, 1.0)
CLUTTER_RGBA = (
    (0.30, 0.32, 0.36, 1.0),
    (0.20, 0.35, 0.60, 1.0),
    (0.25, 0.45, 0.28, 1.0),
    (0.55, 0.52, 0.30, 1.0),
)

#: 感知契约里的新鲜度门限，复用仓库自己的语义量级（moss_env.ARM_STALE_S ≈ 0.2 s）
STALE_S = float(os.environ.get("MV_STALE_S", "0.25"))

#: 外部模型后端（MV_BACKEND=http 时使用）
HTTP_BASE = os.environ.get("MV_HTTP_BASE", "")
HTTP_MODEL = os.environ.get("MV_HTTP_MODEL", "gpt-4o-mini")
HTTP_KEY = os.environ.get("MV_HTTP_KEY", os.environ.get("OPENAI_API_KEY", ""))
HTTP_TIMEOUT = float(os.environ.get("MV_HTTP_TIMEOUT", "20"))


def ensure_dirs() -> None:
    for d in (CAPTURE_ROOT, REPORT_ROOT):
        d.mkdir(parents=True, exist_ok=True)
