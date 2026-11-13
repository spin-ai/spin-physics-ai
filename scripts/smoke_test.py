"""最小自检：建环境 → reset → step → 渲染。

    ./scripts/run_liberoplus.sh python scripts/smoke_test.py
    BENCH_NAME=libero_goal TASK_ID=3 ./scripts/run_liberoplus.sh python scripts/smoke_test.py

产物落在 `benchmarks/LIBERO-plus/outputs/`（该目录不入库）。
"""
import os
import sys
from pathlib import Path

import cv2
import numpy as np


def _libero_root() -> Path:
    """按 LIBERO_PLUS_DIR / 脚本位置解析上游仓库，并切到该目录再干活。"""
    env = os.environ.get("LIBERO_PLUS_DIR")
    cand = (Path(env).expanduser() if env
            else Path(__file__).resolve().parent.parent / "benchmarks" / "LIBERO-plus")
    root = cand.resolve()
    if not (root / "libero").is_dir():
        sys.exit(f"✗ 找不到 LIBERO-plus: {root}\n  先跑 ./scripts/fetch.py libero（或设 LIBERO_PLUS_DIR）")
    return root


LIBERO_ROOT = _libero_root()
sys.path.insert(0, str(LIBERO_ROOT))   # 上游是 src 布局，editable 安装不可靠，直接进 sys.path
os.chdir(LIBERO_ROOT)          # outputs/ 相对路径落在 benchmarks/LIBERO-plus/ 下（已 gitignore）

from libero.libero import benchmark, get_libero_path  # noqa: E402
from libero.libero.envs import OffScreenRenderEnv  # noqa: E402

BENCH_NAME = os.environ.get("BENCH_NAME", "libero_object")
TASK_ID = int(os.environ.get("TASK_ID", "0"))

print("LIBERO-plus:", LIBERO_ROOT)
print("benchmarks:", list(benchmark.get_benchmark_dict().keys()))
print("assets path:", get_libero_path("assets"))

task_suite = benchmark.get_benchmark_dict()[BENCH_NAME]()
print("num tasks:", task_suite.get_num_tasks())
task = task_suite.get_task(TASK_ID)
print("task:", task.name, "| lang:", task.language)

bddl_file = task_suite.get_task_bddl_file_path(TASK_ID)
print("bddl:", bddl_file)

env = OffScreenRenderEnv(bddl_file_name=bddl_file, camera_heights=256, camera_widths=256)
env.seed(0)
env.reset()
obs = env.set_init_state(task_suite.get_task_init_states(TASK_ID)[0])

for _ in range(10):
    obs, rew, done, info = env.step([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0])

img = obs["agentview_image"][::-1]           # mujoco 渲染是上下颠倒的
wrist = obs["robot0_eye_in_hand_image"][::-1]
print("agentview:", img.shape, img.dtype, "reward:", rew, "done:", done)

os.makedirs("outputs", exist_ok=True)
cv2.imwrite("outputs/smoke_agentview.png", img[:, :, ::-1])
cv2.imwrite("outputs/smoke_wrist.png", wrist[:, :, ::-1])
print("saved", LIBERO_ROOT / "outputs/smoke_agentview.png")
env.close()
