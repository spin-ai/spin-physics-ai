"""7 个扰动维度各渲一张，拼成 grid（验证 bddl / init states / 新资产都到位）。

    ./scripts/run_liberoplus.sh python scripts/smoke_test_categories.py
    BENCH_NAME=libero_goal ./scripts/run_liberoplus.sh python scripts/smoke_test_categories.py

产物落在 `benchmarks/LIBERO-plus/outputs/`（该目录不入库）。
"""
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np


def _libero_root() -> Path:
    env = os.environ.get("LIBERO_PLUS_DIR")
    cand = (Path(env).expanduser() if env
            else Path(__file__).resolve().parent.parent / "benchmarks" / "LIBERO-plus")
    root = cand.resolve()
    if not (root / "libero").is_dir():
        sys.exit(f"✗ 找不到 LIBERO-plus: {root}\n  先跑 ./scripts/fetch.py libero（或设 LIBERO_PLUS_DIR）")
    return root


LIBERO_ROOT = _libero_root()
sys.path.insert(0, str(LIBERO_ROOT))   # 上游是 src 布局，editable 安装不可靠，直接进 sys.path
os.chdir(LIBERO_ROOT)

from libero.libero import benchmark  # noqa: E402
from libero.libero.envs import OffScreenRenderEnv  # noqa: E402

SUITE = os.environ.get("BENCH_NAME", "libero_object")
CLASSIFICATION = json.load(
    open(LIBERO_ROOT / "libero/libero/benchmark/task_classification.json")
)[SUITE]

picks = {}
for item in CLASSIFICATION:            # 每个扰动类别取第一个任务
    picks.setdefault(item["category"], item)

suite = benchmark.get_benchmark_dict()[SUITE]()
os.makedirs("outputs", exist_ok=True)
images = []
for cat, item in picks.items():
    task_id = item["id"]               # json 里的 id 是 1-based
    task = suite.get_task(task_id)
    bddl = suite.get_task_bddl_file_path(task_id)
    print(f"\n=== {cat} (level {item['difficulty_level']}) task_id={task_id} ===")
    print("name:", task.name)
    print("lang:", task.language)
    env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=256, camera_widths=256)
    env.seed(0)
    env.reset()
    env.set_init_state(suite.get_task_init_states(task_id)[0])
    obs = None
    for _ in range(5):
        obs, rew, done, _ = env.step([0, 0, 0, 0, 0, 0, -1.0])
    img = obs["agentview_image"][::-1, :, ::-1].copy()
    env.close()
    cv2.imwrite(f"outputs/category_{cat.lower().replace(' ', '_')}.png", img)
    cv2.putText(img, cat[:22], (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    images.append(img)

row1 = np.concatenate(images[:4], axis=1)
row2 = np.concatenate(images[4:] + [np.zeros_like(images[0])] * (4 - len(images[4:])), axis=1)
cv2.imwrite("outputs/categories_grid.png", np.concatenate([row1, row2], axis=0))
print("\nsaved", LIBERO_ROOT / "outputs/categories_grid.png")
