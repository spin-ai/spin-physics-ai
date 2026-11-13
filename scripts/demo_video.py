"""脚本化 pick-and-place 演示：能录 MP4，也能开真窗口看机械臂动。

    ./scripts/run_liberoplus.sh python scripts/demo_video.py                  # 录 output/demo_pick.mp4
    ./scripts/run_liberoplus.sh python scripts/demo_video.py --live            # 开真 3D 窗口（实时）
    ./scripts/run_liberoplus.sh python scripts/demo_video.py --live --repeat 6 --hold 180

控制器是末端执行器的 P 控制（不是学出来的策略）：移到物体上方 → 下降 → 合爪 → 抬起
→ 移到篮子上方 → 松爪 → 退回。

换任务：--suite libero_goal --task 3（对象名用 --target / --place 指定 obs 里的键）。
产物落在 `benchmarks/LIBERO-plus/outputs/`（该目录不入库）。
"""
import argparse
import os
import sys
import time
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
from libero.libero.envs.env_wrapper import ControlEnv  # noqa: E402

DEFAULT_SUITE = "libero_object"
DEFAULT_TASK = 0
DEFAULT_TARGET = "alphabet_soup_1_pos"   # 抓什么（obs 里的键）
DEFAULT_PLACE = "basket_1_pos"           # 放到哪


def live_render(env):
    """LIBERO 包了一层 robosuite env；屏上窗口挂在 env.env 上。"""
    inner = getattr(env, "env", None)
    if inner is not None and hasattr(inner, "render"):
        inner.render()
    else:
        env.render()


def clip_action(target, eef, grip, kp=12.0, max_delta=0.6):
    delta = np.clip((np.asarray(target) - np.asarray(eef)) * kp, -max_delta, max_delta)
    return np.concatenate([delta, [0.0, 0.0, 0.0, grip]])


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--live", action="store_true", help="开真 3D 窗口（实时）而不是录像")
    ap.add_argument("--suite", default=DEFAULT_SUITE, help=f"benchmark 套件（默认 {DEFAULT_SUITE}）")
    ap.add_argument("--task", type=int, default=DEFAULT_TASK, help="任务 id")
    ap.add_argument("--target", default=DEFAULT_TARGET, help="要抓的物体在 obs 里的键")
    ap.add_argument("--place", default=DEFAULT_PLACE, help="放置目标在 obs 里的键")
    ap.add_argument("--steps", type=int, default=45, help="每个路径点的步数")
    ap.add_argument("--repeat", type=int, default=1, help="重复几遍")
    ap.add_argument("--hold", type=float, default=0.0, help="live 模式下结尾让窗口多开几秒")
    args = ap.parse_args()

    suite = benchmark.get_benchmark_dict()[args.suite]()
    bddl = suite.get_task_bddl_file_path(args.task)
    task = suite.get_task(args.task)
    print("task:", task.name)
    print("instruction:", task.language)

    if args.live:
        # OffScreenRenderEnv 里 has_renderer 被写死成 False，开真窗口要用 ControlEnv
        env = ControlEnv(bddl_file_name=bddl, camera_heights=256, camera_widths=256,
                         has_renderer=True, has_offscreen_renderer=True)
        print("LIVE WINDOW: 应该弹出一个 GLFW 3D 窗口（标题含 MuJoCo / python）")
    else:
        env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=256, camera_widths=256)

    env.seed(0)
    env.reset()
    obs = env.set_init_state(suite.get_task_init_states(args.task)[0])
    for _ in range(5):                                   # 让物体落稳
        obs, _, _, _ = env.step(np.zeros(7))

    obj = obs[args.target].copy()
    place = obs[args.place].copy()
    print(f"pick target {obj}, place target {place}")

    grasp_z = max(obj[2] + 0.02, 0.02)
    waypoints = [
        ([obj[0], obj[1], grasp_z + 0.16], -1.0, "move above object"),
        ([obj[0], obj[1], grasp_z], -1.0, "descend"),
        ([obj[0], obj[1], grasp_z], +1.0, "close gripper"),
        ([obj[0], obj[1], grasp_z + 0.20], +1.0, "lift"),
        ([place[0], place[1], grasp_z + 0.20], +1.0, "move above place"),
        ([place[0], place[1], place[2] + 0.12], +1.0, "descend"),
        ([place[0], place[1], place[2] + 0.12], -1.0, "open gripper -> drop"),
        ([place[0], place[1], place[2] + 0.28], -1.0, "retreat"),
    ]

    writer = None
    frames = 0
    if not args.live:
        os.makedirs("outputs", exist_ok=True)
        writer = cv2.VideoWriter("outputs/demo_pick.mp4",
                                 cv2.VideoWriter_fourcc(*"mp4v"), 20, (512, 256))
        print("recording ->", LIBERO_ROOT / "outputs/demo_pick.mp4")

    for rep in range(args.repeat):
        if args.repeat > 1:
            print(f"=== replay {rep + 1}/{args.repeat} ===")
            env.reset()
            obs = env.set_init_state(suite.get_task_init_states(args.task)[0])
            for _ in range(5):
                obs, _, _, _ = env.step(np.zeros(7))
        for target, grip, label in waypoints:
            print(f"  ... {label}")
            for _ in range(args.steps):
                obs, _, done, _ = env.step(
                    clip_action(target, obs["robot0_eef_pos"], grip))
                if args.live:
                    live_render(env)
                    time.sleep(0.02)
                else:
                    ag = obs["agentview_image"][::-1]
                    wr = obs["robot0_eye_in_hand_image"][::-1]
                    writer.write(np.concatenate([ag, wr], axis=1)[:, :, ::-1].copy())
                    frames += 1
                if done:
                    print("  (episode done flag set)")

    try:
        print("task success:", env.check_success())
    except Exception as e:                               # noqa: BLE001
        print("check_success failed:", e)

    if writer is not None:
        writer.release()
        print(f"done: {frames} frames -> {LIBERO_ROOT}/outputs/demo_pick.mp4")
    if args.live and args.hold > 0:
        print(f"holding window open for {args.hold}s ...")
        t0 = time.time()
        while time.time() - t0 < args.hold:
            live_render(env)
            time.sleep(0.02)
    env.close()


if __name__ == "__main__":
    main()
