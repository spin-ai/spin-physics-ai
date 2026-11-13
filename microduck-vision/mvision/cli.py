"""命令行入口。

    python -m mvision.cli selftest
    python -m mvision.cli capture --seed 0 --backend color
    python -m mvision.cli eval --episodes 20 --backends none,color,oracle
    python -m mvision.cli closedloop --steps 30
    python -m mvision.cli show --ep /data/captures/episodes/ep0000
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _cmd_selftest(args) -> int:
    """不动环境、不联网的前提下，把最容易错的四件事验一遍。"""
    import numpy as np

    from . import camera, config

    ok = True

    # 1. 内参与视场角自洽
    for hfov in (60.0, 70.0, 87.0):
        intr = camera.Pinhole.from_hfov(320, 240, hfov)
        err = abs(intr.hfov_deg - hfov)
        print(f"[1] hfov={hfov:5.1f} → fx={intr.fx:7.2f} "
              f"hfov_back={intr.hfov_deg:6.2f} vfov={intr.vfov_deg:5.2f}")
        ok &= err < 1e-6

    # 2. 投影/反投影往返
    intr = camera.Pinhole.from_hfov(320, 240, 70.0)
    uv = np.array([[0.0, 0.0], [160.0, 120.0], [300.0, 30.0]])
    depth = np.array([0.7, 1.2, 0.4])
    x = (uv[:, 0] - intr.cx) / intr.fx * depth
    y = -(uv[:, 1] - intr.cy) / intr.fy * depth
    p_cam = np.column_stack([x, y, -depth])
    img = np.full((240, 320), np.inf, np.float32)
    for (u, v), d in zip(uv.astype(int), depth):
        img[v, u] = d
    pc, _ = camera.unproject(img, intr, max_range=5.0)
    print(f"[2] unproject 点数 {len(pc)} / 期望 {len(uv)}")
    ok &= len(pc) == len(uv)

    # 3. 场景可建 + 相机在 + 真值可读
    from .scene import CAMERAS, MossScene, SceneConfig
    scene = MossScene(SceneConfig(seed=0))
    try:
        gt = scene.gt()
        frames, _ = scene.render_plan()
        print(f"[3] gt pos={np.round(gt.pos, 4).tolist()} upright={gt.upright:.3f} "
              f"top_z={gt.top_z:.4f} axis_defined={gt.axis_defined}")
        for f in frames:
            print(f"    view={f.view:5s} {f.intr.width}x{f.intr.height} "
                  f"rgb={f.rgb.shape} depth中位={np.median(f.depth[f.depth < 2.0]):.3f}m"
                  if (f.depth < 2.0).any() else
                  f"    view={f.view:5s} rgb={f.rgb.shape} depth 全背景")
        ok &= len(frames) > 0 and np.isfinite(gt.pos).all()
        print(f"[3] {len(frames)} 视角 / {len(CAMERAS)} 个相机定义")
    finally:
        scene.close()

    # 4. 一条完整感知链
    from .capture import capture_episode
    ep = capture_episode(0, out_root=config.CAPTURE_ROOT / "_selftest",
                         backend="color", save_depth=False, save_rgb=False,
                         do_overlay=False)
    p, g = ep["perception"], ep["gt"]
    if p["pos"] is None:
        print("[4] 没有检出目标 ✗")
        ok = False
    else:
        d = np.linalg.norm(np.array(p["pos"]) - np.array(g["pos"])) * 1000
        print(f"[4] 检出 pos 误差 {d:.1f} mm, n={p['n_points']}, "
              f"axis={p['axis_double']}, 语义命中 {ep['masks_hits']}/{ep['n_views']}")
        ok &= d < 25.0

    print("\nSELFTEST", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def _cmd_capture(args) -> int:
    from . import capture, config
    scene_kw = {}
    if args.clutter is not None:
        scene_kw["clutter"] = args.clutter
    if args.pose is not None:
        scene_kw["target_pose"] = args.pose
    ep = capture.capture_episode(args.seed, out_root=args.out or config.CAPTURE_ROOT,
                                 backend=args.backend, instruction=args.instruction,
                                 save_depth=not args.no_depth,
                                 save_rgb=not args.no_rgb,
                                 scene_kw=scene_kw or None)
    print(json.dumps({"dir": ep["dir"], "gt": ep["gt"],
                      "perception": {k: ep["perception"][k]
                                     for k in ("pos", "axis", "upright", "top_z",
                                               "n_points", "source")},
                      "timings": ep["timings"]}, indent=1))
    return 0


def _cmd_eval(args) -> int:
    from . import config, evaluate
    backends = [b.strip() for b in args.backends.split(",") if b.strip()]
    scene_kw = {}
    if args.clutter is not None:
        scene_kw["clutter"] = args.clutter
    if args.pose is not None:
        scene_kw["target_pose"] = args.pose
    rows = evaluate.sweep(args.episodes, backends=backends, seed0=args.seed0,
                          out_root=args.out or config.CAPTURE_ROOT,
                          scene_kw=scene_kw or None,
                          verbose=not args.quiet)
    md, js = evaluate.write_report(
        rows, args.name,
        config_used={"episodes_per_backend": args.episodes,
                     "seed0": args.seed0,
                     "clutter": "default" if args.clutter is None else args.clutter,
                     "target_pose": args.pose or "random",
                     "backends": ",".join(backends),
                     "render": f"{config.RENDER_W}x{config.RENDER_H}",
                     "roi": list(config.ROI)},
        report_root=args.reports or config.REPORT_ROOT)
    print(f"\n报告: {md}\n指标: {js}")
    return 0


def _cmd_closedloop(args) -> int:
    from . import closedloop
    print("=== seam: 真 MossPickEnv + 感知写 obs ===")
    rep = closedloop.seam_check(steps=args.steps)
    print(json.dumps(rep.to_dict(), indent=1))
    print("\n=== latency: 控制步 vs 插入重建 ===")
    lat = closedloop.latency_check(steps=args.steps)
    print(json.dumps(lat, indent=1))
    ok = rep.ok and lat["verdict"] == "ok"
    print("\nCLOSEDLOOP", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def _cmd_show(args) -> int:
    import numpy as np
    ep_dir = Path(args.ep)
    perc = json.loads((ep_dir / "perception.json").read_text())
    gt = perc["gt"]
    p = perc["perception"]
    print(f"episode {ep_dir.name}  backend={perc['backend']}  "
          f"views={perc['n_views']}  masks={perc['masks_hits']}")
    print(f"  GT   pos={np.round(gt['pos'], 4).tolist()} axis={np.round(gt['axis'], 3).tolist()} "
          f"upright={gt['upright']:.3f} top={gt['top_z']:.4f}")
    if p["pos"] is None:
        print("  EST  未检出")
        return 1
    d = np.linalg.norm(np.array(p["pos"]) - np.array(gt["pos"])) * 1000
    print(f"  EST  pos={np.round(p['pos'], 4).tolist()} axis={np.round(p['axis'], 3).tolist()} "
          f"upright={p['upright']:.3f} top={p['top_z']:.4f} n={p['n_points']}")
    print(f"  误差 位置 {d:.1f} mm  顶高 {abs(p['top_z'] - gt['top_z']) * 1000:.1f} mm")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="mvision", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("selftest", help="自检：投影/场景/一条完整链路")
    s.set_defaults(fn=_cmd_selftest)

    c = sub.add_parser("capture", help="采集一个 episode")
    c.add_argument("--seed", type=int, default=0)
    c.add_argument("--backend", default="color", choices=["none", "color", "oracle", "http"])
    c.add_argument("--instruction", default="the orange can")
    c.add_argument("--out", default=None)
    c.add_argument("--no-depth", action="store_true")
    c.add_argument("--no-rgb", action="store_true")
    c.add_argument("--clutter", type=int, default=None)
    c.add_argument("--pose", default=None, choices=["upright", "lying", "random"])
    c.set_defaults(fn=_cmd_capture)

    e = sub.add_parser("eval", help="批量评测 + 出报告")
    e.add_argument("--episodes", type=int, default=10)
    e.add_argument("--seed0", type=int, default=0)
    e.add_argument("--backends", default="none,color")
    e.add_argument("--name", default="perception")
    e.add_argument("--out", default=None)
    e.add_argument("--reports", default=None)
    e.add_argument("--quiet", action="store_true")
    e.add_argument("--clutter", type=int, default=None,
                   help="干扰物数量（默认 2）；加大可以看无语义后端怎么退化")
    e.add_argument("--pose", default=None, choices=["upright", "lying", "random"],
                   help="目标姿态")
    e.set_defaults(fn=_cmd_eval)

    cl = sub.add_parser("closedloop", help="seam + 时延检查")
    cl.add_argument("--steps", type=int, default=30)
    cl.set_defaults(fn=_cmd_closedloop)

    sh = sub.add_parser("show", help="看一个 episode 的结果")
    sh.add_argument("--ep", required=True)
    sh.set_defaults(fn=_cmd_show)

    args = ap.parse_args(argv)
    return int(args.fn(args) or 0)


if __name__ == "__main__":
    sys.exit(main())
