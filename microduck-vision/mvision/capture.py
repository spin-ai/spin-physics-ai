"""采集一个 episode：多视角渲染 → 落盘 → 语义 → 重建 → 感知 JSON。"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from . import adapters, config, geometry, semantics
from .scene import MossScene, SceneConfig, ViewFrame


def _jsonable(obj):
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return obj


def _write_rgb(path: Path, rgb: np.ndarray) -> None:
    from PIL import Image
    Image.fromarray(rgb.astype(np.uint8)).save(path)


def _write_overlay(frame: ViewFrame, gt_pos: np.ndarray,
                   est_pos: np.ndarray | None, path: Path) -> None:
    """把 GT（白圈）与估计（黄十字）投到同一张照片上 —— 失败样本一眼可读。"""
    from PIL import Image, ImageDraw
    img = Image.fromarray(frame.rgb.astype(np.uint8)).convert("RGB")
    draw = ImageDraw.Draw(img)
    uv_gt, d_gt = geometry_project(frame, gt_pos)
    if d_gt > 0 and 0 <= uv_gt[0] < frame.intr.width and 0 <= uv_gt[1] < frame.intr.height:
        r = 8
        draw.ellipse([uv_gt[0] - r, uv_gt[1] - r, uv_gt[0] + r, uv_gt[1] + r],
                     outline=(255, 255, 255), width=2)
    if est_pos is not None:
        uv, d = geometry_project(frame, est_pos)
        if d > 0 and 0 <= uv[0] < frame.intr.width and 0 <= uv[1] < frame.intr.height:
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                draw.line([uv[0] - 7 * dx, uv[1] - 7 * dy, uv[0] + 7 * dx,
                           uv[1] + 7 * dy], fill=(255, 220, 0), width=2)
    img.save(path)


def geometry_project(frame: ViewFrame, point: np.ndarray):
    from .camera import project
    uv, depth = project(np.asarray(point, float)[None, :], frame.cam_xpos,
                        frame.cam_xmat, frame.intr)
    return uv[0], float(depth[0])


def capture_episode(seed: int, out_root: Path | str = config.CAPTURE_ROOT,
                    backend: str = "none", instruction: str = "the orange can",
                    save_depth: bool = True, save_rgb: bool = True,
                    scene_kw: dict | None = None,
                    do_overlay: bool = True) -> dict:
    """跑一次"拍几张照片 → 重建"，把产物和指标都写进 episode 目录。"""
    t_all = time.perf_counter()
    out_dir = Path(out_root) / f"ep{seed:04d}"
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = SceneConfig(seed=seed, **(scene_kw or {}))
    with MossScene(cfg) as scene:
        t0 = time.perf_counter()
        frames, gt = scene.render_plan(depth=True)   # 几何靠深度，必须渲
        t_render = time.perf_counter() - t0

        sem = semantics.build(backend, gt=gt)
        t0 = time.perf_counter()
        masks, hits = semantics.masks_for(sem, frames, instruction)
        t_sem = time.perf_counter() - t0

        t0 = time.perf_counter()
        if len(frames) > 1:
            clusters = geometry.reconstruct_consensus(
                frames, masks=masks, shape=scene.prop.shape)
        else:
            clusters = geometry.reconstruct(frames, masks=masks,
                                           shape=scene.prop.shape)
        t_geom = time.perf_counter() - t0

        pick = geometry.dominant(clusters)
        now = float(frames[-1].t) if frames else 0.0
        if pick is None:
            perception = adapters.Perception(t=now, conf=0.0,
                                             source=f"geometry+{backend}")
        else:
            perception = adapters.from_cluster(
                pick, t=now, label=scene.cfg.prop,
                conf=float(min(1.0, pick.n / 400.0)),
                source=f"geometry+{backend}")

    ep = {
        "seed": seed,
        "dir": str(out_dir),
        "instruction": instruction,
        "backend": backend,
        "masks_hits": hits,
        "n_views": len(frames),
        "gt": gt.to_dict(),
        "perception": perception.to_dict(),
        "clusters": [c.to_dict(label="candidate", conf=0.0) for c in clusters[:6]],
        "timings": {"render": t_render, "semantics": t_sem, "geometry": t_geom,
                    "total": time.perf_counter() - t_all},
        "scene": {"prop": cfg.prop, "clutter": cfg.clutter,
                  "target_pose": getattr(scene, "_target_pose_name", "upright")},
    }

    if save_rgb or save_depth:
        for i, f in enumerate(frames):
            vdir = out_dir / f"view_{i:02d}_{f.view}"
            vdir.mkdir(exist_ok=True)
            if save_rgb:
                _write_rgb(vdir / "rgb.png", f.rgb)
            if save_depth:
                np.save(vdir / "depth.npy", f.depth)
            (vdir / "meta.json").write_text(
                json.dumps(_jsonable(f.to_meta()), indent=1))
            if masks is not None and masks[i] is not None:
                np.save(vdir / "mask.npy", masks[i])
        if do_overlay and frames:
            _write_overlay(frames[0], gt.pos, perception.pos,
                    out_dir / "overlay_view0.png")

    (out_dir / "scene.json").write_text(json.dumps({
        "seed": seed, "instruction": instruction, "backend": backend,
        "gt": gt.to_dict(),
        "views": [f.to_meta() for f in frames],
        "scene": ep["scene"],
    }, indent=1, default=str))
    (out_dir / "perception.json").write_text(json.dumps(_jsonable(ep), indent=1))
    return ep
