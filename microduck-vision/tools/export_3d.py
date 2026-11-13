"""把一次重建的中间结果导出成 HTML 里那个 3D 控件要的 JSON。

    MUJOCO_GL=osmesa PYTHONPATH=/data/vision:/app/microduck_local/src \
      /opt/venv/bin/python tools/export_3d.py --seed 1 --out reports/scene3d.json

导出四段：原始四视角点云 → 去掉地面 → 聚类 → 拟合圆柱 + 六元组。
每一段的点都量化成 int16（以场景中心/半径为基准，≈0.03 mm 分辨率），
颜色压成 1 字节调色板索引 —— 四段加起来 ~80 KB，能直接内嵌进单文件 HTML。
"""

from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path

import numpy as np

from mvision import config, geometry, semantics
from mvision.scene import MossScene, SceneConfig

#: 调色板（HTML 里也抄了一份，两边必须一致）
PALETTE = [
    (0x59, 0xa6, 0xff),   # 0 前视 v0
    (0x6e, 0xe7, 0xb7),   # 1 前视 v1
    (0xff, 0xd1, 0x66),   # 2 前视 v2
    (0xc7, 0x92, 0xea),   # 3 腕部
    (0x8a, 0x94, 0xa6),   # 4 其它簇
    (0xff, 0x6b, 0x6b),   # 5 地面内点
    (0xff, 0xff, 0xff),   # 6 拟合用的点
]


def q16(points: np.ndarray, center: np.ndarray, scale: float) -> str:
    q = np.clip(np.round((points - center) / scale * 32767.0), -32768, 32767)
    return base64.b64encode(q.astype("<i2").tobytes()).decode()


def col8(ids: np.ndarray) -> str:
    return base64.b64encode(np.asarray(ids, np.uint8).tobytes()).decode()


def thin(arr: np.ndarray, ids: np.ndarray, limit: int):
    if len(arr) <= limit:
        return arr, ids
    idx = np.linspace(0, len(arr) - 1, limit).astype(int)
    return arr[idx], ids[idx]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--backend", default="color")
    ap.add_argument("--out", default="reports/scene3d.json")
    ap.add_argument("--limit", type=int, default=2400, help="每段最多多少点")
    args = ap.parse_args()

    scene = MossScene(SceneConfig(seed=args.seed, clutter=2))
    try:
        frames, gt = scene.render_plan(depth=True)
        sem = semantics.build(args.backend, gt=gt)
        masks, hits = semantics.masks_for(sem, frames, "the orange can")
        print(f"seed {args.seed}: {len(frames)} views, 语义命中 {hits}, "
              f"GT pos={np.round(gt.pos, 4).tolist()} upright={gt.upright:.2f}")

        # ---- ① 逐视角点云（不加语义掩码：这就是相机看到的东西）
        per_view = [geometry.points_from_views([f], masks=None) for f in frames]
        raw = np.concatenate(per_view) if per_view else np.zeros((0, 3), np.float32)
        raw_id = np.concatenate([np.full(len(p), i, np.uint8)
                                 for i, p in enumerate(per_view)])

        # ---- ② 去掉地面
        nrm, off, inl = geometry.ransac_plane(raw)
        floor_ok = abs(float(nrm[2])) > 0.85 and inl.sum() > 0.25 * len(raw)
        nf = raw[~inl] if floor_ok else raw
        nf_id = raw_id[~inl] if floor_ok else raw_id
        floor_pts = raw[inl][::max(1, int(inl.sum() / 500))] if floor_ok else np.zeros((0, 3))
        floor_deg = float(np.degrees(np.arccos(min(1.0, abs(float(nrm[2]))))))
        floor_n = int(inl.sum())

        # ---- ③ 语义掩码（color 后端）+ 聚类，和主流程一致
        masked = geometry.points_from_views(frames, masks=masks)
        ds = geometry.voxel_downsample(masked, 0.005)
        clusters = geometry.cluster_points(ds, voxel=0.010, min_points=80)
        cl_pts, cl_col = [], []
        for k, c in enumerate(clusters[:5]):
            cl_pts.append(c)
            cl_col.append(np.full(len(c), min(k, 4), np.uint8))
        clusters_all = (np.concatenate(cl_pts) if cl_pts else np.zeros((0, 3)))
        clusters_col = (np.concatenate(cl_col) if cl_col else np.zeros(0, np.uint8))

        # ---- 共识重建（主流程用的就是它）
        final = geometry.reconstruct_consensus(frames, masks=masks, shape=scene.prop.shape)
        pick = geometry.dominant(final)
        if pick is None:
            print("!! 没有检出目标")
            return 1

        # ---- 场景包围盒（HTML 里用它把坐标归一化）
        allpts = np.concatenate([raw, gt.pos[None], pick.centroid[None]])
        center = (allpts.min(0) + allpts.max(0)) / 2.0
        radius = float(np.linalg.norm(allpts - center, axis=1).max())

        def stage(name, desc, pts, ids, limit=None):
            p, i = thin(pts, ids, limit or args.limit)
            return {"name": name, "desc": desc, "n": int(len(pts)),
                    "pts": q16(p, center, radius), "col": col8(i)}

        stages = [
            stage("① 四视角点云",
                  f"{len(frames)} 张照片各自按深度反投影成点云，按视角着色。"
                  f"此时还没有任何「物体」概念 —— 地面、车身、目标混在一起。",
                  raw, raw_id),
            stage("② 去掉地面",
                  f"RANSAC 在地面平面找到 {floor_n} 个内点（法向与竖直差 {floor_deg:.1f}°），"
                  f"把它们从点云里剔掉，剩下的才是「可能属于物体的东西」。",
                  nf, nf_id),
            stage("③ 语义掩码 + 欧式聚类",
                  f"品红掩码把候选缩到目标色；体素并查集把剩下 {len(masked)} 个点分成 "
                  f"{len(clusters)} 个团（按簇着色）。"
                  f"再按「多个视角都指向同一处」投票，剔除被毁掉的单帧。",
                  clusters_all, clusters_col),
            stage("④ 圆柱代数拟合 → 六元组",
                  "在主轴的法平面里做代数圆拟合（圆心与半径一起解），沿轴取稳健中点。"
                  "输出的就是 obs 槽位里的那 6 个数。",
                  pick.points, np.full(len(pick.points), 6, np.uint8)),
        ]

        data = {
            "center": center.tolist(),
            "radius": radius,
            "palette": ["#%02x%02x%02x" % c for c in PALETTE],
            "gt": {"pos": gt.pos.tolist(), "axis": gt.axis.tolist(),
                   "r": gt.radius, "h": gt.half_height,
                   "upright": gt.upright, "top_z": gt.top_z},
            "fit": {"pos": pick.centroid.tolist(), "axis": pick.axis.tolist(),
                    "r": pick.radius, "h": pick.half_length,
                    "upright": pick.upright, "top_z": pick.top_z,
                    "yaw_double": pick.yaw_double, "n_views": pick.n_views},
            "floor": {"pts": q16(floor_pts, center, radius),
                      "n": int(len(floor_pts))},
            "cams": [{"pos": f.cam_xpos.tolist(),
                      "axes": f.cam_xmat.reshape(3, 3).tolist(),
                      "fovy": f.intr.vfov_deg,
                      "aspect": f.intr.width / f.intr.height,
                      "label": f"{f.view}{i}"}
                     for i, f in enumerate(frames)],
            "stages": stages,
            "metric": {"pos_err_mm": float(np.linalg.norm(pick.centroid - gt.pos) * 1000),
                       "top_err_mm": float(abs(pick.top_z - gt.top_z) * 1000)},
        }
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(data, separators=(",", ":")))
        print(f"写出 {out} ({out.stat().st_size // 1024} KB)  "
              f"位置误差 {data['metric']['pos_err_mm']:.2f} mm  "
              f"顶高误差 {data['metric']['top_err_mm']:.2f} mm")
        for s in stages:
            print(f"  {s['name']}: {s['n']} 点 → 保留 {len(base64.b64decode(s['pts'])) // 6}")
    finally:
        scene.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
