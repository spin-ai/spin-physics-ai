"""给 HTML 报告生成"看得懂"的图：四张照片、深度图、俯视布局图。

    MUJOCO_GL=osmesa PYTHONPATH=/data/vision:/app/microduck_local/src \
      /opt/venv/bin/python tools/export_figures.py --seed 1 --out reports/img --width 480

产出的东西都是"原始数据换个画法"，没有修饰：
  photo_i.png   第 i 个视角的原图（前视做了亮度提升以便阅读，文件名标注了倍数）
  depth_i.png   同一视角的深度缓冲（自己实现的 turbo 色表，0.05–0.85 m）
  topview.svg   俯视布局：机器人、目标、四个相机的位置与朝向（矢量，可打印）
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

W = "480"
for i, a in enumerate(os.sys.argv):          # 在 import mvision 之前定分辨率
    if a == "--width" and i + 1 < len(os.sys.argv):
        os.environ["MV_WIDTH"] = os.sys.argv[i + 1]

import numpy as np                            # noqa: E402
from PIL import Image, ImageEnhance           # noqa: E402

from mvision.scene import MossScene, SceneConfig   # noqa: E402

# ---------------------------------------------------------------- 色表

_STOPS = [
    (0.00, (0.19, 0.07, 0.23)), (0.18, (0.24, 0.28, 0.63)),
    (0.36, (0.13, 0.62, 0.75)), (0.54, (0.28, 0.80, 0.51)),
    (0.70, (0.80, 0.83, 0.20)), (0.85, (0.97, 0.53, 0.15)),
    (1.00, (0.97, 0.24, 0.13)),
]


def turbo(t: np.ndarray) -> np.ndarray:
    """给深度图上色（避免引入 matplotlib）。"""
    t = np.clip(t, 0.0, 1.0)
    out = np.zeros(t.shape + (3,), np.float32)
    xs = np.array([s[0] for s in _STOPS])
    cs = np.array([s[1] for s in _STOPS], np.float32)
    for k in range(3):
        out[..., k] = np.interp(t, xs, cs[:, k])
    return (out * 255).astype(np.uint8)


def depth_png(depth: np.ndarray, path: Path, near=0.05, far=0.85) -> None:
    d = np.array(depth, np.float32)
    valid = np.isfinite(d) & (d > near) & (d < far)
    img = turbo((d - near) / (far - near))
    img[~valid] = (10, 13, 19)
    Image.fromarray(img).save(path)


# ---------------------------------------------------------------- 俯视图


def topview_svg(frames, gt, path: Path) -> None:
    """俯视布局：4 个拍摄点各一条视线连到目标。刻意不画视场扇形 —— 87 度一画
    就糊成一团，而读者真正要看的是「4 个位置、同一个罐子」。"""
    sc = 700.0
    ox, oy = 300.0, 235.0          # 参考系原点（底盘 home 位姿）
    Wc, Hc = 760, 470

    def xy(x, y):
        return ox + x * sc, oy - y * sc

    def rot(v, a):
        return (v[0] * math.cos(a) - v[1] * math.sin(a),
                v[0] * math.sin(a) + v[1] * math.cos(a))

    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {Wc} {Hc}" '
             f'width="100%" role="img" aria-label="俯视布局">',
             '<rect width="100%" height="100%" fill="#0d1219"/>']

    for k in range(-7, 13):                        # 5 cm 网格
        x = k * 0.05
        a, b = xy(x, -0.30), xy(x, 0.30)
        parts.append(f'<line x1="{a[0]:.1f}" y1="{a[1]:.1f}" x2="{b[0]:.1f}" '
                     f'y2="{b[1]:.1f}" stroke="#1a2230" stroke-width="1"/>')
    for k in range(-6, 7):
        y = k * 0.05
        a, b = xy(-0.34, y), xy(0.55, y)
        parts.append(f'<line x1="{a[0]:.1f}" y1="{a[1]:.1f}" x2="{b[0]:.1f}" '
                     f'y2="{b[1]:.1f}" stroke="#1a2230" stroke-width="1"/>')

    # 底盘外形（home 位姿，虚线）
    corners = [xy(*rot(c, 0.0)) for c in
               ((-0.11, -0.212), (0.186, -0.212), (0.186, 0.212), (-0.11, 0.212))]
    d = "M" + " L".join(f"{c[0]:.1f} {c[1]:.1f}" for c in corners) + " Z"
    parts.append(f'<path d="{d}" fill="#182136" fill-opacity="0.55" stroke="#3f5580" '
                 f'stroke-width="1.2" stroke-dasharray="5 4"/>')
    lab = xy(0.04, 0.0)
    parts.append(f'<text x="{lab[0]:.0f}" y="{lab[1]+4:.0f}" fill="#6f86b0" font-size="11" '
                 f'text-anchor="middle" font-family="monospace">MOSS</text>')

    # 目标
    t = xy(gt.pos[0], gt.pos[1])
    parts.append(f'<circle cx="{t[0]:.1f}" cy="{t[1]:.1f}" r="{gt.radius*sc:.1f}" '
                 f'fill="#cc1ab8" stroke="#fff" stroke-width="1.6"/>')
    parts.append(f'<text x="{t[0]:.0f}" y="{t[1]-gt.radius*sc-10:.0f}" fill="#e7e9ee" '
                 f'font-size="12.5" text-anchor="middle">目标罐（r=33 mm）</text>')

    colors = ["#59a6ff", "#6ee7b7", "#ffd166", "#c792ea"]
    labels = ["前视 v0", "前视 v1", "前视 v2", "腕部 v3"]
    anchors = ["end", "end", "end", "start"]
    dy = [4, -6, 4, 4]

    for i, f in enumerate(frames):
        T = f.T_base_cam
        pos = T[:3, 3]
        axis = -T[:3, 2]
        cp = xy(pos[0], pos[1])

        # 视线：拍摄点 → 目标
        parts.append(f'<line x1="{cp[0]:.1f}" y1="{cp[1]:.1f}" x2="{t[0]:.1f}" '
                     f'y2="{t[1]:.1f}" stroke="{colors[i]}" stroke-width="1" '
                     f'stroke-opacity="0.35" stroke-dasharray="2 5"/>')
        # 朝向小箭头（相机光轴，10 cm）
        e = xy(pos[0] + axis[0] * 0.10, pos[1] + axis[1] * 0.10)
        parts.append(f'<line x1="{cp[0]:.1f}" y1="{cp[1]:.1f}" x2="{e[0]:.1f}" '
                     f'y2="{e[1]:.1f}" stroke="{colors[i]}" stroke-width="2"/>')
        parts.append(f'<circle cx="{cp[0]:.1f}" cy="{cp[1]:.1f}" r="5.5" '
                     f'fill="{colors[i]}" stroke="#0d1219" stroke-width="1.5"/>')
        # 标签：前视在左、腕部在右，避让目标
        off = -12 if anchors[i] == "end" else 12
        parts.append(f'<text x="{cp[0]+off:.0f}" y="{cp[1]+dy[i]:.0f}" fill="{colors[i]}" '
                     f'font-size="11.5" text-anchor="{anchors[i]}">{labels[i]}</text>')

    # 比例尺
    x0, y0 = xy(-0.32, 0.25)
    x1, _ = xy(-0.27, 0.25)
    parts.append(f'<line x1="{x0:.0f}" y1="{y0:.0f}" x2="{x1:.0f}" y2="{y0:.0f}" '
                 f'stroke="#8894a8" stroke-width="2"/>')
    parts.append(f'<text x="{(x0+x1)/2:.0f}" y="{y0-7:.0f}" fill="#8894a8" '
                 f'font-size="11" text-anchor="middle">5 cm</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts))


# ---------------------------------------------------------------- main


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default="reports/img")
    ap.add_argument("--width", type=int, default=480)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    os.environ["MV_WIDTH"] = str(args.width)
    scene = MossScene(SceneConfig(seed=args.seed, clutter=2))
    try:
        frames, gt = scene.render_plan(depth=True)
        names = []
        for i, f in enumerate(frames):
            gain = 2.6 if f.view == "front" else 1.2
            img = Image.fromarray(f.rgb).convert("RGB")
            img = ImageEnhance.Brightness(img).enhance(gain)
            p = out / f"photo_{i}.png"
            img.save(p)
            names.append((p.name, f.view, f.intr.width, f.intr.height,
                          round(float(np.median(f.depth[(f.depth > 0.05) & (f.depth < 2.0)])), 3),
                          gain))
            depth_png(f.depth, out / f"depth_{i}.png")
        topview_svg(frames, gt, out / "topview.svg")
        meta = {"seed": args.seed, "gt": gt.to_dict(), "photos": names,
                "cams": [{"view": f.view, "pos": f.T_base_cam[:3, 3].tolist(),
                          "intr": f.intr.to_dict()} for f in frames]}
        (out / "meta.json").write_text(json.dumps(meta, indent=1))
        print(f"写出 {len(frames)} 组照片/深度 + topview.svg → {out}")
        for n in names:
            print(f"  {n[0]:14s} {n[1]:5s} {n[2]}x{n[3]}  中位深度 {n[4]} m  亮度 ×{n[5]}")
    finally:
        scene.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
