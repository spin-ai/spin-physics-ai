"""拿网上的真实人照片跑一遍：单目深度 → 点云 → 量身高 → 双视角位姿。

    TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 \
    /opt/venv/bin/python tools/export_real.py --out reports/real

照片来源（Wikimedia Commons，全部可在报告里注明许可）：
  Portrait_of_a_toddler_standing_next_to_a.jpg    CC BY 4.0  (Auckland Museum)
  toddler2.jpg  （同一场景的第二张）               CC BY 4.0
  Donald_Trump_official_portrait.jpg.jpg           Public domain
  Portrait_Michelle_Bachelet.jpg.jpg               CC BY 3.0 cl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from mvision import real

#: 每张照片的"锚"：只提供一个假定的真实尺寸，用来把相对深度换成米。
#: 这是全流程唯一的外部先验，也是它不可用于抓取的根本原因。
#: 手工多边形（归一化），描的是人的轮廓 —— 只用于全身照（要量身高）。
#: 半身像不量化，所以不描。
POLYS = {
    "toddler1": [(0.400,0.030),(0.530,0.028),(0.565,0.060),(0.585,0.110),(0.610,0.230),
                 (0.620,0.380),(0.615,0.470),(0.600,0.560),(0.600,0.660),(0.585,0.780),
                 (0.580,0.890),(0.600,0.955),(0.380,0.958),(0.395,0.890),(0.400,0.780),
                 (0.385,0.660),(0.380,0.560),(0.365,0.470),(0.360,0.380),(0.370,0.230),
                 (0.385,0.110),(0.390,0.060)],
    "toddler2": [(0.470,0.125),(0.600,0.120),(0.635,0.150),(0.650,0.200),(0.665,0.300),
                 (0.675,0.420),(0.670,0.520),(0.655,0.620),(0.660,0.720),(0.645,0.820),
                 (0.650,0.870),(0.445,0.875),(0.455,0.820),(0.465,0.720),(0.450,0.620),
                 (0.435,0.520),(0.430,0.420),(0.445,0.300),(0.455,0.200),(0.460,0.150)],
}

ANCHORS = {
    "toddler1": {"what": "孩子身高 1.05 m（假定）", "height_m": 1.05},
    "toddler2": {"what": "孩子身高 1.05 m（假定）", "height_m": 1.05},
    "portrait": {"what": "成年男性身高 1.80 m（假定）", "height_m": 1.80},
    "portrait2": {"what": "成年女性身高 1.65 m（假定）", "height_m": 1.65},
}

PHOTOS = [
    ("toddler1", "/data/photos/Portrait_of_a_toddler_standing_next_to_a.jpg",
     "Wikimedia Commons · CC BY 4.0 · Auckland Museum"),
    ("toddler2", "/data/photos/toddler2.jpg",
     "Wikimedia Commons · CC BY 4.0 · Auckland Museum"),
    ("portrait", "/data/photos/Donald_Trump_official_portrait.jpg.jpg",
     "Wikimedia Commons · Public domain"),
    ("portrait2", "/data/photos/Portrait_Michelle_Bachelet.jpg.jpg",
     "Wikimedia Commons · CC BY 3.0 cl"),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="reports/real")
    ap.add_argument("--model", default="/data/models/dav2b")   # Base 明显比 Small 干净（1.0 s/张）
    ap.add_argument("--only", default="")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    md = real.MonocularDepth(args.model)
    want = set(args.only.split(",")) if args.only else None

    summary = {}
    recon = {}
    for name, path, credit in PHOTOS:
        if want and name not in want:
            continue
        if not Path(path).exists():
            print(f"!! 缺图 {path}")
            continue
        rgb = real.load_photo(path)
        disp = md.disparity(rgb)
        rec = real.reconstruct(name, rgb, disp)
        info = (real.segment_polygon(rec, POLYS[name]) if name in POLYS
                else real.segment_subject(rec))
        anchor = ANCHORS.get(name, {"height_m": 1.7, "what": "假定身高 1.70 m"})
        # 尺度 k：先把点乘 k，再让主体高度等于锚
        base = 1.0 / float(np.median(-rec.points[:, 2]))     # 单位化
        m0 = rec.measure_pinhole(base)
        if not m0.get("ok"):
            print(f"!! {name}: 没切出主体")
            continue
        if name not in POLYS:                     # 半身像：不量身高
            (out / f"{name}.json").write_text(json.dumps(
                {**real.to_widget(rec, max_points=9000), "name": name, "credit": credit,
                 "seg": info, "measure": None, "anchor": None}, separators=(",", ":")))
            real.dump_png(rgb, out / f"{name}_photo.png")
            real.dump_subject_overlay(rec, out / f"{name}_subject.png")
            real.dump_depth_png(disp, out / f"{name}_depth.png")
            summary[name] = {"credit": credit, "seg": info, "measure": None,
                             "anchor": None, "note": "半身像，不量身高"}
            recon[name] = rec
            print(f"{name:10s} {rgb.shape[1]}x{rgb.shape[0]} 点云 {len(rec.points):6d} 点（半身像，不量身高）")
            continue
        k = base * anchor["height_m"] / max(m0["height_m"], 1e-6)
        m = rec.measure_pinhole(k)
        wj = real.to_widget(rec, max_points=16000 if name == "toddler1" else (9000 if name == "portrait" else 1500))
        (out / f"{name}.json").write_text(json.dumps(
            {**wj, "name": name, "credit": credit, "seg": info,
             "measure": m, "anchor": anchor}, separators=(",", ":")))
        np.save(out / f"{name}_disp.npy", disp)
        real.dump_png(rgb, out / f"{name}_photo.png")
        real.dump_subject_overlay(rec, out / f"{name}_subject.png")
        real.dump_depth_png(disp, out / f"{name}_depth.png")
        summary[name] = {"credit": credit, "seg": info, "measure": m,
                         "anchor": anchor, "k": k}
        recon[name] = rec
        print(f"{name:10s} {rgb.shape[1]}x{rgb.shape[0]} 主体 {info.get('subject_points', 0):6d} 点  "
              f"高 {m['height_m']*100:5.1f} cm  宽 {m['width_m']*100:5.1f} cm  "
              f"主体深度 {m['subject_depth_m']:.2f} m  ROI 深度 {m['roi_depth_range_m'][0]:.2f}–{m['roi_depth_range_m'][1]:.2f} m")

    # ---- 双视角：同场景的两张
    if "toddler1" in recon and "toddler2" in recon:
        tv = real.two_view(recon["toddler1"].rgb, recon["toddler2"].rgb)
        summary["two_view"] = tv
        print("\n双视角(toddler1 ↔ toddler2):", json.dumps(tv, ensure_ascii=False))

    (out / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False))
    print(f"\n写出 {out}/summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
