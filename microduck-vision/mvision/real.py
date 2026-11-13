"""真实照片适配器：网上真实的人照片 → 单目深度 → 点云 → 量身高。

和仿真管线的差别，全部集中在三个"仿真免费、真实要付钱"的地方：

  1. **没有深度**。这里用 Depth Anything V2 Small（95 MB，CPU 0.6 s/张）。
     它的输出是**视差**（越大越近），只能给出 depth = k / disparity，
     k 未知 ⇒ 重建是"保序但不可度量"的。
  2. **没有相机位姿**。单张不需要；两张必须自己解（这里用 ORB + 本质矩阵）。
  3. **人不是圆柱**。六元组里的"半径/半长"没有意义，改用轴对齐包围盒 +
     身高 / 肩宽 / 厚度，并把"顶端高度"解释成"头顶高度"。

于是本模块只回答一件事：**这条路能走多远、断在哪儿**。答案写在输出的
`scale_unknown` 与 `pose_ok` 两个字段里，不看漂亮图。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import geometry

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)

#: 假设的相机（手机主摄等效）：水平视场 65°，方形像素。
#: 真实照片不给内参，这是第二个"必须假设"的量。
ASSUMED_HFOV_DEG = 65.0


# --------------------------------------------------------------------------- #
# 深度模型
# --------------------------------------------------------------------------- #


class MonocularDepth:
    """Depth Anything V2 Small，torch CPU 推理。"""

    def __init__(self, model_dir: str = "/data/models/dav2s", long_side: int = 518):
        import torch
        from transformers import AutoModelForDepthEstimation
        self._torch = torch
        self.long_side = long_side
        self.net = AutoModelForDepthEstimation.from_pretrained(model_dir).eval()

    def disparity(self, rgb: np.ndarray) -> np.ndarray:
        """返回视差图（越大越近），尺寸与输入一致。"""
        import torch.nn.functional as F
        from PIL import Image
        img = Image.fromarray(rgb.astype(np.uint8))
        w, h = img.size
        s = self.long_side / max(w, h)
        nw = max(14, int(round(w * s / 14)) * 14)
        nh = max(14, int(round(h * s / 14)) * 14)
        a = np.asarray(img.resize((nw, nh), Image.BICUBIC), np.float32) / 255.0
        x = self._torch.from_numpy((a - MEAN) / STD).permute(2, 0, 1).unsqueeze(0)
        with self._torch.no_grad():
            out = self.net(pixel_values=x)
        p = F.interpolate(out.predicted_depth.unsqueeze(1), size=(h, w),
                          mode="bicubic", align_corners=False)[0, 0]
        return p.numpy().astype(np.float32)


# --------------------------------------------------------------------------- #
# 数据结构
# --------------------------------------------------------------------------- #


@dataclass
class RealRecon:
    name: str
    rgb: np.ndarray
    disp: np.ndarray
    points: np.ndarray                  # (N,3) 相机系，单位 k
    colors: np.ndarray                  # (N,3) uint8
    intr: dict
    subject: np.ndarray | None = None   # 主体点在 points 里的下标
    pix: np.ndarray | None = None       # 每个点对应的扁平像素下标（u = pix % w）
    stats: dict = field(default_factory=dict)

    def measure_pinhole(self, k: float) -> dict:
        """用针孔关系量尺寸，而不是对点云取包围盒。

        为什么不取包围盒：ROI 里除了人还有地面（很近）和背景（很远），
        3D bbox 的深度方向会被它们撑到 1.9 m。这里改用：
          · 主体深度 = ROI 内视差的**众数带**（人占的面积最大），z = k / disp
          · 身高 = 像素纵向跨度 × z / fy
          · 宽度 = 像素横向跨度 × z / fx   ← 注意这不是"肩宽"，是主体外接宽度
        """
        if self.subject is None or len(self.subject) < 50:
            return {"ok": False}
        h, w = self.disp.shape[:2]
        u = (self.pix[self.subject] % w).astype(int)
        v = (self.pix[self.subject] // w).astype(int)
        d = self.disp[v, u]
        hist, edges = np.histogram(d, bins=48)
        k_mode = int(np.argmax(hist))
        band = (d >= edges[max(0, k_mode - 3)]) & (d <= edges[min(len(edges) - 1, k_mode + 3)])
        if band.sum() < 30:
            band = np.ones(len(d), bool)
        z = float(k / max(np.median(d[band]), 1e-6))
        h_px = float(v.max() - v.min() + 1)
        w_px = float(u.max() - u.min() + 1)
        return {"ok": True,
                "height_m": h_px * z / self.intr["fy"],
                "width_m": w_px * z / self.intr["fx"],
                "subject_depth_m": z,
                "pixel_h": h_px, "pixel_w": w_px,
                "roi_depth_range_m": [float(k / max(np.percentile(d, 99), 1e-6)),
                                      float(k / max(np.percentile(d, 1), 1e-6))],
                "n_points": int(len(u))}

    def measure(self, k: float) -> dict:
        """用尺度 k（disparity → 深度）换算成米，再量三个数。"""
        if self.subject is None or len(self.subject) < 50:
            return {"ok": False}
        p = self.points[self.subject] * k
        # 用 1%/99% 分位当包围盒：轮廓边缘的深度跳变会产生孤立点，
        # min/max 会被它们支配（实测把肩宽放大到 168 cm）
        lo, hi = np.percentile(p, 1.0, axis=0), np.percentile(p, 99.0, axis=0)
        # 相机系：x 右、y 上、z 向后（与 MuJoCo 一致），所以竖直方向是 index 1
        return {"ok": True,
                "height_m": float(hi[1] - lo[1]),
                "width_m": float(hi[0] - lo[0]),
                "thickness_m": float(hi[2] - lo[2]),
                "centroid_m": ((lo + hi) / 2).tolist(),
                "head_y_m": float(hi[1]), "foot_y_m": float(lo[1]),
                "depth_range_m": [float(-hi[2]), float(-lo[2])],
                "n_points": int(len(p))}


# --------------------------------------------------------------------------- #
# 重建
# --------------------------------------------------------------------------- #


def _intr(w: int, h: int, hfov_deg: float = ASSUMED_HFOV_DEG) -> dict:
    fx = 0.5 * w / math.tan(math.radians(hfov_deg) / 2)
    return {"width": w, "height": h, "fx": float(fx), "fy": float(fx),
            "cx": 0.5 * w, "cy": 0.5 * h, "hfov_deg": hfov_deg}


def load_photo(path: str | Path, max_side: int = 720) -> np.ndarray:
    from PIL import Image
    img = Image.open(path).convert("RGB")
    w, h = img.size
    if max(w, h) > max_side:
        s = max_side / max(w, h)
        img = img.resize((int(w * s), int(h * s)), Image.LANCZOS)
    return np.asarray(img)


def smooth_disparity(disp: np.ndarray, k: int = 5, sigma_r: float = 0.035) -> np.ndarray:
    """双边滤波去单目深度的细碎条纹，但保住轮廓。

    DA-V2 在弱纹理处会抖出 1–2 px 的条纹，反投影后就是一层雾；但中值滤波会把
    腿缝、胳膊这些 1–2 px 的结构抹平（实测腿就并成一片）。双边在"值域"上设阈值，
    跨越轮廓的邻居不参与平均，所以边缘留得住。
    """
    import cv2
    d = disp.astype(np.float32)
    lo, hi = float(np.percentile(d, 1)), float(np.percentile(d, 99))
    span = max(hi - lo, 1e-6)
    dn = np.clip((d - lo) / span, 0.0, 1.0)          # 归一到 0..1 再滤，
    out = cv2.bilateralFilter(dn, k, sigma_r, 2.0)   # σ_color 才有明确含义
    return out * span + lo


def reconstruct(name: str, rgb: np.ndarray, disp: np.ndarray,
                disp_floor: float = 0.05, max_points: int = 20000) -> RealRecon:
    h, w = disp.shape[:2]
    intr = _intr(w, h)
    disp = smooth_disparity(disp)
    d = np.maximum(disp, disp_floor)
    z = 1.0 / d                                  # 相对深度：近 → 小
    vs, us = np.mgrid[0:h, 0:w]
    X = (us + 0.5 - intr["cx"]) / intr["fx"] * z
    Y = -(vs + 0.5 - intr["cy"]) / intr["fy"] * z
    P = np.stack([X, Y, -z], axis=-1).reshape(-1, 3)   # 相机系：-z 朝前（与 MuJoCo 一致）
    C = rgb.reshape(-1, 3)
    # 降采样：按像素步长
    step = max(1, int(round(math.sqrt(len(P) / max_points))))
    sel = np.arange(0, len(P), step)          # 扁平像素下标，row-major
    rec = RealRecon(name=name, rgb=rgb, disp=disp, points=P[sel].astype(np.float32),
                    colors=C[sel].astype(np.uint8), intr=intr)
    rec.pix = sel
    return rec


def segment_roi(rec: RealRecon, roi: tuple[float, float, float, float]) -> dict:
    """人工框一个 ROI 当主体掩码。

    为什么用手工框而不是自动分割：这几个场景里"人"和"背景"在单目视差图上
    是连通的（树丛的视差正好落在人身上），Otsu + 连通域会把整幅画面并进主体
    —— 实测把肩宽量成 168 cm。人像分割是另一个问题（需要专门的模型），
    本实验只想知道"给定主体，几何量得准不准"，所以这里明确用 ROI。
    """
    h, w = rec.disp.shape[:2]
    x0, y0, x1, y1 = roi
    mask = np.zeros((h, w), bool)
    mask[int(y0 * h):int(y1 * h), int(x0 * w):int(x1 * w)] = True
    u = (rec.pix % w).astype(int)
    v = (rec.pix // w).astype(int)
    sel = mask[v, u]
    rec.subject = np.nonzero(sel)[0]
    rec.mask = mask
    return {"ok": bool(sel.any()), "kind": "manual_roi", "roi": list(roi),
            "subject_points": int(sel.sum())}


def segment_polygon(rec: RealRecon, poly: list[tuple[float, float]]) -> dict:
    """用一条**手工多边形**当主体掩码（归一化坐标）。

    为什么手画轮廓而不是自动分割：这几个场景里"人 / 树丛 / 地面"在单目深度上
    连成一片（实测按视差阈值切，不是把整片背景并进来，就是只剩头和一条腿）。
    自动分割是另一个问题（要人像模型），本实验只关心"给定主体，几何准不准"。
    """
    import cv2
    h, w = rec.disp.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    pts = np.array([[int(x * w), int(y * h)] for x, y in poly], np.int32)
    cv2.fillPoly(mask, [pts], 1)
    u = (rec.pix % w).astype(int)
    v = (rec.pix // w).astype(int)
    sel = mask[v, u].astype(bool)
    rec.subject = np.nonzero(sel)[0]
    rec.mask = mask.astype(bool)
    return {"ok": bool(sel.any()), "kind": "manual_polygon", "n_verts": len(poly),
            "subject_points": int(sel.sum())}


def segment_subject(rec: RealRecon, min_area_frac: float = 0.02) -> dict:
    """在**图像平面**上把主体切出来 —— 这是真实照片和仿真最大的操作差别之一。

    仿真里点云先按语义掩码过滤再聚类就够了；真实照片的"背景"（树丛、墙、天空）
    常常和主体连成一片体素，3D 聚类会把整块背景并进主体（实测把肩宽量成 168 cm）。
    改成：视差图上做 Otsu 一分为二（近 / 远），连通域里挑"又高又靠中间"的那个。
    """
    import cv2
    h, w = rec.disp.shape[:2]
    d8 = np.clip((rec.disp - np.percentile(rec.disp, 1))
                 / (np.percentile(rec.disp, 99) - np.percentile(rec.disp, 1) + 1e-9)
                 * 255, 0, 255).astype(np.uint8)
    _t, near = cv2.threshold(d8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    near = cv2.morphologyEx(near, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    near = cv2.morphologyEx(near, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    n_lab, labels, stats, cents = cv2.connectedComponentsWithStats(near, 8)
    best, best_score = None, 0.0
    info_comps = []
    for k in range(1, n_lab):
        x, y, bw, bh, area = stats[k]
        if area < min_area_frac * h * w:
            continue
        aspect = bh / max(bw, 1)
        central = 1.0 - abs((cents[k][0] / w) - 0.5) * 1.2      # 越靠中线越高
        score = area * (0.6 + 0.6 * min(aspect, 2.5)) * max(central, 0.15)
        info_comps.append({"area": int(area), "aspect": round(float(aspect), 2),
                           "cx": round(float(cents[k][0] / w), 2)})
        if score > best_score:
            best, best_score = k, score
    if best is None:
        rec.subject = None
        return {"ok": False, "components": info_comps}

    mask = (labels == best)
    # 点 → 像素：reconstruct 存的是**扁平**下标（row-major），不是二维网格
    u = (rec.pix % w).astype(int)
    v = (rec.pix // w).astype(int)
    sel = mask[v, u]
    rec.subject = np.nonzero(sel)[0]
    rec.mask = mask
    return {"ok": True, "subject_points": int(sel.sum()),
            "components": info_comps[:6],
            "component_used": info_comps[best - 1] if best - 1 < len(info_comps) else None}


# --------------------------------------------------------------------------- #
# 双视角：ORB + 本质矩阵
# --------------------------------------------------------------------------- #


def two_view(a: np.ndarray, b: np.ndarray) -> dict:
    """真实照片要位姿只能自己解。返回内点数、基线方向、以及"能不能用"。"""
    import cv2
    ga = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY)
    gb = cv2.cvtColor(b, cv2.COLOR_RGB2GRAY)
    orb = cv2.ORB_create(nfeatures=4000)
    ka, da = orb.detectAndCompute(ga, None)
    kb, db = orb.detectAndCompute(gb, None)
    if da is None or db is None:
        return {"ok": False, "reason": "no features"}
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = sorted(bf.match(da, db), key=lambda m: m.distance)
    if len(matches) < 20:
        return {"ok": False, "reason": f"only {len(matches)} matches"}
    pts_a = np.float32([ka[m.queryIdx].pt for m in matches])
    pts_b = np.float32([kb[m.trainIdx].pt for m in matches])
    F, inl = cv2.findFundamentalMat(pts_a, pts_b, cv2.FM_RANSAC, 3.0, 0.999)
    if F is None:
        return {"ok": False, "reason": "no F"}
    inl = inl.ravel().astype(bool)
    n_in = int(inl.sum())
    inta = _intr(*ga.shape[::-1])
    K = np.array([[inta["fx"], 0, inta["cx"]], [0, inta["fy"], inta["cy"]], [0, 0, 1]])
    E = K.T @ F @ K
    ok, R, t, mask_pose = cv2.recoverPose(E, pts_a[inl], pts_b[inl], K)
    ang = math.degrees(math.acos(max(-1.0, min(1.0, (np.trace(R) - 1) / 2))))
    t = t.ravel()
    usable = n_in >= 40 and 0.5 < ang < 90
    return {"ok": True, "usable": bool(usable), "matches": int(len(matches)),
            "inliers": n_in, "inlier_ratio": float(n_in / max(len(matches), 1)),
            "rotation_deg": float(ang),
            "translation_dir": (t / (np.linalg.norm(t) + 1e-9)).tolist(),
            "note": ("内点够、转角合理，可以喂三角化" if usable
                     else "内点太少或转角退化：真实照片之间没有共享的观测几何")}


def surface_shade(rec: RealRecon, light: tuple[float, float, float] = (-0.45, 0.55, -0.70),
                  floor: float = 0.05) -> np.ndarray:
    """按深度图的梯度算法向，再用一个固定方向光算 Lambert 明暗。

    为什么要它：单目点云是"从相机看过去的一张薄片"，只按深度上色仍然像贴图；
    有了法向明暗，额头、鼻梁、衣褶会自己亮起来，看起来才像一个表面。
    """
    h, w = rec.disp.shape[:2]
    d = np.maximum(rec.disp, floor)
    z = 1.0 / d
    fx, fy = rec.intr["fx"], rec.intr["fy"]
    cx, cy = rec.intr["cx"], rec.intr["cy"]
    us, vs = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    X = (us + 0.5 - cx) / fx * z
    Y = -(vs + 0.5 - cy) / fy * z
    Z = -z
    # 像素邻域差分 → 切向量 → 法向（相机系：x 右、y 上、z 向后）
    du = np.stack([np.gradient(X, axis=1), np.gradient(Y, axis=1), np.gradient(Z, axis=1)], -1)
    dv = np.stack([np.gradient(X, axis=0), np.gradient(Y, axis=0), np.gradient(Z, axis=0)], -1)
    n = np.cross(dv, du)
    norm = np.linalg.norm(n, axis=-1, keepdims=True) + 1e-9
    n = n / norm
    flip = np.sign(n[..., 2:3])
    flip[flip == 0] = 1.0
    n = n * flip                                  # 让法向朝相机（-z 方向看过去）
    L = np.asarray(light, np.float32)
    L = L / np.linalg.norm(L)
    # 半兰伯特：硬 Lambert 会让朝向偏离光源的那半边整块变黑（实测 31% 的点 = 0），
    # 点云本来就稀，再压黑就只剩几片亮斑了
    lam = 0.5 + 0.5 * np.clip(n @ L, -1.0, 1.0)
    # 法向来自深度梯度，深度一有噪声就变成横向条纹；对明暗图本身轻度模糊，
    # 条纹会融成平滑的明暗过渡（轮廓由点云位置决定，不会被模糊掉）
    import cv2
    lam = cv2.GaussianBlur(lam.astype(np.float32), (0, 0), 2.0)
    return np.clip(lam, 0.0, 1.0).astype(np.float32).__mul__(255).astype(np.uint8)


# --------------------------------------------------------------------------- #
# 导出（给 HTML 里的 3D 控件）
# --------------------------------------------------------------------------- #


def to_widget(rec: RealRecon, max_points: int = 12000,
              subject_only: bool = True, far_k: float = 0.55) -> dict:
    """把点云转成 HTML 3D 控件吃的格式（int16 坐标 + RGB + 法向明暗）。

    三条处理决定了"看不看得出人"：
      1. `subject_only`：只留主体掩码 + 主体深度带内的点（ROI 里还有地面和背景，
         全画进去人就被埋了）；
      2. **随机**抽样而不是等步长（等步长在斜面上形成规则点阵 = 摩尔纹）；
      3. 带宽带法向算出的 Lambert 明暗 `shd`（单目点云是薄片，只上色仍像贴图）。
    """
    import base64
    p = rec.points
    c = rec.colors
    pidx = rec.pix.copy()

    if subject_only and rec.subject is not None and len(rec.subject) > 200:
        idx = rec.subject
        w = rec.disp.shape[1]
        # 只砍"远背景"，不砍近处：早先用 ±6 个直方图 bin 切一个窄带，结果把
        # 人的躯干切成几片（头、身、腿的深度本来就不一样）。现在用
        #   depth < far_k × 主体中位深度
        # 这样地面、脚、身体都在，只有身后的树丛被去掉。
        d = rec.disp[(pidx[idx] // w).astype(int), (pidx[idx] % w).astype(int)]
        # 注意用**主体自身**的视差中位数，不是整个 ROI 的：ROI 里一半是背景，
        # 用它的中位数当阈值等于没砍（实测 z 跨 1.09，人后面挂了一圈"光晕"）
        # 掩码内部的"空隙"仍然是背景（手臂与身体之间、两腿之间），它们的深度
        # 可能是主体的 3 倍。所以再按主体自身的视差中位数砍一刀远背景。
        ds = rec.disp[(pidx[rec.subject] // w).astype(int),
                      (pidx[rec.subject] % w).astype(int)]
        d_ref = float(np.median(ds))
        idx = idx[d > far_k * d_ref]
        p, c, pidx = p[idx], c[idx], pidx[idx]

    if len(p) > max_points:
        rng = np.random.default_rng(0)
        pick = rng.choice(len(p), max_points, replace=False)
        pick.sort()
        p, c, pidx = p[pick], c[pick], pidx[pick]

    # 稳健分位而不是 min/max：背景那一层会把深度轴撑到 8 倍
    lo, hi = np.percentile(p, 1.0, axis=0), np.percentile(p, 99.0, axis=0)
    inside = np.all((p >= lo - 1e-9) & (p <= hi + 1e-9), axis=1)
    p, c, pidx = p[inside], c[inside], pidx[inside]
    center = (lo + hi) / 2
    radius = float(np.linalg.norm(hi - lo) / 2)

    w = rec.disp.shape[1]
    shade_map = surface_shade(rec)
    shd = shade_map[(pidx // w).astype(int), (pidx % w).astype(int)]

    q = np.clip(np.round((p - center) / radius * 32767), -32768, 32767).astype("<i2")
    # 取景用**过滤之后**的这团点：掩码里的空隙（手臂之间、腿之间）深度可能是
    # 主体的 3 倍，拿它算包围盒会把主体缩成一个小点（实测直接看不见）。
    slo, shi = np.percentile(p, 2.0, axis=0), np.percentile(p, 98.0, axis=0)
    focus = {"center": ((slo + shi) / 2).tolist(),
             "radius": float(np.linalg.norm(shi - slo) / 2)}

    gain = 1.0
    mean_lum = float(np.mean(c.astype(np.float32))) if len(c) else 128.0
    if mean_lum < 110:
        gain = min(2.2, 135.0 / max(mean_lum, 40.0))
    return {
        "center": center.tolist(), "radius": radius, "focus": focus,
        "gain": round(gain, 2),
        "pts": base64.b64encode(q.tobytes()).decode(),
        "col": base64.b64encode(c.astype(np.uint8).tobytes()).decode(),
        "shd": base64.b64encode(shd.astype(np.uint8).tobytes()).decode(),
        "n": int(len(p)),
        "intr": rec.intr,
    }


# --------------------------------------------------------------------------- #
# 落盘
# --------------------------------------------------------------------------- #

_STOPS = [(0.0, (0.19, 0.07, 0.23)), (0.2, (0.24, 0.28, 0.63)),
          (0.4, (0.13, 0.62, 0.75)), (0.6, (0.28, 0.80, 0.51)),
          (0.8, (0.97, 0.53, 0.15)), (1.0, (0.97, 0.24, 0.13))]


def _turbo(t: np.ndarray) -> np.ndarray:
    t = np.clip(t, 0, 1)
    xs = np.array([s[0] for s in _STOPS])
    cs = np.array([s[1] for s in _STOPS], np.float32)
    return (np.stack([np.interp(t, xs, cs[:, k]) for k in range(3)], -1) * 255).astype(np.uint8)


def dump_png(rgb: np.ndarray, path: Path) -> None:
    from PIL import Image
    Image.fromarray(rgb.astype(np.uint8)).save(path)


def dump_depth_png(disp: np.ndarray, path: Path, lo_p=1.0, hi_p=99.0) -> None:
    """视差图：近红远蓝（和报告里其它深度图同一套色表）。"""
    from PIL import Image
    lo, hi = np.percentile(disp, lo_p), np.percentile(disp, hi_p)
    t = (disp - lo) / (hi - lo + 1e-9)
    Image.fromarray(_turbo(t)).save(path)


def dump_subject_overlay(rec: RealRecon, path: Path, alpha: float = 0.55) -> None:
    """把判定为主体的点投回图像坐标画成红色，肉眼核对"切出来的到底是谁"。"""
    from PIL import Image
    img = rec.rgb.astype(np.float32).copy()
    if rec.subject is None or not len(rec.subject):
        Image.fromarray(img.astype(np.uint8)).save(path)
        return
    p = rec.points[rec.subject]
    z = -p[:, 2]
    ok = z > 1e-6
    u = rec.intr["cx"] + p[ok, 0] / z[ok] * rec.intr["fx"]
    v = rec.intr["cy"] - p[ok, 1] / z[ok] * rec.intr["fy"]
    h, w = img.shape[:2]
    ui = np.clip(u.astype(int), 0, w - 1)
    vi = np.clip(v.astype(int), 0, h - 1)
    img[vi, ui] = img[vi, ui] * (1 - alpha) + np.array([255, 40, 40]) * alpha
    Image.fromarray(img.astype(np.uint8)).save(path)
