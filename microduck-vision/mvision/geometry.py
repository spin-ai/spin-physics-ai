"""几何重建：深度 → 点云 → 平面 → 聚类 → 六元组。

纯净 numpy，不依赖 open3d（aarch64 上 wheel 不可靠）。所有中间量都在
**rover 系**，与 `moss.OBS_TARGET_BASE` 的语义一致。

输出（`Cluster`）就是感知契约里的东西：
    pos      质心，米
    axis     主轴单位向量（圆柱 = 轴向，方盒 = 最长边）
    upright  |axis·z|
    top_z    物体顶端高度，米
    radius   垂直于主轴的半径（圆柱/近似）
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from . import config
from .scene import ViewFrame


@dataclass
class Cluster:
    points: np.ndarray        # (N,3) rover 系
    centroid: np.ndarray      # (3,)
    axis: np.ndarray          # (3,) 单位
    top_z: float
    radius: float
    extent: np.ndarray        # (3,) 特征值，降序
    half_length: float = 0.0
    #: 共识重建里投了票的视角数（单帧重建时不设）
    n_views: int = 1
    #: 轴线点可能落在点云之外（校正后），所以 bbox 单独留着
    _bbox: np.ndarray | None = None

    @property
    def n(self) -> int:
        return int(len(self.points))

    @property
    def upright(self) -> float:
        return float(abs(self.axis[2]))

    @property
    def axis_defined(self) -> bool:
        return self.upright < 0.90

    @property
    def yaw_double(self) -> tuple[float, float] | None:
        if not self.axis_defined:
            return None
        psi = math.atan2(self.axis[1], self.axis[0])
        return (math.cos(2 * psi), math.sin(2 * psi))

    @property
    def box(self) -> np.ndarray:
        if self._bbox is not None:
            return self._bbox
        lo = self.points.min(axis=0)
        hi = self.points.max(axis=0)
        return np.concatenate([lo, hi])

    def to_dict(self, label: str = "object", conf: float = 1.0) -> dict:
        return {"label": label, "conf": float(conf), "n": self.n,
                "pos": self.centroid.tolist(), "axis": self.axis.tolist(),
                "upright": self.upright, "top_z": float(self.top_z),
                "radius": float(self.radius),
                "yaw_double": self.yaw_double, "n_views": self.n_views,
                "bbox": self.box.tolist()}


# --------------------------------------------------------------------------- #
# 点云
# --------------------------------------------------------------------------- #


def points_from_views(frames: list[ViewFrame], masks: list[np.ndarray] | None = None,
                      max_range: float = config.MAX_RANGE_M,
                      roi: tuple[float, ...] = config.ROI,
                      stride: int = 1) -> np.ndarray:
    """把多视角深度图拼成 rover 系点云。masks 为 [(H,W) bool] 时只保留掩码内点。"""
    out: list[np.ndarray] = []
    for i, f in enumerate(frames):
        dep = f.depth[::stride, ::stride]
        valid = np.isfinite(dep) & (dep > 0.0) & (dep < max_range)
        if masks is not None and masks[i] is not None:
            valid &= np.asarray(masks[i], bool)[::stride, ::stride]
        vs, us = np.nonzero(valid)
        if len(vs) == 0:
            continue
        d = dep[vs, us]
        x = (us * stride + 0.5 - f.intr.cx) / f.intr.fx * d
        y = -(vs * stride + 0.5 - f.intr.cy) / f.intr.fy * d
        z = -d
        p_cam = np.column_stack([x, y, z])
        R = f.T_base_cam[:3, :3]
        t = f.T_base_cam[:3, 3]
        out.append(p_cam @ R.T + t)
    if not out:
        return np.zeros((0, 3), np.float32)
    pts = np.concatenate(out, axis=0).astype(np.float32)
    return in_roi(pts, roi)


def in_roi(points: np.ndarray, roi: tuple[float, ...] = config.ROI) -> np.ndarray:
    x0, x1, y0, y1, z0, z1 = roi
    m = ((points[:, 0] >= x0) & (points[:, 0] <= x1)
         & (points[:, 1] >= y0) & (points[:, 1] <= y1)
         & (points[:, 2] >= z0) & (points[:, 2] <= z1))
    return points[m]


def voxel_downsample(points: np.ndarray, voxel: float = 0.006) -> np.ndarray:
    if len(points) == 0:
        return points
    keys = np.floor(points / voxel).astype(np.int64)
    _, idx = np.unique(keys, axis=0, return_index=True)
    return points[np.sort(idx)]


# --------------------------------------------------------------------------- #
# 平面
# --------------------------------------------------------------------------- #


def ransac_plane(points: np.ndarray, thresh: float = 0.006, iters: int = 240,
                 rng: np.random.Generator | None = None):
    """返回 (normal, offset, inlier_mask)，平面为 n·x + d = 0。"""
    rng = rng or np.random.default_rng(0)
    n_pts = len(points)
    if n_pts < 3:
        return np.array([0.0, 0.0, 1.0]), 0.0, np.zeros(n_pts, bool)
    best = np.zeros(n_pts, bool)
    best_n, best_d = np.array([0.0, 0.0, 1.0]), 0.0
    for _ in range(iters):
        idx = rng.choice(n_pts, 3, replace=False)
        p0, p1, p2 = points[idx]
        nrm = np.cross(p1 - p0, p2 - p0)
        norm = np.linalg.norm(nrm)
        if norm < 1e-9:
            continue
        nrm = nrm / norm
        d = -float(nrm @ p0)
        inl = np.abs(points @ nrm + d) < thresh
        if inl.sum() > best.sum():
            best, best_n, best_d = inl, nrm, d
    return best_n, best_d, best


# --------------------------------------------------------------------------- #
# 聚类
# --------------------------------------------------------------------------- #

#: 聚类只连"前向"邻居（避免重复）；法向估计要全部 26 个邻居。
_NEIGH = [(dx, dy, dz) for dx in (-1, 0, 1) for dy in (-1, 0, 1)
          for dz in (-1, 0, 1) if (dx, dy, dz) > (0, 0, 0)]
_NEIGH_ALL = [(dx, dy, dz) for dx in (-1, 0, 1) for dy in (-1, 0, 1)
              for dz in (-1, 0, 1) if (dx, dy, dz) != (0, 0, 0)]


def cluster_points(points: np.ndarray, voxel: float = 0.010,
                   min_points: int = 60) -> list[np.ndarray]:
    """体素邻居 + 并查集（27 邻域）。"""
    if len(points) == 0:
        return []
    keys = np.floor(points / voxel).astype(np.int64)
    _, inv = np.unique(keys, axis=0, return_inverse=True)
    n_cells = int(inv.max()) + 1
    parent = np.arange(n_cells)

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    cell_key: dict[tuple, int] = {}
    for i, c in enumerate(inv):
        cell_key.setdefault(tuple(keys[i]), int(c))

    for k, c in cell_key.items():
        for dx, dy, dz in _NEIGH:
            other = cell_key.get((k[0] + dx, k[1] + dy, k[2] + dz))
            if other is not None:
                ra, rb = find(c), find(other)
                if ra != rb:
                    parent[rb] = ra

    labels = np.array([find(int(c)) for c in inv])
    out: list[np.ndarray] = []
    for lab in np.unique(labels):
        m = labels == lab
        if m.sum() >= min_points:
            out.append(points[m])
    out.sort(key=len, reverse=True)
    return out


# --------------------------------------------------------------------------- #
# 主轴 / 半径 / 顶高
# --------------------------------------------------------------------------- #


def _pca_axis(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    c = points.mean(axis=0)
    p = points - c
    cov = p.T @ p / max(len(points) - 1, 1)
    w, v = np.linalg.eigh(cov)
    order = np.argsort(w)[::-1]
    return v[:, order[0]], w[order]


def _grid_normals(points: np.ndarray, voxel: float = 0.010,
                  k_min: int = 6) -> np.ndarray:
    """体素邻域的 PCA 法向（无 sklearn；2x10^4 点约 0.2 s）。"""
    keys = np.floor(points / voxel).astype(np.int64)
    cells: dict[tuple, list[int]] = {}
    for i, k in enumerate(keys):
        cells.setdefault(tuple(k), []).append(i)

    normals = np.zeros_like(points)
    for k, idxs in cells.items():
        sel = list(idxs)
        for dx, dy, dz in _NEIGH_ALL:
            other = cells.get((k[0] + dx, k[1] + dy, k[2] + dz))
            if other:
                sel.extend(other)
        if len(sel) < k_min:
            continue
        p = points[sel]
        p = p - p.mean(axis=0)
        w, v = np.linalg.eigh(p.T @ p)
        n = v[:, 0]
        if (p @ n).std() > 0.01:      # 邻域不是平面（棱、噪声）→ 丢弃
            continue
        n = n / max(np.linalg.norm(n), 1e-9)
        normals[idxs] = n
    return normals


def estimate_axis(points: np.ndarray, shape: str = "cylinder") -> tuple[np.ndarray, np.ndarray]:
    """返回 (axis, eigenvalues)。圆柱走「侧表面法向的零空间」，其余走 PCA。"""
    v1, ev = _pca_axis(points)
    if shape != "cylinder" or len(points) < 200:
        return _canonical(v1), ev
    normals = _grid_normals(points)
    valid = np.linalg.norm(normals, axis=1) > 0.5
    if valid.sum() < 100:
        return _canonical(v1), ev
    # 用第一轮的主轴剔除"顶/底面"（法向与候选轴平行的点），只留侧表面
    side = valid & (np.abs(normals @ v1) < 0.6)
    if side.sum() < 80:
        side = valid
    n = normals[side]
    w, v = np.linalg.eigh(n.T @ n / len(n))
    axis = v[:, 0]
    # 主轴应尽量与 PCA 主轴同向（消除法向拟合在退化时的跳变）
    if abs(float(axis @ v1)) < 0.5:
        axis = v1
    elif float(axis @ v1) < 0:
        axis = -axis
    return _canonical(axis), ev


def _canonical(axis: np.ndarray) -> np.ndarray:
    """统一符号：z 分量为正；接近水平时取 x 为正。"""
    a = np.asarray(axis, float)
    a = a / max(np.linalg.norm(a), 1e-9)
    if a[2] < 0:
        a = -a
    elif abs(a[2]) < 1e-6 and a[0] < 0:
        a = -a
    return a


def _basis_perp(axis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """与 axis 正交的两个单位向量。"""
    a = np.asarray(axis, float)
    seed = np.array([0.0, 0.0, 1.0]) if abs(a[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    e1 = np.cross(a, seed)
    e1 /= max(np.linalg.norm(e1), 1e-9)
    return e1, np.cross(a, e1)


def _circle_fit(a: np.ndarray, b: np.ndarray) -> tuple[float, float, float, float]:
    """代数圆拟合（Kåsa）+ 两轮剔除。返回 (cx, cy, r, 残差)。

    为什么不用"点 - r·径向"的迭代：只看到一段窄弧时，初始 r 被低估，
    修正量跟着被低估，迭代收敛到一个偏心的解 —— 实测直立罐子稳定偏 15 mm，
    正好是半径的低估量（21.6 vs 33.0）。代数拟合把 c 和 r 一起解出来，没有
    这个自洽陷阱。
    """
    m = np.ones(len(a), bool)
    cx = cy = 0.0
    r = 0.0
    res = np.inf
    for _ in range(3):
        if m.sum() < 6:
            break
        A = np.column_stack([a[m], b[m], np.ones(m.sum())])
        rhs = -(a[m] ** 2 + b[m] ** 2)
        sol, *_ = np.linalg.lstsq(A, rhs, rcond=None)
        D, E, F = sol
        cx, cy = -0.5 * D, -0.5 * E
        disc = cx * cx + cy * cy - F
        if disc <= 0:
            break
        r = float(np.sqrt(disc))
        res = float(np.mean(np.abs(np.hypot(a - cx, b - cy) - r)))
        # 剔除残差最大的 20%（端面、遮挡边界、掩码边缘的混血像素）
        err = np.abs(np.hypot(a - cx, b - cy) - r)
        thr = np.percentile(err, 80)
        m = err <= max(thr, 1e-4)
    return float(cx), float(cy), float(r), res


def _cylinder_fit(points: np.ndarray, axis: np.ndarray):
    """返回 (轴线中心, 半径, 半长, 圆拟合残差)。"""
    e1, e2 = _basis_perp(axis)
    c0 = points.mean(axis=0)
    a = (points - c0) @ e1
    b = (points - c0) @ e2
    cx, cy, r, res = _circle_fit(a, b)
    c = c0 + e1 * cx + e2 * cy
    along = (points - c) @ axis
    lo, hi = np.percentile(along, [0.5, 99.5])
    c = c + axis * (0.5 * (lo + hi))
    return c, float(r), float(0.5 * (hi - lo)), float(res)


def cluster_from_points(points: np.ndarray, shape: str = "cylinder") -> Cluster:
    axis, ev = estimate_axis(points, shape=shape)
    up = float(abs(axis[2]))
    if shape == "cylinder" and len(points) >= 120:
        centroid, radius, half_len, circ_res = _cylinder_fit(points, axis)
        if not (0.004 < radius < 0.4) or circ_res > 0.02:
            centroid = points.mean(axis=0)
            rel = points - centroid
            radius = float(np.median(np.linalg.norm(
                rel - np.outer(rel @ axis, axis), axis=1)))
            half_len = float(0.5 * np.ptp(rel @ axis))
        # 顶高由**模型**给，而不是点云的 z 分位：躺着的圆柱从平视角度看
        # 顶面是掠射的，点云在顶部天然欠采样，99 分位会比真值低 10 mm。
        top_z = float(centroid[2] + half_len * up + radius * (1.0 - up))
    else:
        centroid = points.mean(axis=0)
        rel = points - centroid
        perp = rel - np.outer(rel @ axis, axis)
        radius = float(np.median(np.linalg.norm(perp, axis=1)))
        half_len = float(0.5 * np.ptp((points - centroid) @ axis))
        top_z = float(np.percentile(points[:, 2], 99.0))
    return Cluster(points=points, centroid=centroid, axis=axis, top_z=top_z,
                   radius=float(radius), extent=ev, half_length=float(half_len))


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #


def reconstruct(frames: list[ViewFrame], masks: list[np.ndarray] | None = None,
                roi: tuple[float, ...] = config.ROI,
                max_range: float = config.MAX_RANGE_M,
                shape: str = "cylinder",
                voxel: float = 0.010,
                min_points: int = 80,
                drop_floor: bool = True) -> list[Cluster]:
    """多视角点云 → 去掉地板 → 聚类 → 每个簇一个六元组。按点数降序。"""
    pts = points_from_views(frames, masks=masks, max_range=max_range, roi=roi)
    if len(pts) < min_points:
        return []
    if drop_floor:
        n, d, inl = ransac_plane(pts)
        # 只把"接近竖直法向"的平面当地板，避免把墙面/机壳当桌面扔掉
        if abs(float(n[2])) > 0.85 and inl.sum() > 0.25 * len(pts):
            pts = pts[~inl]
    if len(pts) < min_points:
        return []
    pts = voxel_downsample(pts, voxel=0.005)
    clusters = cluster_points(pts, voxel=voxel, min_points=min_points)
    return [cluster_from_points(c, shape=shape) for c in clusters]


def reconstruct_consensus(frames: list[ViewFrame], masks: list[np.ndarray] | None = None,
                          roi: tuple[float, ...] = config.ROI,
                          max_range: float = config.MAX_RANGE_M,
                          shape: str = "cylinder", agree_m: float = 0.030,
                          min_views: int = 2,
                          **kw) -> list[Cluster]:
    """多视角**共识**重建：先每帧各自出候选，再按空间一致性投票，最后合并重拟合。

    为什么需要它：单帧的重建会被"目标只剩一条边"或"掩码吃到机器人自身"毁掉。
    实测有种子在腕部视角上偏 191 mm，但另外三个视角都在 1 mm 内 —— 共识把
    那一帧当噪声剔掉，而不是让它投票。只有一帧可用时退化为普通重建。
    """
    per_view: list[list[Cluster]] = []
    for i, f in enumerate(frames):
        m = None if masks is None else [masks[i]] * 1
        per_view.append(reconstruct([f], masks=m, roi=roi, max_range=max_range,
                                    shape=shape, **kw))
    cands: list[tuple[int, Cluster]] = []
    for vi, cl in enumerate(per_view):
        for c in cl[:3]:
            cands.append((vi, c))
    if not cands:
        return []

    # 贪心分组：质心距离 < agree_m
    groups: list[list[tuple[int, Cluster]]] = []
    for vi, c in sorted(cands, key=lambda t: -t[1].n):
        for g in groups:
            if np.linalg.norm(g[0][1].centroid - c.centroid) < agree_m:
                g.append((vi, c))
                break
        else:
            groups.append([(vi, c)])

    def score(g):
        return (len({vi for vi, _ in g}), sum(c.n for _, c in g))

    groups.sort(key=score, reverse=True)
    best = groups[0]
    if len({vi for vi, _ in best}) < min_views and len(frames) > 1:
        # 没有共识：说明目标只在个别视角里，退回单帧最好的那个
        best = [max(cands, key=lambda t: t[1].n)]
    pts = np.concatenate([c.points for _, c in best], axis=0)
    merged = cluster_from_points(voxel_downsample(pts, 0.004), shape=shape)
    merged.n_views = len({vi for vi, _ in best})
    return [merged]


def dominant(clusters: list[Cluster]) -> Cluster | None:
    return clusters[0] if clusters else None


def degrees_between(a: np.ndarray, b: np.ndarray, symmetric: bool = True) -> float:
    """两个主轴之间的夹角（度）。`symmetric` 时按 mod π 取小 —— 圆柱没有头尾。"""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    c = abs(float(a @ b)) / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-12)
    if not symmetric:
        c = float(a @ b) / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-12)
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))
