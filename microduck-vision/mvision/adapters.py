"""感知 → obs。左边是 `Perception`，右边是两条真实存在的契约。

MOSS（32 维，见 robots/moss.py 的 contract v0）：
    24:27  target_base   目标在 rover 系的 xyz，米
    27     target_seen   该定位新鲜
    28:30  target_axis   cos(2ψ), sin(2ψ) —— 倍角，圆柱不区分"头尾"
    30     target_upright |cos(tilt)|
    31     spare         空闲槽；本方案放顶端高度（米）

microduck（61 维，见 sensors/detector.py）：
    没有 3D 槽位，视觉走的是 `Detection.as_payload()`：
    {cls, bearing, elevation, width, conf}，异常角、无度量。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from . import config


@dataclass
class Perception:
    t: float = 0.0
    label: str = "object"
    pos: np.ndarray | None = None
    axis: np.ndarray | None = None
    upright: float | None = None
    top_z: float | None = None
    radius: float | None = None
    conf: float = 1.0
    source: str = "geometry"
    n_points: int = 0
    extra: dict = field(default_factory=dict)

    # ------------------------------------------------------------ 新鲜度

    def is_stale(self, now: float, limit: float = config.STALE_S) -> bool:
        return (self.pos is None) or (now - self.t > limit)

    # ------------------------------------------------------------ 派生量

    @property
    def axis_double(self) -> tuple[float, float] | None:
        if self.axis is None or (self.upright is not None and self.upright >= 0.90):
            return None
        psi = math.atan2(float(self.axis[1]), float(self.axis[0]))
        return (math.cos(2 * psi), math.sin(2 * psi))

    def to_dict(self) -> dict:
        return {"t": self.t, "label": self.label,
                "pos": None if self.pos is None else np.asarray(self.pos).tolist(),
                "axis": None if self.axis is None else np.asarray(self.axis).tolist(),
                "upright": self.upright, "top_z": self.top_z,
                "radius": self.radius, "conf": self.conf,
                "source": self.source, "n_points": self.n_points,
                "axis_double": self.axis_double, "extra": self.extra}

    @classmethod
    def from_dict(cls, d: dict) -> "Perception":
        return cls(t=d.get("t", 0.0), label=d.get("label", "object"),
                   pos=None if d.get("pos") is None else np.array(d["pos"], float),
                   axis=None if d.get("axis") is None else np.array(d["axis"], float),
                   upright=d.get("upright"), top_z=d.get("top_z"),
                   radius=d.get("radius"), conf=d.get("conf", 1.0),
                   source=d.get("source", "geometry"),
                   n_points=d.get("n_points", 0), extra=d.get("extra") or {})


def from_cluster(cluster, t: float, label: str = "object",
                 conf: float = 1.0, source: str = "geometry") -> Perception:
    return Perception(t=t, label=label, pos=np.asarray(cluster.centroid, float),
                      axis=np.asarray(cluster.axis, float),
                      upright=cluster.upright, top_z=float(cluster.top_z),
                      radius=float(cluster.radius), conf=conf, source=source,
                      n_points=int(cluster.n))


# --------------------------------------------------------------------------- #
# MOSS 32 维
# --------------------------------------------------------------------------- #


def moss_slots(p: Perception, now: float, *, publish_axis: bool = True,
               publish_top: bool = True, publish_upright: bool = True,
               stale_s: float = config.STALE_S) -> dict:
    """按 task 开关产出槽位值。（`approach`/`stow` 会把 28:32 恒置零，
    所以这里必须能分别关掉，否则会破坏已训练策略的输入分布。）"""
    seen = 0.0 if p.is_stale(now, stale_s) else 1.0
    out = {
        "target_base": np.zeros(3), "target_seen": seen,
        "target_axis": np.zeros(2), "target_upright": 0.0, "spare": 0.0,
    }
    if seen == 0.0:
        return out
    out["target_base"] = np.asarray(p.pos, float)
    if publish_axis:
        ad = p.axis_double
        if ad is not None:
            out["target_axis"] = np.array(ad, float)
    if publish_upright and p.upright is not None:
        out["target_upright"] = float(p.upright)
    if publish_top and p.top_z is not None:
        out["spare"] = float(p.top_z)
    return out


def apply_to_moss_obs(obs: np.ndarray, slots: dict) -> np.ndarray:
    """把 moss_slots 写进 32 维 obs。返回同一个数组（原地写）。"""
    import microduck_local.robots.moss as moss
    obs = np.asarray(obs)
    if obs.shape[-1] != moss.OBS_DIM:
        raise ValueError(f"expected {moss.OBS_DIM}-dim MOSS obs, got {obs.shape}")
    obs[moss.OBS_TARGET_BASE] = slots["target_base"]
    obs[moss.OBS_TARGET_SEEN] = slots["target_seen"]
    if obs[moss.OBS_TARGET_AXIS].size == 2:
        obs[moss.OBS_TARGET_AXIS] = slots["target_axis"]
    if obs[moss.OBS_TARGET_UPRIGHT].size == 1:
        obs[moss.OBS_TARGET_UPRIGHT] = slots["target_upright"]
    if obs[moss.OBS_SPARE].size == 1:
        obs[moss.OBS_SPARE] = slots["spare"]
    return obs


# --------------------------------------------------------------------------- #
# microduck 61 维 → detector payload
# --------------------------------------------------------------------------- #


def _mount_from_camera(mount_xmat_cam_frame: np.ndarray) -> np.ndarray:
    """MuJoCo 相机轴 (右, 上, 后) → 安装帧 (前, 左, 上)。detector.py 用后者。"""
    R = np.asarray(mount_xmat_cam_frame, float).reshape(3, 3)
    # 相机三轴在世界的表达是 R 的列；安装帧三轴 = (-z_cam, -x_cam, +y_cam)
    return np.column_stack([-R[:, 2], -R[:, 0], R[:, 1]])


def duck_payload(p: Perception, cam_xpos: np.ndarray, cam_xmat: np.ndarray,
                 radius: float | None = None, cls: str = "toy") -> dict:
    """产出与 `sensors/detector.Detection.as_payload()` 同形的 dict。"""
    R = _mount_from_camera(cam_xmat)
    pc = R.T @ (np.asarray(p.pos, float) - np.asarray(cam_xpos, float))
    rng = float(np.linalg.norm(pc))
    bearing = math.atan2(pc[1], pc[0])
    elevation = math.atan2(pc[2], math.hypot(pc[0], pc[1]))
    r = float(radius if radius is not None else (p.radius or 0.03))
    width = 2.0 * math.atan2(r, max(rng, 1e-3))
    return {"cls": cls, "bearing": float(bearing), "elevation": float(elevation),
            "width": float(width), "conf": float(p.conf),
            "range": rng, "pos_base": np.asarray(p.pos, float).tolist(),
            "t": float(p.t)}


def duck_bearing_command(payload: dict, gain: float = 1.0,
                         deadband: float = 0.05) -> float:
    """检测结果 → 转动命令的参考实现（微鸭的 brain 就是这么用 bearing 的）。"""
    b = float(payload["bearing"])
    if abs(b) < deadband:
        return 0.0
    return float(np.clip(gain * b, -1.0, 1.0))
