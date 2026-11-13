"""相机：内参、投影/反投影、挂载到 MOSS 的安装帧，以及渲染器池。

坐标系约定（已在本容器内实测验证，见 tools/verify_projection.py）：

    MuJoCo <camera> 元素：-Z 是光轴（看向 -Z），+X 是图像右，+Y 是图像上。
    MOSS 的安装帧（`moss_camera` / `moss_arm_camera`，URDF 风格）：+X 是光轴，
    +Y 朝左，+Z 朝上（见 detector.py 的 docstring）。
    所以相机相对安装帧的旋转就是把 (右, 上, 后) 映射到 (-y, +z, -x)。

实测（160x120, fovy=60, f=1 m 处 x=+0.2 的球）：
    预测 u=100.8 v=60.0，图像里红色像素质心 (100.5, 59.5)。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import numpy as np

# --------------------------------------------------------------------------- #
# 内参
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Pinhole:
    """方形像素针孔相机。MuJoCo 的相机模型就是这个。"""

    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float

    @classmethod
    def from_hfov(cls, width: int, height: int, hfov_deg: float) -> "Pinhole":
        """从水平视场角构造 —— 用 hfov 而不是 fovy，因为 MOSS 的两个相机
        的数据手册给的是 87x62（16:10）与 70x55（4:3），横向是传感器的定义量。"""
        fx = 0.5 * width / math.tan(0.5 * math.radians(hfov_deg))
        return cls(width=width, height=height, fx=float(fx), fy=float(fx),
                   cx=0.5 * width, cy=0.5 * height)

    @property
    def vfov_deg(self) -> float:
        """渲染裁剪下的垂直视场角 —— 直接喂给 MuJoCo 的 `camera.fovy`。"""
        return math.degrees(2.0 * math.atan(0.5 * self.height / self.fy))

    @property
    def hfov_deg(self) -> float:
        return math.degrees(2.0 * math.atan(0.5 * self.width / self.fx))

    def to_dict(self) -> dict:
        return {"width": self.width, "height": self.height, "fx": self.fx,
                "fy": self.fy, "cx": self.cx, "cy": self.cy,
                "hfov_deg": self.hfov_deg, "vfov_deg": self.vfov_deg}

    @classmethod
    def from_dict(cls, d: dict) -> "Pinhole":
        return cls(d["width"], d["height"], d["fx"], d["fy"], d["cx"], d["cy"])


# --------------------------------------------------------------------------- #
# 四元数 / 旋转
# --------------------------------------------------------------------------- #


def _quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """MuJoCo 顺序 (w, x, y, z)，与 mju_mulQuat 一致：a 左乘 b。"""
    import mujoco
    out = np.zeros(4)
    mujoco.mju_mulQuat(out, np.asarray(a, float), np.asarray(b, float))
    return out


def mat_to_quat(mat: np.ndarray) -> np.ndarray:
    import mujoco
    out = np.zeros(4)
    mujoco.mju_mat2Quat(out, np.asarray(mat, float).ravel())
    return out


def quat_to_mat(q: Iterable[float]) -> np.ndarray:
    import mujoco
    out = np.zeros(9)
    mujoco.mju_quat2Mat(out, np.asarray(list(q), float))
    return out.reshape(3, 3)


#: 相机轴 (右, 上, 后) 在安装帧 (-y, +z, -x) 里的表达 —— 列向量即相机三轴。
_MOUNT_TO_CAM_MAT = np.array([
    [0.0, 0.0, -1.0],
    [-1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
], float)


def mount_to_camera_quat(mount_quat: Iterable[float]) -> np.ndarray:
    """安装帧的朝向 → 相机元素应有的朝向（含 MuJoCo 的 -Z 光轴约定）。"""
    return _quat_mul(np.asarray(list(mount_quat), float),
                     mat_to_quat(_MOUNT_TO_CAM_MAT))


def look_at_mount_quat(pos: Iterable[float], aim: Iterable[float]) -> np.ndarray:
    """URDF 风格安装帧：+X 指向 aim，+Z 尽量朝上。与 moss.arm_camera_quat 同构。"""
    p = np.asarray(list(pos), float)
    a = np.asarray(list(aim), float)
    x = a - p
    x /= np.linalg.norm(x)
    up = np.array([-(a[0] - p[0]), -(a[1] - p[1]), 0.0])
    z = up - x * float(up @ x)
    if np.linalg.norm(z) < 1e-6:            # 光轴竖直时退化
        z = np.array([0.0, 0.0, 1.0])
        z = z - x * float(z @ x)
    z /= np.linalg.norm(z)
    return mat_to_quat(np.column_stack([x, np.cross(z, x), z]).ravel())


# --------------------------------------------------------------------------- #
# 投影 / 反投影
# --------------------------------------------------------------------------- #


def project(points_world: np.ndarray, cam_xpos: np.ndarray,
            cam_xmat: np.ndarray, intr: Pinhole) -> tuple[np.ndarray, np.ndarray]:
    """世界点 → (u, v) 与"沿光轴的正深度"。点在相机后方时深度为负。"""
    pts = np.atleast_2d(np.asarray(points_world, float))
    R = np.asarray(cam_xmat, float).reshape(3, 3)
    pc = (pts - np.asarray(cam_xpos, float)) @ R      # R^T (p - t)
    depth = -pc[:, 2]                                  # MuJoCo: 看向 -Z
    with np.errstate(divide="ignore", invalid="ignore"):
        u = intr.cx + intr.fx * pc[:, 0] / depth
        v = intr.cy - intr.fy * pc[:, 1] / depth
    return np.column_stack([u, v]), depth


def unproject(depth_img: np.ndarray, intr: Pinhole,
              max_range: float = 2.0) -> tuple[np.ndarray, np.ndarray]:
    """深度图 → 相机系点云 (N,3) 与有效掩码 (H,W)。"""
    h, w = depth_img.shape
    valid = np.isfinite(depth_img) & (depth_img > 0.0) & (depth_img < max_range)
    vs, us = np.nonzero(valid)
    d = depth_img[vs, us]
    x = (us + 0.5 - intr.cx) / intr.fx * d
    y = -(vs + 0.5 - intr.cy) / intr.fy * d
    z = -d
    return np.column_stack([x, y, z]), valid


def frame_to_base(points_cam: np.ndarray, cam_xpos: np.ndarray,
                  cam_xmat: np.ndarray, base_xpos: np.ndarray,
                  base_xmat: np.ndarray) -> np.ndarray:
    """相机系点云 → rover 系。"""
    R_cam = np.asarray(cam_xmat, float).reshape(3, 3)
    world = points_cam @ R_cam.T + np.asarray(cam_xpos, float)
    R_base = np.asarray(base_xmat, float).reshape(3, 3)
    return (world - np.asarray(base_xpos, float)) @ R_base


def matrix_to_base(cam_xpos, cam_xmat, base_xpos, base_xmat) -> np.ndarray:
    """相机位姿 → rover 系的 4x4（列是相机轴，故 T 的旋转块 = R_base^T R_cam）。"""
    R_cam = np.asarray(cam_xmat, float).reshape(3, 3)
    R_base = np.asarray(base_xmat, float).reshape(3, 3)
    T = np.eye(4)
    T[:3, :3] = R_base.T @ R_cam
    T[:3, 3] = R_base.T @ (np.asarray(cam_xpos, float) - np.asarray(base_xpos, float))
    return T


# --------------------------------------------------------------------------- #
# 渲染器池
# --------------------------------------------------------------------------- #


class RendererPool:
    """按 (W,H) 复用 `mujoco.Renderer`。

    两个坑，都在本容器里实测过：
      1. 每次新建 Renderer 而不 close，第 2 个之后返回的是未初始化内存（随机色块）。
      2. 同一个 Renderer 在 RGB / 深度之间切换是安全的，但 `update_scene` 必须
         在切换后重新调用。
    """

    def __init__(self, model):
        self._model = model
        self._pool: dict[tuple[int, int], object] = {}
        self._depth = False

    def get(self, width: int, height: int):
        import mujoco
        key = (int(width), int(height))
        r = self._pool.get(key)
        if r is None:
            r = mujoco.Renderer(self._model, height=int(height), width=int(width))
            self._pool[key] = r
        return r

    def render(self, data, camera: str, width: int, height: int,
               depth: bool = False, scene_option=None):
        r = self.get(width, height)
        if depth and not self._depth:
            r.enable_depth_rendering()
            self._depth = True
        elif not depth and self._depth:
            r.disable_depth_rendering()
            self._depth = False
        if scene_option is not None:
            r.update_scene(data, camera=camera, scene_option=scene_option)
        else:
            r.update_scene(data, camera=camera)
        out = r.render()
        return np.array(out, copy=True)

    def close(self) -> None:
        for r in self._pool.values():
            try:
                r.close()
            except Exception:
                pass
        self._pool.clear()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False
