"""MOSS 场景：构建 spec、摆物体、读真值、按视角渲染。

这里刻意**不用** `MossPickEnv` —— 那个 env 的 obs 契约、抓取物理、handover 采样
都跟"拍几张照片"无关，而且它每次 reset 要跑 2 s 的 deploy。我们要的是：
一个能任意摆物体、任意摆相机、并且随时能读真值的静态场景。
所以直接用 `moss.scene_spec()` + 自己的 keyframe 摆放，物理仍然是真的
（同样的 MJCF、同样的 2 ms 步长），只是不由策略驱动。
"""

from __future__ import annotations

import dataclasses
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import config
from .camera import (Pinhole, RendererPool, matrix_to_base,
                     mount_to_camera_quat)

if config.MICRODUCK_SRC not in sys.path:
    sys.path.insert(0, config.MICRODUCK_SRC)


#: 上游的 119 个文件；少于这个数就是半成品目录（`fetch-robot` 逐文件拉，
#: 网络一抖就留下一个"看起来有东西"的目录，然后在 apply_visuals 里报缺 STL）。
MOSS_MIN_FILES = 110


def _usable_moss_dir(path: str | os.PathLike | None) -> bool:
    if not path:
        return False
    root = Path(path)
    if not (root / "model" / "moss_robot.xml").is_file():
        return False
    if not (root / "model" / "meshes").is_dir():
        return False
    return sum(1 for _ in (root / "model").rglob("*")) >= MOSS_MIN_FILES


def _resolve_moss_dir() -> Path:
    """定出 MOSS 资产目录，**必须在 import moss 之前**做 —— 它读一次模块常量。

    为什么要扫这么多地方：`MICRODUCK_MOSS_DIR` 在镜像里指向 /opt/moss（构建时
    烤进去的），但开发时更常见的是它已经在 /data/moss（named volume）。上游的
    `fetch-robot moss` 从 raw.githubusercontent.com 逐文件拉 119 次，网络一抖就
    留个半成品目录 —— 所以这里要求 `moss_robot.xml` 和 `meshes/` 都在才算数。
    """
    tried = []
    for cand in (os.environ.get("MICRODUCK_MOSS_DIR"), "/opt/moss", "/data/moss",
                 str(Path.home() / ".cache" / "moss"),
                 "/app/microduck_local/.cache/moss"):
        if not cand:
            continue
        tried.append(cand)
        if _usable_moss_dir(cand):
            os.environ["MICRODUCK_MOSS_DIR"] = str(cand)
            return Path(cand)
    raise RuntimeError(
        "MOSS 资产不完整，试过：" + ", ".join(tried) + "\n"
        "修法（容器里）：\n"
        "  docker exec -u root -e HOME=/root -e MICRODUCK_MOSS_DIR=/opt/moss \\n"
        "      microduck fetch-robot moss\n"
        "只跑 microduck 四足的话不需要 MOSS 资产。")


MOSS_DIR = _resolve_moss_dir()

import mujoco  # noqa: E402

import microduck_local.robots.moss as moss  # noqa: E402
from microduck_local.robots.moss_env import GRASP_PROPS, POCKET  # noqa: E402


# --------------------------------------------------------------------------- #
# 数据结构
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class CameraSpec:
    name: str          # MJCF <camera> 名
    view: str          # "front" | "wrist"
    width: int
    height: int
    hfov_deg: float

    @property
    def intr(self) -> Pinhole:
        return Pinhole.from_hfov(self.width, self.height, self.hfov_deg)


#: 前视 D455：87x62 度（16:10，所以按 320x200 渲染而不是 4:3 —— 用 4:3 会把
#: 水平视场裁成 77 度，跟手册对不上）。腕部 D405：70x55 度（恰好 4:3）。
FRONT = CameraSpec(config.CAM_FRONT, "front", config.RENDER_W,
                   int(round(config.RENDER_W * 62 / 87)), moss.CAMERA_HFOV_DEG)
#: 腕部是 4:3（70x55 度），高度由宽度推出 —— 否则改 MV_WIDTH 时会悄悄变成 2:1，
#: 内参、视场角、渲染内容一起偏掉。
WRIST = CameraSpec(config.CAM_WRIST, "wrist", config.RENDER_W,
                   int(round(config.RENDER_W * 3 / 4)), moss.ARM_CAMERA_HFOV_DEG)

CAMERAS: dict[str, CameraSpec] = {c.view: c for c in (FRONT, WRIST)}


@dataclass
class ViewFrame:
    view: str
    rgb: np.ndarray        # (H, W, 3) uint8
    depth: np.ndarray      # (H, W) float32, 米
    intr: Pinhole
    T_base_cam: np.ndarray  # 4x4，相机轴 (右/上/后) 在 rover 系
    cam_xpos: np.ndarray
    cam_xmat: np.ndarray
    t: float
    base_pose: dict
    arm_pose: np.ndarray

    def to_meta(self) -> dict:
        return {"view": self.view, "t": self.t,
                "intr": self.intr.to_dict(),
                "T_base_cam": self.T_base_cam.tolist(),
                "cam_xpos": self.cam_xpos.tolist(),
                "cam_xmat": self.cam_xmat.reshape(3, 3).tolist(),
                "base_pose": self.base_pose,
                "arm_pose": self.arm_pose.tolist()}


@dataclass
class GroundTruth:
    label: str
    pos: np.ndarray        # (3,) rover 系
    axis: np.ndarray       # (3,) 单位向量，rover 系
    radius: float
    half_height: float
    shape: str

    @property
    def upright(self) -> float:
        return float(abs(self.axis[2]))

    @property
    def top_z(self) -> float:
        up = self.upright
        return float(self.pos[2] + self.half_height * up
                     + self.radius * (1.0 - up))

    @property
    def axis_defined(self) -> bool:
        """侧躺时主轴有意义；竖直时它的方位角是噪声（仓库同样把 AXIS 置零）。"""
        return self.upright < 0.90

    @property
    def yaw_double(self) -> tuple[float, float] | None:
        if not self.axis_defined:
            return None
        psi = math.atan2(self.axis[1], self.axis[0])
        return (math.cos(2 * psi), math.sin(2 * psi))

    def to_dict(self) -> dict:
        return {"label": self.label, "pos": self.pos.tolist(),
                "axis": self.axis.tolist(), "upright": self.upright,
                "top_z": self.top_z, "radius": self.radius,
                "half_height": self.half_height, "shape": self.shape,
                "axis_defined": self.axis_defined}


# --------------------------------------------------------------------------- #
# 场景
# --------------------------------------------------------------------------- #


def _quat_from_axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = np.asarray(axis, float)
    axis = axis / np.linalg.norm(axis)
    half = 0.5 * angle
    return np.array([math.cos(half), *(math.sin(half) * axis)])


def _up_to(axis: np.ndarray) -> np.ndarray:
    """把物体的 +z 转到给定方向的四元数（绕与 z 的叉积轴旋转）。"""
    z = np.array([0.0, 0.0, 1.0])
    a = np.asarray(axis, float)
    a = a / np.linalg.norm(a)
    dot = float(np.clip(z @ a, -1.0, 1.0))
    if dot > 0.999999:
        return np.array([1.0, 0.0, 0.0, 0.0])
    if dot < -0.999999:
        return np.array([0.0, 1.0, 0.0, 0.0])
    return _quat_from_axis_angle(np.cross(z, a), math.acos(dot))


@dataclass
class SceneConfig:
    seed: int = 0
    prop: str = "can"
    clutter: int = 2
    #: 目标姿态：upright = 直立，lying = 侧躺
    target_pose: str = "random"       # upright | lying | random
    #: 目标位置抖动（rover 系）
    x_range: tuple[float, float] = (0.23, 0.30)
    y_range: tuple[float, float] = (-0.10, 0.10)
    #: 视角。三个前视（底盘沿 y/yaw 平移，形成宽基线）+ 一个腕部（抓取视角）。
    views: tuple[str, ...] = ("front", "front", "front", "wrist")
    #: (base_x, base_y, base_yaw)。前视要退到 ~0.5 m —— D455 的实测最小
    #: 深度是 0.52 m（moss.DEPTH_MIN_RANGE_M），贴着 12 cm 拍是不真实的。
    base_variants: tuple[tuple[float, float, float], ...] = (
        (-0.42, 0.00, 0.00), (-0.34, 0.13, 0.22), (-0.34, -0.13, -0.22),
        (-0.26, 0.05, 0.08))
    render_wrist: bool = True


class MossScene:
    """一个 MOSS + 一个橙色目标 + N 个干扰物的静态工作区。"""

    def __init__(self, cfg: SceneConfig | None = None):
        self.cfg = cfg or SceneConfig()
        self.rng = np.random.default_rng(self.cfg.seed)
        self.spec = moss.scene_spec()

        base_prop = GRASP_PROPS[self.cfg.prop]
        self.prop = dataclasses.replace(base_prop, rgba=config.TARGET_RGBA)
        self.prop.add_to(self.spec, name="target")

        clutter_props = []
        kinds = [k for k in GRASP_PROPS if k != self.cfg.prop] or ["can"]
        for k in range(self.cfg.clutter):
            c = dataclasses.replace(GRASP_PROPS[kinds[k % len(kinds)]],
                                    rgba=config.CLUTTER_RGBA[k % len(config.CLUTTER_RGBA)])
            c.add_to(self.spec, name=f"clutter{k}")
            clutter_props.append(c)
        self.clutter_props = clutter_props

        self._add_cameras()

        # keyframe：10 个机器人自由度 + 每个自由体 7 个
        key = self.spec.key(moss.HOME_KEY)
        qpos = list(np.asarray(key.qpos, float)[:10])
        target_qpos = self._sample_target()
        self._target_xy = np.asarray(target_qpos[:2], float)
        qpos += list(target_qpos)
        for k, c in enumerate(clutter_props):
            qpos += list(self._sample_clutter(k, c))
        key.qpos = qpos

        self.model = self.spec.compile()
        self.data = mujoco.MjData(self.model)
        self._widen_framebuffer()

        self.target_bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "target")
        self.rover_bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY,
                                           config.FRAME_BODY)
        self._qadr = {
            name: int(self.model.jnt_qposadr[
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)])
            for name in (*moss.BASE_JOINTS, *moss.JOINT_NAMES)
        }
        self.pool = RendererPool(self.model)
        self.reset()
        #: 参考帧 = 底盘在 home（base 关节全 0）时的 rover 位姿。
        #: 多视角必须并到这个帧里才能拼点云 —— 每帧各自用"当前 base"会让
        #: 不同视角的点云落在不同坐标系里，三角化的效果直接归零。
        self._ref_pos = self.data.xpos[self.rover_bid].copy()
        self._ref_mat = self.data.xmat[self.rover_bid].reshape(3, 3).copy()

    # ---------------------------------------------------------------- 摆位

    def _sample_target(self) -> np.ndarray:
        rng = self.rng
        x = rng.uniform(*self.cfg.x_range)
        y = rng.uniform(*self.cfg.y_range)
        hh = self.prop.half_height
        rr = self.prop.radius
        pose = self.cfg.target_pose
        if pose == "random":
            pose = "lying" if rng.random() < 0.5 else "upright"
        if pose == "lying":
            psi = rng.uniform(-math.pi, math.pi)
            axis = np.array([math.cos(psi), math.sin(psi), 0.0])
            pos = np.array([x, y, rr])           # 躺着时中心离地 = 半径
        else:
            psi = rng.uniform(-math.pi, math.pi)
            axis = np.array([0.0, 0.0, 1.0])
            pos = np.array([x, y, hh])
        q = _up_to(axis)
        self._target_pose_name = pose
        return np.concatenate([pos, q])

    def _sample_clutter(self, k: int, prop) -> np.ndarray:
        rng = self.rng
        for _ in range(40):
            x = POCKET[0] + rng.uniform(-0.05, 0.05) + 0.02 * k
            y = rng.uniform(-0.16, 0.16)
            if abs(y - float(self._target_xy[1])) > 0.075:
                break
        else:
            y = 0.16 * (1 if k % 2 else -1)
        hh = prop.half_height
        return np.array([x, y, hh, 1.0, 0.0, 0.0, 0.0])

    def _add_cameras(self) -> None:
        # 前视：rover 上的 D455，pitch 由 moss 的实测常量给出
        front_mount = np.array([math.cos(moss.CAMERA_PITCH_RAD / 2), 0.0,
                                math.sin(moss.CAMERA_PITCH_RAD / 2), 0.0])
        cam = self.spec.body(moss.BASE_BODY).add_camera()
        cam.name = config.CAM_FRONT
        cam.pos = list(moss.CAMERA_POS)
        cam.quat = list(mount_to_camera_quat(front_mount))
        cam.fovy = FRONT.intr.vfov_deg

        # 腕部：gripper_frame_link 上的 D405
        wrist_mount = moss.arm_camera_quat()
        cam = self.spec.body(moss.GRIPPER_FRAME_BODY).add_camera()
        cam.name = config.CAM_WRIST
        cam.pos = list(moss.ARM_CAMERA_POS)
        cam.quat = list(mount_to_camera_quat(wrist_mount))
        cam.fovy = WRIST.intr.vfov_deg

    def _widen_framebuffer(self) -> None:
        need_w = max(c.width for c in CAMERAS.values())
        need_h = max(c.height for c in CAMERAS.values())
        g = self.model.vis.global_
        g.offwidth = max(int(g.offwidth), need_w)
        g.offheight = max(int(g.offheight), need_h)

    # ---------------------------------------------------------------- 控制

    def reset(self) -> None:
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        mujoco.mj_forward(self.model, self.data)

    def set_base(self, y: float = 0.0, yaw: float = 0.0, x: float = 0.0) -> None:
        self.data.qpos[self._qadr["base_x"]] = x
        self.data.qpos[self._qadr["base_y"]] = y
        self.data.qpos[self._qadr["base_yaw"]] = yaw
        mujoco.mj_forward(self.model, self.data)

    def set_arm(self, deltas: np.ndarray | None = None) -> None:
        home = np.asarray(self.model.key_qpos[0], float)
        for i, name in enumerate(moss.JOINT_NAMES):
            adr = self._qadr[name]
            self.data.qpos[adr] = home[adr] + (0.0 if deltas is None else float(deltas[i]))
        mujoco.mj_forward(self.model, self.data)

    def base_pose(self) -> dict:
        return {"x": float(self.data.qpos[self._qadr["base_x"]]),
                "y": float(self.data.qpos[self._qadr["base_y"]]),
                "yaw": float(self.data.qpos[self._qadr["base_yaw"]])}

    def arm_pose(self) -> np.ndarray:
        return np.array([float(self.data.qpos[self._qadr[n]])
                         for n in moss.JOINT_NAMES])

    # ---------------------------------------------------------------- 真值

    def gt(self) -> GroundTruth:
        R_base = self._ref_mat
        base_pos = self._ref_pos
        pos_w = self.data.xpos[self.target_bid]
        R_t = self.data.xmat[self.target_bid].reshape(3, 3)
        pos = R_base.T @ (pos_w - base_pos)
        axis = R_base.T @ R_t[:, 2]
        axis = axis / max(np.linalg.norm(axis), 1e-9)
        return GroundTruth(label=self.cfg.prop, pos=pos, axis=axis,
                           radius=self.prop.radius,
                           half_height=self.prop.half_height,
                           shape=self.prop.shape)

    # ---------------------------------------------------------------- 渲染

    def viewpoint_plan(self) -> list[dict]:
        """视角计划：3 个前视（底盘沿 y/yaw 平移）+ 1 个腕部（底盘回零）。"""
        plan: list[dict] = []
        variants = list(self.cfg.base_variants)
        for view in self.cfg.views:
            if view == "front":
                k = len([p for p in plan if p["view"] == "front"])
                x, y, yaw = variants[k % len(variants)]
                plan.append({"view": "front", "x": x, "y": y, "yaw": yaw,
                             "arm": None})
            else:
                if not self.cfg.render_wrist:
                    continue
                # 腕部保持 home —— 这就是真正的抓取视角，动它反而会丢掉目标
                plan.append({"view": "wrist", "y": 0.0, "yaw": 0.0, "arm": None})
        return plan

    def render_view(self, spec: dict, depth: bool = True) -> ViewFrame:
        cam_spec = CAMERAS[spec["view"]]
        self.set_base(y=spec.get("y", 0.0), yaw=spec.get("yaw", 0.0),
                      x=spec.get("x", 0.0))
        self.set_arm(spec.get("arm"))
        cid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, cam_spec.name)
        if cid < 0:
            raise RuntimeError(f"scene has no camera {cam_spec.name!r}")
        rgb = self.pool.render(self.data, cam_spec.name, cam_spec.width,
                               cam_spec.height, depth=False)
        if depth:
            dep = self.pool.render(self.data, cam_spec.name, cam_spec.width,
                                   cam_spec.height, depth=True).astype(np.float32)
        else:
            dep = np.zeros((cam_spec.height, cam_spec.width), np.float32)
        base_pos = self._ref_pos
        base_mat = self._ref_mat
        cam_pos = self.data.cam_xpos[cid].copy()
        cam_mat = self.data.cam_xmat[cid].reshape(3, 3).copy()
        return ViewFrame(view=spec["view"], rgb=rgb, depth=dep,
                         intr=cam_spec.intr,
                         T_base_cam=matrix_to_base(cam_pos, cam_mat, base_pos, base_mat),
                         cam_xpos=cam_pos, cam_xmat=cam_mat, t=float(self.data.time),
                         base_pose=self.base_pose(), arm_pose=self.arm_pose())

    def render_plan(self, depth: bool = True) -> tuple[list[ViewFrame], GroundTruth]:
        frames = []
        for spec in self.viewpoint_plan():
            frames.append(self.render_view(spec, depth=depth))
        self.set_base(0.0, 0.0)
        self.set_arm(None)
        return frames, self.gt()

    def close(self) -> None:
        self.pool.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


def build(seed: int = 0, **kw) -> MossScene:
    return MossScene(SceneConfig(seed=seed, **kw))
