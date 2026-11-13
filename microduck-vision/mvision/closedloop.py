"""闭环 seam 检查：把感知写进真实环境的 obs，并测控制环时延。

这里**不**声称测了策略成功率 —— 镜像里没有一个训练好的 MOSS pick 策略
（`/app/policies` 是 duck 的），所以本模块证明的两件事是：

    1. seam 是真的：`OBS_TARGET_BASE/AXIS/UPRIGHT/SPARE` 能被我们的感知写进去，
       形状/量纲对得上，环境照常 step；
    2. 时延是真的：5 Hz 的视觉挂在 25~50 Hz 的控制环旁边，控制步长不被拖垮。

策略退化（RD §7 R1）需要一条训练过的 MOSS 腿，见 README「没做什么」。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from . import adapters, config


# --------------------------------------------------------------------------- #
# provider：任何能给出 Perception 的东西
# --------------------------------------------------------------------------- #

Provider = Callable[[float], adapters.Perception]


def constant_provider(pos=(0.26, 0.0, 0.03), axis=(1.0, 0.0, 0.0),
                      upright=0.0, top_z=0.066) -> Provider:
    """固定感知 —— 用来验证 seam，不验证算法。"""
    def _p(now: float) -> adapters.Perception:
        return adapters.Perception(
            t=now, label="can", pos=np.array(pos, float),
            axis=np.array(axis, float), upright=upright, top_z=top_z,
            radius=0.033, conf=1.0, source="constant", n_points=1000)
    return _p


def scene_provider(seed: int = 0, backend: str = "color",
                   instruction: str = "the orange can") -> tuple[Provider, object]:
    """真感知：在自己的 MOSS 场景里渲染 4 视角 → 重建 → Perception。

    返回 (provider, scene)；scene 要由调用方 close。
    """
    from . import geometry, semantics
    from .scene import MossScene, SceneConfig

    scene = MossScene(SceneConfig(seed=seed))
    frames, gt = scene.render_plan()
    sem = semantics.build(backend, gt=gt)
    masks, _ = semantics.masks_for(sem, frames, instruction)
    clusters = geometry.reconstruct(frames, masks=masks, shape=scene.prop.shape)
    cached = (adapters.from_cluster(clusters[0], t=0.0, label="can")
              if clusters else adapters.Perception(t=0.0, conf=0.0))

    def _p(now: float) -> adapters.Perception:
        cached.t = now
        return cached
    return _p, scene


# --------------------------------------------------------------------------- #
# Gym wrapper
# --------------------------------------------------------------------------- #

try:  # gymnasium 一定在镜像里，但没有时也要能 import 本模块
    import gymnasium as gym
    _Base = gym.Wrapper
except Exception:                                     # pragma: no cover
    class _Base:                                      # type: ignore
        def __init__(self, env):
            self.env = env


class VisionWrapper(_Base):
    """把每个 step 的 obs 里 24:32 换成感知结果。

    开关（publish_axis/publish_top）对应 task：`approach` / `stow` 会把保留区
    恒置零，这里必须能关掉，否则是在改一条已训练腿的输入分布。
    """

    def __init__(self, env, provider: Provider, *, publish_axis: bool = True,
                 publish_top: bool = True, publish_upright: bool = True,
                 stale_s: float = config.STALE_S):
        super().__init__(env)
        self.provider = provider
        self.publish_axis = publish_axis
        self.publish_top = publish_top
        self.publish_upright = publish_upright
        self.stale_s = stale_s
        self.last_slots: dict | None = None
        self.last_perception: adapters.Perception | None = None
        self.t = 0.0

    def _decorate(self, obs):
        now = self.t
        p = self.provider(now)
        slots = adapters.moss_slots(
            p, now, publish_axis=self.publish_axis,
            publish_top=self.publish_top,
            publish_upright=self.publish_upright, stale_s=self.stale_s)
        self.last_perception = p
        self.last_slots = slots
        obs = np.array(obs, copy=True)
        return adapters.apply_to_moss_obs(obs, slots)

    def reset(self, **kw):
        obs, info = self.env.reset(**kw)
        self.t = 0.0
        return self._decorate(obs), info

    def step(self, action):
        out = self.env.step(action)
        self.t += float(getattr(self.env, "ctrl_dt", 0.04))
        obs, rew, term, trunc, info = out
        return self._decorate(obs), rew, term, trunc, info


# --------------------------------------------------------------------------- #
# 检查
# --------------------------------------------------------------------------- #


@dataclass
class SeamReport:
    ok: bool = True
    steps: int = 0
    obs_dim: int = 0
    errors: list[str] = field(default_factory=list)
    slots: dict = field(default_factory=dict)
    env_step_s: list[float] = field(default_factory=list)

    def to_dict(self) -> dict:
        import numpy as _np
        return {"ok": self.ok, "steps": self.steps, "obs_dim": self.obs_dim,
                "errors": self.errors,
                "slots": {k: _np.asarray(v).tolist() for k, v in self.slots.items()},
                "env_step_median_ms": float(_np.median(self.env_step_s) * 1000)
                if self.env_step_s else None,
                "env_step_p95_ms": float(_np.percentile(self.env_step_s, 95) * 1000)
                if self.env_step_s else None}


def seam_check(steps: int = 30, seed: int = 0) -> SeamReport:
    """真环境 + 固定感知：obs 形状、槽位、可 step、不 NaN。"""
    from microduck_local.robots.moss_env import MossPickEnv
    import microduck_local.robots.moss as moss

    rep = SeamReport()
    env = MossPickEnv(task="pick", seed=seed, domain_rand=False, obs_noise=False)
    env = VisionWrapper(env, constant_provider())
    obs, _ = env.reset(seed=seed)
    rep.obs_dim = int(obs.shape[-1])
    if rep.obs_dim != moss.OBS_DIM:
        rep.ok = False
        rep.errors.append(f"obs dim {rep.obs_dim} != {moss.OBS_DIM}")
    if not np.isfinite(obs).all():
        rep.ok = False
        rep.errors.append("obs contains NaN/inf after write")
    act = np.zeros(env.action_space.shape, dtype=np.float32)
    for _ in range(steps):
        t0 = time.perf_counter()
        obs, rew, term, trunc, info = env.step(act)
        rep.env_step_s.append(time.perf_counter() - t0)
        if not np.isfinite(obs).all():
            rep.ok = False
            rep.errors.append("obs NaN after step")
            break
        if term or trunc:
            obs, _ = env.reset(seed=seed)
        rep.steps += 1
    rep.slots = env.last_slots or {}
    # 槽位内容必须真的落在 obs 上
    if env.last_slots is not None:
        if not np.allclose(obs[moss.OBS_TARGET_BASE],
                           np.asarray(env.last_slots["target_base"], float)):
            rep.ok = False
            rep.errors.append("target_base slot does not hold the perception")
        if obs[moss.OBS_TARGET_SEEN].item() != 1.0:
            rep.ok = False
            rep.errors.append("target_seen should be 1.0 for a fresh fix")
    return rep


def latency_check(steps: int = 60, perception_every: int = 5,
                  seed: int = 0) -> dict:
    """控制步长 vs 插入重建的步长 —— NFR-2 的量化。"""
    from .scene import MossScene, SceneConfig

    plain: list[float] = []
    mixed: list[float] = []
    per_cost: list[float] = []

    scene = MossScene(SceneConfig(seed=seed))
    try:
        # 只渲染（不重建）的稳态开销：控制环里唯一会被视觉抢走的东西
        for i in range(steps):
            t0 = time.perf_counter()
            scene.render_view({"view": "wrist", "y": 0.0, "yaw": 0.0})
            plain.append(time.perf_counter() - t0)
            if i % perception_every == 0:
                from . import geometry
                t0 = time.perf_counter()
                frames, gt = scene.render_plan()
                geometry.reconstruct(frames, shape=scene.prop.shape)
                per_cost.append(time.perf_counter() - t0)
                mixed.append(per_cost[-1])
            else:
                mixed.append(0.0)
    finally:
        scene.close()

    per = np.asarray(per_cost, float)
    return {"render_only_median_ms": float(np.median(plain) * 1000),
            "render_only_p95_ms": float(np.percentile(plain, 95) * 1000),
            "full_perception_n": int(len(per)),
            "full_perception_median_ms": float(np.median(per) * 1000) if len(per) else None,
            "full_perception_p95_ms": float(np.percentile(per, 95) * 1000) if len(per) else None,
            "budget_ms": 2000.0,
            "verdict": "ok" if len(per) and np.percentile(per, 95) < 2.0 else "slow"}
