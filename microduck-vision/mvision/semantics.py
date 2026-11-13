"""语义后端：给每张照片一个"目标在哪里"的掩码。

三个实现，同一个接口（`mask(frame) -> (H,W) bool | None`）：

    oracle  用 GT 投影出掩码 —— 不是方法，是**上限与自检**：如果 oracle 也差很多，
            那问题在几何不在语义。
    color   橙/非橙阈值。目标物体在 scene 里被显式染成橙色，干扰物是灰/蓝/绿。
            这是不联网、不训练的诚实基线。
    http    外部 VLM（OpenAI 兼容 /chat/completions）。把它写成 bbox 而不是像素
            掩码 —— 要求模型输出坐标比输出"哪一张里"错得更少。

`None` 表示"这一帧没有语义约束"，几何会退化成"取最大簇"。
"""

from __future__ import annotations

import base64
import io
import json
import re
import urllib.error
import urllib.request

import numpy as np

from . import config
from .scene import GroundTruth, ViewFrame


class Backend:
    name = "base"

    def mask(self, frame: ViewFrame, instruction: str = "") -> np.ndarray | None:
        raise NotImplementedError

    def to_dict(self) -> dict:
        return {"backend": self.name}


# --------------------------------------------------------------------------- #


class Oracle(Backend):
    """GT 投影掩码 —— 上限参考。"""

    name = "oracle"

    def __init__(self, gt: GroundTruth):
        self.gt = gt

    def mask(self, frame: ViewFrame, instruction: str = "") -> np.ndarray:
        """3D 判据：把每个有效像素反投影回 rover 系，落在目标包围球里的就是它。

        不用"投影圆"：罐子的半径 33 mm、半高 57.5 mm，用 hypot(两者)=66 mm 的
        圆去框，会把周围一大片地板算进来（实测这让 oracle 变成 30 mm 误差，
        比 color 还差）。3D 判据没有这个问题，而且它就是真值的定义。
        """
        from .camera import unproject
        pcam, valid = unproject(frame.depth, frame.intr, max_range=config.MAX_RANGE_M)
        m = np.zeros_like(valid)
        if len(pcam) == 0:
            return m
        R = frame.T_base_cam[:3, :3]
        t = frame.T_base_cam[:3, 3]
        pts = pcam @ R.T + t
        r_big = float(np.hypot(self.gt.radius, self.gt.half_height)) + 0.004
        inside = np.linalg.norm(pts - self.gt.pos, axis=1) < r_big
        vs, us = np.nonzero(valid)
        m[vs[inside], us[inside]] = True
        return m


class Color(Backend):
    """品红阈值。目标 = (0.80, 0.10, 0.75)：红和蓝都压过绿，在 Lambert 光照下
    通道比例基本保留。选品红是因为 MOSS 自己的身体是橙红 —— 用橙色会把自己的
    机械臂分割成目标（实测过）。"""

    name = "color"

    def __init__(self, min_level: float = 0.16, g_ratio: float = 1.45,
                 min_px: int = 40):
        self.min_level = min_level
        self.g_ratio = g_ratio
        self.min_px = min_px

    def mask(self, frame: ViewFrame, instruction: str = "") -> np.ndarray | None:
        rgb = frame.rgb.astype(np.float32) / 255.0
        r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
        m = ((np.minimum(r, b) > self.min_level)
             & (r > self.g_ratio * g) & (b > self.g_ratio * g))
        if m.sum() < self.min_px:
            return None
        return m


class Http(Backend):
    """外部 VLM：返回归一化 bbox → 掩码。失败一律回落 None（几何接管）。"""

    name = "http"

    PROMPT = (
        "You are a robot's perception module. In the image, find {target}. "
        "Reply ONLY with JSON: {{\"found\": true|false, "
        "\"bbox\": [x0,y0,x1,y1]}} with coordinates normalized to 0..1 "
        "relative to the image, origin at top-left. No prose."
    )

    def __init__(self, base: str | None = None, model: str | None = None,
                 key: str | None = None, timeout: float | None = None,
                 image_max: int = 512):
        self.base = (base or config.HTTP_BASE).rstrip("/")
        self.model = model or config.HTTP_MODEL
        self.key = key or config.HTTP_KEY
        self.timeout = timeout or config.HTTP_TIMEOUT
        self.image_max = image_max
        self.last_error: str | None = None

    # ---------------------------------------------------------------- 请求

    def _encode(self, rgb: np.ndarray) -> str:
        from PIL import Image
        img = Image.fromarray(rgb.astype(np.uint8))
        if max(img.size) > self.image_max:
            s = self.image_max / max(img.size)
            img = img.resize((max(1, int(img.width * s)), max(1, int(img.height * s))))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=88)
        return base64.b64encode(buf.getvalue()).decode()

    def _ask(self, frame: ViewFrame, instruction: str) -> dict | None:
        if not self.base or not self.key:
            self.last_error = "MV_HTTP_BASE / MV_HTTP_KEY not set"
            return None
        target = instruction or "the orange cylinder"
        payload = {
            "model": self.model,
            "temperature": 0,
            "max_tokens": 120,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": self.PROMPT.format(target=target)},
                    {"type": "image_url", "image_url": {
                        "url": f"data:image/jpeg;base64,{self._encode(frame.rgb)}"}},
                ],
            }],
        }
        req = urllib.request.Request(
            f"{self.base}/chat/completions",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {self.key}"},
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                body = json.loads(resp.read().decode())
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
            self.last_error = f"{type(e).__name__}: {e}"
            return None
        try:
            text = body["choices"][0]["message"]["content"]
        except (KeyError, IndexError):
            self.last_error = f"unexpected body: {str(body)[:200]}"
            return None
        return _parse_json(text)

    # ---------------------------------------------------------------- 掩码

    def mask(self, frame: ViewFrame, instruction: str = "") -> np.ndarray | None:
        out = self._ask(frame, instruction)
        if not out or not out.get("found", False):
            return None
        bbox = out.get("bbox")
        if not (isinstance(bbox, (list, tuple)) and len(bbox) == 4):
            return None
        h, w = frame.rgb.shape[:2]
        x0, y0, x1, y1 = [float(v) for v in bbox]
        if max(x0, y0, x1, y1) <= 1.001:            # 归一化坐标
            x0, x1 = x0 * w, x1 * w
            y0, y1 = y0 * h, y1 * h
        xa, xb = sorted((int(np.clip(x0, 0, w - 1)), int(np.clip(x1, 0, w - 1))))
        ya, yb = sorted((int(np.clip(y0, 0, h - 1)), int(np.clip(y1, 0, h - 1))))
        m = np.zeros((h, w), bool)
        if xb - xa < 3 or yb - ya < 3:
            return None
        m[ya:yb + 1, xa:xb + 1] = True
        return m

    def to_dict(self) -> dict:
        return {"backend": self.name, "model": self.model}


def _parse_json(text: str) -> dict | None:
    """模型很爱加 ```json 围栏或前后废话 —— 取第一个 {...}。"""
    if not text:
        return None
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


# --------------------------------------------------------------------------- #


def build(name: str, *, gt: GroundTruth | None = None, **kw) -> Backend | None:
    name = (name or "none").lower()
    if name in ("none", "off", "geometry"):
        return None
    if name == "oracle":
        if gt is None:
            raise ValueError("oracle backend needs the ground truth")
        return Oracle(gt)
    if name == "color":
        return Color(**kw)
    if name == "http":
        return Http(**kw)
    raise ValueError(f"unknown semantic backend {name!r} "
                     f"(have: none, oracle, color, http)")


def masks_for(backend: Backend | None, frames: list[ViewFrame],
              instruction: str = "") -> tuple[list[np.ndarray] | None, int]:
    """返回 (掩码列表, 命中帧数)。backend 为 None 时返回 (None, 0)。"""
    if backend is None:
        return None, 0
    masks = []
    hits = 0
    for f in frames:
        m = backend.mask(f, instruction)
        if m is not None:
            hits += 1
        masks.append(m)
    if hits == 0:
        return None, 0
    return masks, hits
