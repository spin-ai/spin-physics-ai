"""验证 MuJoCo 的相机约定 —— 这是整个几何链的地基，值得单独跑一次。

    MUJOCO_GL=osmesa /opt/venv/bin/python tools/verify_projection.py

断言四件事：
  1. 相机看向自己的 -Z（不是 +Z）；
  2. +X 是图像右、+Y 是图像上；
  3. fx = fy = 0.5*H / tan(fovy/2)，主点在图像中心；
  4. 深度缓冲是**米**，不是 [0,1] 归一化。
"""

from __future__ import annotations

import math

import mujoco
import numpy as np


def render(model, data, name, w, h, depth=False):
    r = mujoco.Renderer(model, h, w)
    try:
        if depth:
            r.enable_depth_rendering()
        r.update_scene(data, camera=name)
        return np.array(r.render(), copy=True)
    finally:
        r.close()


def main() -> int:
    ok = True
    xml = """<mujoco><worldbody>
      <light pos="0 0 3" dir="0 0 -1"/>
      <geom name="ball" type="sphere" size=".03" pos="0.2 0 0" rgba="1 0 0 1"/>
      <camera name="c" pos="0 0 1" fovy="60"/>
    </worldbody></mujoco>"""
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    W, H = 160, 120
    img = render(model, data, "c", W, H)
    red = (img[:, :, 0].astype(int) > 100) & (img[:, :, 1].astype(int) < 60)
    ys, xs = np.nonzero(red)
    cid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "c")
    R = data.cam_xmat[cid].reshape(3, 3)
    t = data.cam_xpos[cid]
    fy = 0.5 * H / math.tan(math.radians(model.cam_fovy[cid]) / 2)
    p = np.array([0.2, 0.0, 0.0])
    pc = R.T @ (p - t)
    u = W / 2 + fy * pc[0] / (-pc[2])
    v = H / 2 - fy * pc[1] / (-pc[2])

    print(f"实测红色像素质心: ({xs.mean():.1f}, {ys.mean():.1f})  n={len(xs)}")
    print(f"-Z 光轴假设预测: ({u:.1f}, {v:.1f})")
    e1 = abs(xs.mean() - u) < 2.0 and abs(ys.mean() - v) < 2.0
    print(f"[1,2,3] 光轴 -Z / +X 右 / +Y 上 / 方形像素: {'PASS' if e1 else 'FAIL'}")
    ok &= e1

    # 深度：相机在 z=+1，把球放到 z=-0.5 → 沿光轴 1.5 m，减去半径 0.03
    data.qpos[:] = 0
    mujoco.mj_forward(model, data)
    model.geom_pos[0] = [0.0, 0.0, -0.5]
    mujoco.mj_forward(model, data)
    dep = render(model, data, "c", W, H, depth=True)
    d = float(dep[H // 2, W // 2])
    want = 1.47
    e2 = abs(d - want) < 0.01
    print(f"[4] 中心深度 {d:.4f} m (期望 ~{want}) → {'PASS' if e2 else 'FAIL'}")
    ok &= e2

    print("\nPROJECTION", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
