"""契约测试：感知 → obs 槽位。

不需要渲染就能跑，但需要 microduck_local 的源码在 sys.path 上（槽位下标从
`robots/moss.py` 读，绝不在这里抄一份 —— 抄一份就是第二个真相来源）。
"""

from __future__ import annotations

import math
import sys
import unittest

import numpy as np


def _skip_module(reason: str) -> None:  # pragma: no cover - 裸克隆走这里
    """pytest / unittest 两种跑法都能优雅跳过（而不是 ERROR）。"""
    if "pytest" in sys.modules:
        import pytest

        pytest.skip(reason, allow_module_level=True)
    raise unittest.SkipTest(reason)


try:
    from mvision import adapters, config
except Exception as exc:  # noqa: BLE001 —— 裸克隆缺 MOSS 资产时直接跳过整个文件
    _skip_module(f"需要 mvision 运行环境（MOSS 资产 / 容器）: {exc}")


def _has_microduck_local() -> bool:
    """MOSS 32 维槽位下标只能从上游读，所以这两个用例必须在容器里跑。"""
    try:
        import microduck_local.robots.moss  # noqa: F401
    except Exception:  # noqa: BLE001
        return False
    return True


HAS_MICRODUCK_LOCAL = _has_microduck_local()


def perception(**kw):
    base = dict(t=1.0, label="can", pos=np.array([0.26, 0.0, 0.033]),
                axis=np.array([1.0, 0.0, 0.0]), upright=0.0, top_z=0.066,
                radius=0.033, conf=1.0)
    base.update(kw)
    return adapters.Perception(**base)


class TestSlots(unittest.TestCase):
    def test_fresh_fix_fills_slots(self):
        s = adapters.moss_slots(perception(), now=1.0)
        self.assertEqual(s["target_seen"], 1.0)
        np.testing.assert_allclose(s["target_base"], [0.26, 0.0, 0.033])
        self.assertAlmostEqual(s["target_upright"], 0.0)
        self.assertAlmostEqual(s["spare"], 0.066)
        # 倍角：轴沿 +x → ψ=0 → (cos0, sin0) = (1, 0)
        np.testing.assert_allclose(s["target_axis"], [1.0, 0.0], atol=1e-9)

    def test_double_angle_is_pi_periodic(self):
        """轴翻转 180° 是同一个抓取，所以倍角必须相等。"""
        a = adapters.moss_slots(perception(axis=np.array([1.0, 0.0, 0.0])), now=1.0)
        b = adapters.moss_slots(perception(axis=np.array([-1.0, 0.0, 0.0])), now=1.0)
        np.testing.assert_allclose(a["target_axis"], b["target_axis"], atol=1e-9)

    def test_upright_object_has_no_axis(self):
        p = perception(axis=np.array([0.0, 0.0, 1.0]), upright=1.0)
        s = adapters.moss_slots(p, now=1.0)
        np.testing.assert_allclose(s["target_axis"], [0.0, 0.0])

    def test_stale_fix_clears_everything(self):
        s = adapters.moss_slots(perception(t=0.0), now=10.0, stale_s=0.25)
        self.assertEqual(s["target_seen"], 0.0)
        np.testing.assert_allclose(s["target_base"], np.zeros(3))
        self.assertAlmostEqual(s["spare"], 0.0)

    def test_missing_detection_is_stale(self):
        s = adapters.moss_slots(perception(pos=None), now=1.0)
        self.assertEqual(s["target_seen"], 0.0)

    def test_task_switches_zero_the_reserved_block(self):
        """`approach`/`stow` 用的是恒零的保留区，不能被我们填。"""
        s = adapters.moss_slots(perception(), now=1.0, publish_axis=False,
                                publish_top=False, publish_upright=False)
        np.testing.assert_allclose(s["target_axis"], np.zeros(2))
        self.assertEqual(s["target_upright"], 0.0)
        self.assertEqual(s["spare"], 0.0)
        # 但位置和新鲜度仍然要写 —— 那是 24:27/27，本来就有
        np.testing.assert_allclose(s["target_base"], [0.26, 0.0, 0.033])
        self.assertEqual(s["target_seen"], 1.0)


class TestMossObs(unittest.TestCase):
    @unittest.skipUnless(HAS_MICRODUCK_LOCAL,
                         "需要容器内的 microduck_local（robots/moss.py 提供槽位下标）")
    def test_write_into_32dim_obs(self):
        import microduck_local.robots.moss as moss
        obs = np.zeros(moss.OBS_DIM, np.float32)
        s = adapters.moss_slots(perception(), now=1.0)
        out = adapters.apply_to_moss_obs(obs, s)
        np.testing.assert_allclose(out[moss.OBS_TARGET_BASE], [0.26, 0.0, 0.033])
        self.assertEqual(out[moss.OBS_TARGET_SEEN].item(), 1.0)
        np.testing.assert_allclose(out[moss.OBS_TARGET_AXIS], [1.0, 0.0])
        self.assertAlmostEqual(out[moss.OBS_TARGET_UPRIGHT].item(), 0.0)
        self.assertAlmostEqual(out[moss.OBS_SPARE].item(), 0.066)
        # 前面的关节块不能被碰
        np.testing.assert_allclose(out[moss.OBS_JOINT_POS], 0.0)

    @unittest.skipUnless(HAS_MICRODUCK_LOCAL,
                         "需要容器内的 microduck_local（robots/moss.py 提供槽位下标）")
    def test_wrong_dim_raises(self):
        with self.assertRaises(ValueError):
            adapters.apply_to_moss_obs(np.zeros(61, np.float32), {})


class TestDuckPayload(unittest.TestCase):
    def test_bearing_sign_and_magnitude(self):
        """目标在安装帧左前方 45°、同一高度 → bearing=+45°（detector 约定左为正）。"""
        # 安装帧 x 前 / y 左 / z 上。目标在左前方 45°（相对 x 偏 +y）
        p = perception(pos=np.array([1.0, 1.0, 1.0]))
        # MuJoCo 相机架：x 右、y 上、-z 光轴。安装帧 = (-z, -x, +y)。
        # 让光轴指向世界 +x：相机 z = -x，相机 x(右) = -y，相机 y(上) = +z
        cam_mat = np.column_stack([np.array([0, -1.0, 0]),
                                   np.array([0, 0, 1.0]),
                                   np.array([-1.0, 0, 0])])
        payload = adapters.duck_payload(p, np.array([0.0, 0.0, 1.0]), cam_mat,
                                        radius=0.05)
        self.assertAlmostEqual(payload["bearing"], math.pi / 4, places=5)
        self.assertAlmostEqual(payload["elevation"], 0.0, places=5)
        self.assertAlmostEqual(payload["range"], math.sqrt(2.0), places=5)
        self.assertAlmostEqual(payload["width"],
                               2 * math.atan2(0.05, math.sqrt(2.0)), places=6)

    def test_turn_command_deadband(self):
        self.assertEqual(adapters.duck_bearing_command({"bearing": 0.01}), 0.0)
        self.assertGreater(adapters.duck_bearing_command({"bearing": 0.5}), 0.0)
        self.assertLess(adapters.duck_bearing_command({"bearing": -0.5}), 0.0)


class TestFreshness(unittest.TestCase):
    def test_stale_window(self):
        p = perception(t=1.0)
        self.assertFalse(p.is_stale(1.1, limit=config.STALE_S))
        self.assertTrue(p.is_stale(1.0 + config.STALE_S + 0.01))


if __name__ == "__main__":
    unittest.main()
