"""纯 numpy 的几何测试 —— 不需要渲染、不需要 MuJoCo。

    PYTHONPATH=/data/vision /opt/venv/bin/python -m unittest discover -s tests -v

注：几何本身只需 numpy，但 `mvision.geometry` 在 import 时会连 `mvision.scene`
（要 MOSS 资产）。裸克隆没有资产时整文件跳过，不会报 ERROR。
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
    from mvision import geometry
except Exception as exc:  # noqa: BLE001
    _skip_module(f"需要 mvision 运行环境（MOSS 资产 / 容器）: {exc}")


def cylinder_points(radius=0.033, half_height=0.0575, axis=(1, 0, 0),
                    centre=(0.26, 0.0, 0.033), n=4000, noise=0.0008, seed=0):
    """合成一个圆柱侧表面点云（不含端面，模拟"只看到侧面"的最坏情况）。"""
    rng = np.random.default_rng(seed)
    axis = np.asarray(axis, float)
    axis /= np.linalg.norm(axis)
    tmp = np.array([0.0, 0.0, 1.0]) if abs(axis[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    e1 = np.cross(axis, tmp)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(axis, e1)
    theta = rng.uniform(0, 2 * math.pi, n)
    t = rng.uniform(-half_height, half_height, n)
    pts = (radius * (np.cos(theta)[:, None] * e1 + np.sin(theta)[:, None] * e2)
           + np.outer(t, axis) + np.asarray(centre, float))
    return pts + rng.normal(0, noise, pts.shape)


class TestAxis(unittest.TestCase):
    def test_horizontal_cylinder_axis(self):
        gt_axis = np.array([1.0, 0.0, 0.0])
        pts = cylinder_points(axis=gt_axis)
        c = geometry.cluster_from_points(pts, shape="cylinder")
        err = geometry.degrees_between(c.axis, gt_axis)
        self.assertLess(err, 5.0, f"axis error {err:.2f} deg")
        self.assertLess(c.upright, 0.1)
        self.assertIsNotNone(c.yaw_double)

    def test_tilted_cylinder_axis(self):
        gt_axis = np.array([0.5, 0.5, math.sqrt(0.5)])
        pts = cylinder_points(axis=gt_axis)
        c = geometry.cluster_from_points(pts, shape="cylinder")
        self.assertLess(geometry.degrees_between(c.axis, gt_axis), 6.0)

    def test_upright_cylinder_has_no_yaw(self):
        pts = cylinder_points(axis=(0, 0, 1))
        c = geometry.cluster_from_points(pts, shape="cylinder")
        self.assertGreater(c.upright, 0.95)
        self.assertIsNone(c.yaw_double)

    def test_centroid_and_radius(self):
        pts = cylinder_points(centre=(0.25, -0.05, 0.033))
        c = geometry.cluster_from_points(pts, shape="cylinder")
        self.assertLess(np.linalg.norm(c.centroid - np.array([0.25, -0.05, 0.033])) * 1000, 3.0)
        self.assertAlmostEqual(c.radius, 0.033, delta=0.004)
        # 顶高 = 中心 z + 半径（躺着的圆柱）
        self.assertAlmostEqual(c.top_z, 0.066, delta=0.006)


class TestPlane(unittest.TestCase):
    def test_ransac_finds_floor(self):
        rng = np.random.default_rng(1)
        floor = np.column_stack([rng.uniform(-0.3, 0.3, 3000),
                                 rng.uniform(-0.3, 0.3, 3000),
                                 np.full(3000, 0.0)])
        floor = floor + rng.normal(0, 0.0005, floor.shape)
        obj = cylinder_points(centre=(0.1, 0.0, 0.033), n=800)
        pts = np.concatenate([floor, obj])
        n, d, inl = geometry.ransac_plane(pts)
        self.assertGreater(abs(n[2]), 0.95)
        self.assertGreater(inl.sum(), 2500)
        self.assertLess(inl.sum(), 3500)


class TestClustering(unittest.TestCase):
    def test_two_objects_split(self):
        a = cylinder_points(centre=(0.10, 0.0, 0.033), n=1200, seed=3)
        b = cylinder_points(centre=(0.28, 0.0, 0.033), axis=(0, 1, 0), n=1200, seed=4)
        clusters = geometry.cluster_points(np.concatenate([a, b]), voxel=0.010,
                                          min_points=100)
        self.assertEqual(len(clusters), 2)
        centres = sorted(c.mean(axis=0)[0] for c in clusters)
        self.assertAlmostEqual(centres[0], 0.10, delta=0.01)
        self.assertAlmostEqual(centres[1], 0.28, delta=0.01)

    def test_voxel_downsample_shrinks(self):
        pts = cylinder_points(n=5000)
        small = geometry.voxel_downsample(pts, voxel=0.005)
        self.assertLess(len(small), len(pts))
        self.assertGreater(len(small), 100)


if __name__ == "__main__":
    unittest.main()
