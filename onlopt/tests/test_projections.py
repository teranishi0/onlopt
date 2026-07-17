"""射影の正当性テスト(KKT 条件の数値確認)。"""

import numpy as np
import pytest

from onlopt.geometry import (
    project_l2_ball,
    project_product_simplex,
    project_simplex,
)
from onlopt.utils import make_rng


class TestProjectSimplex:
    def test_feasibility(self):
        rng = make_rng(0)
        for _ in range(50):
            v = rng.normal(size=rng.integers(2, 30)) * 10
            x = project_simplex(v)
            assert np.all(x >= 0)
            assert np.isclose(x.sum(), 1.0)

    def test_idempotent(self):
        rng = make_rng(1)
        v = rng.dirichlet(np.ones(10))
        assert np.allclose(project_simplex(v), v)

    def test_kkt_optimality(self):
        """任意の実行可能点より射影点の方が近い(変分不等式の数値確認)。"""
        rng = make_rng(2)
        for _ in range(20):
            v = rng.normal(size=8) * 5
            x = project_simplex(v)
            for _ in range(50):
                y = rng.dirichlet(np.ones(8))
                # <v - x, y - x> <= 0 が射影の特徴付け
                assert (v - x) @ (y - x) <= 1e-9

    def test_known_case(self):
        x = project_simplex(np.array([1.0, 0.0]))
        assert np.allclose(x, [1.0, 0.0])
        x = project_simplex(np.array([0.5, 0.5, -100.0]))
        assert np.allclose(x, [0.5, 0.5, 0.0])

    def test_invalid_z(self):
        with pytest.raises(ValueError):
            project_simplex(np.ones(3), z=0.0)


class TestProjectL2Ball:
    def test_inside_unchanged(self):
        v = np.array([0.1, 0.2])
        assert np.allclose(project_l2_ball(v, radius=1.0), v)

    def test_outside_on_boundary(self):
        rng = make_rng(3)
        for _ in range(20):
            v = rng.normal(size=5) * 10
            c = rng.normal(size=5)
            r = float(rng.uniform(0.5, 2.0))
            x = project_l2_ball(v, radius=r, center=c)
            if np.linalg.norm(v - c) > r:
                assert np.isclose(np.linalg.norm(x - c), r)
                # 射影点は中心と元の点を結ぶ直線上にある
                d1 = (v - c) / np.linalg.norm(v - c)
                d2 = (x - c) / np.linalg.norm(x - c)
                assert np.allclose(d1, d2)


class TestProjectProductSimplex:
    def test_blocks(self):
        rng = make_rng(4)
        v = rng.normal(size=7)
        x = project_product_simplex(v, [3, 4])
        assert np.isclose(x[:3].sum(), 1.0)
        assert np.isclose(x[3:].sum(), 1.0)
        assert np.all(x >= 0)
        assert np.allclose(x[:3], project_simplex(v[:3]))

    def test_size_mismatch(self):
        with pytest.raises(ValueError):
            project_product_simplex(np.ones(5), [2, 2])
