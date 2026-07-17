"""乱数生成器の一元管理。

モジュール全体で ``numpy.random.Generator`` のみを使用し、
グローバル乱数状態(``np.random.seed`` 等)には一切依存しない。
"""

from __future__ import annotations

import numpy as np


def make_rng(seed: int) -> np.random.Generator:
    """シードから独立した乱数生成器を作る。"""
    return np.random.default_rng(seed)


def spawn_rngs(seed: int, n: int) -> list[np.random.Generator]:
    """1つの親シードから統計的に独立な n 個の生成器を派生させる。"""
    seq = np.random.SeedSequence(seed)
    return [np.random.default_rng(child) for child in seq.spawn(n)]
