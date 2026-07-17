"""決定集合への射影ユーティリティ。"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def project_simplex(v: np.ndarray, z: float = 1.0) -> np.ndarray:
    """確率単体 {x >= 0, sum(x) = z} への L2 射影(ソート法、O(K log K))。

    Held, Wolfe & Crowder (1974) / Duchi et al. (2008) のアルゴリズム。
    """
    if z <= 0:
        raise ValueError("z must be positive")
    v = np.asarray(v, dtype=np.float64)
    u = np.sort(v)[::-1]
    css = np.cumsum(u) - z
    ind = np.arange(1, v.size + 1)
    cond = u - css / ind > 0
    rho = int(ind[cond][-1])
    theta = css[rho - 1] / rho
    return np.maximum(v - theta, 0.0)


def project_l2_ball(
    v: np.ndarray, radius: float = 1.0, center: np.ndarray | None = None
) -> np.ndarray:
    """L2 球 {x : ||x - center|| <= radius} への射影。"""
    if radius <= 0:
        raise ValueError("radius must be positive")
    v = np.asarray(v, dtype=np.float64)
    c = np.zeros_like(v) if center is None else np.asarray(center, dtype=np.float64)
    d = v - c
    norm = float(np.linalg.norm(d))
    if norm <= radius:
        return v.copy()
    return c + d * (radius / norm)


def project_product_simplex(
    v: np.ndarray, block_sizes: Sequence[int]
) -> np.ndarray:
    """単体の直積への射影。

    v をブロックに分割し、各ブロックを確率単体へ射影して連結する。
    負荷分散向け決定集合(v0.1 では単体の直積まで。組合せ的集合は
    v0.2 で拡張)。
    """
    v = np.asarray(v, dtype=np.float64)
    if sum(block_sizes) != v.size:
        raise ValueError("block sizes must sum to len(v)")
    out = np.empty_like(v)
    start = 0
    for size in block_sizes:
        out[start : start + size] = project_simplex(v[start : start + size])
        start += size
    return out
