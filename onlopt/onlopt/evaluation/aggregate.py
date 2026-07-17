"""複数シード実行結果の集計(平均・標準偏差・bootstrap 信頼区間)。"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np

from onlopt.core.simulator import RunResult
from onlopt.utils.rng import make_rng

AggMethod = Literal["std", "bootstrap"]


@dataclass
class AggregateResult:
    """ラウンドごとの平均リグレットと帯(std または bootstrap CI)。"""

    mean: np.ndarray  # (T,)
    lower: np.ndarray  # (T,)
    upper: np.ndarray  # (T,)
    n_runs: int
    method: str


def aggregate_curves(
    curves: np.ndarray,
    method: AggMethod = "std",
    confidence: float = 0.95,
    n_boot: int = 1000,
    boot_seed: int = 0,
) -> AggregateResult:
    """曲線の束 (n_runs, T) を平均と帯に集計する。"""
    curves = np.asarray(curves, dtype=np.float64)
    if curves.ndim != 2:
        raise ValueError("curves must be 2-dimensional (n_runs, T)")
    n = curves.shape[0]
    mean = curves.mean(axis=0)
    if method == "std":
        std = curves.std(axis=0, ddof=1) if n > 1 else np.zeros_like(mean)
        return AggregateResult(mean, mean - std, mean + std, n, "std")
    if method == "bootstrap":
        rng = make_rng(boot_seed)
        idx = rng.integers(0, n, size=(n_boot, n))
        boot_means = curves[idx].mean(axis=1)  # (n_boot, T)
        alpha = (1.0 - confidence) / 2.0
        lower = np.quantile(boot_means, alpha, axis=0)
        upper = np.quantile(boot_means, 1.0 - alpha, axis=0)
        return AggregateResult(mean, lower, upper, n, f"bootstrap{confidence:g}")
    raise ValueError(f"unknown aggregation method: {method}")


def aggregate_regret(
    results: Sequence[RunResult],
    key: Literal["cum_regret", "cum_pseudo_regret"] = "cum_regret",
    method: AggMethod = "std",
    confidence: float = 0.95,
    n_boot: int = 1000,
) -> AggregateResult:
    """複数シードの RunResult からリグレット曲線を集計する。"""
    if not results:
        raise ValueError("results is empty")
    curves = []
    for r in results:
        curve = getattr(r, key)
        if curve is None:
            raise ValueError(f"{key} is not available in RunResult(seed={r.seed})")
        curves.append(curve)
    return aggregate_curves(
        np.stack(curves), method=method, confidence=confidence, n_boot=n_boot
    )
