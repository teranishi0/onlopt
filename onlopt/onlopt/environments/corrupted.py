"""汚染付き確率的環境。BOBW の中間レジーム検証用。

ベースの確率的環境の損失を、予算 C の範囲内で敵対的に改変する。
汚染量はラウンドごとの max ノルム ||corrupted - clean||_inf で計上し、
実際に消費した総量を記録して RunResult.config に含める。
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from onlopt.core.environment import Environment, History
from onlopt.environments.stochastic import StochasticEnv

# 汚染戦略: (t, clean_loss, mean_loss, remaining_budget, rng) -> 汚染後の損失
AttackFn = Callable[
    [int, np.ndarray, np.ndarray | None, float, np.random.Generator], np.ndarray
]


def flip_best_arm_attack(
    t: int,
    clean: np.ndarray,
    means: np.ndarray | None,
    remaining: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """既定の汚染戦略: 最良腕の損失を 1、それ以外を 0 に反転する。"""
    corrupted = np.zeros_like(clean)
    best = int(np.argmin(means)) if means is not None else 0
    corrupted[best] = 1.0
    return corrupted


class CorruptedStochasticEnv(Environment):
    def __init__(
        self,
        base: StochasticEnv,
        budget: float,
        attack: AttackFn = flip_best_arm_attack,
    ) -> None:
        if budget < 0:
            raise ValueError("budget must be non-negative")
        self.base = base
        self.budget = budget
        self.attack = attack
        self.n_actions = base.n_actions
        self._used = 0.0

    def reset(self) -> None:
        self.base.reset()
        self._used = 0.0

    def get_loss(
        self, t: int, history: History | None, rng: np.random.Generator
    ) -> np.ndarray:
        clean = self.base.get_loss(t, history, rng)
        remaining = self.budget - self._used
        if remaining <= 0.0:
            return clean
        corrupted = np.asarray(
            self.attack(t, clean, self.base.mean_loss, remaining, rng),
            dtype=np.float64,
        )
        cost = float(np.abs(corrupted - clean).max())
        if cost <= remaining:
            self._used += cost
            return corrupted
        return clean

    @property
    def mean_loss(self) -> np.ndarray | None:
        # 擬似リグレットはベース環境の期待損失に対して定義する
        return self.base.mean_loss

    @property
    def corruption_used(self) -> float:
        return self._used

    def run_metadata(self) -> dict[str, object]:
        return {
            "corruption_budget": self.budget,
            "corruption_used": self._used,
        }

    def config(self) -> dict[str, object]:
        return {
            "class": type(self).__name__,
            "base": self.base.config(),
            "budget": self.budget,
        }
