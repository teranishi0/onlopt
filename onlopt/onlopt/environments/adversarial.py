"""敵対的環境(固定的/適応的)。"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from onlopt.core.environment import Environment, History

# 非適応的敵対者の損失生成規則: (t, rng) -> 損失ベクトル
ObliviousLossFn = Callable[[int, np.random.Generator], np.ndarray]
# 適応的敵対者の戦略: (t, history, rng) -> 損失ベクトル
AdaptiveStrategy = Callable[[int, History | None, np.random.Generator], np.ndarray]


class ObliviousAdversary(Environment):
    """学習者の行動を参照しない(oblivious)敵対者。

    損失系列の生成規則を callable で与える。O(√T) 挙動の確認に使う。
    """

    def __init__(self, n_actions: int, loss_fn: ObliviousLossFn) -> None:
        self.n_actions = n_actions
        self._loss_fn = loss_fn

    @classmethod
    def from_matrix(cls, loss_matrix: np.ndarray) -> ObliviousAdversary:
        """固定の損失行列 (T, K) から系列を再生する敵対者。

        T を超えたラウンドは先頭から巡回する。
        """
        m = np.asarray(loss_matrix, dtype=np.float64)
        if m.ndim != 2:
            raise ValueError("loss_matrix must be 2-dimensional (T, K)")

        def fn(t: int, rng: np.random.Generator) -> np.ndarray:
            return m[t % m.shape[0]]

        return cls(n_actions=m.shape[1], loss_fn=fn)

    def get_loss(
        self, t: int, history: History | None, rng: np.random.Generator
    ) -> np.ndarray:
        return np.asarray(self._loss_fn(t, rng), dtype=np.float64)


class AdaptiveAdversary(Environment):
    """学習者の行動履歴を参照する適応的敵対者。"""

    def __init__(self, n_actions: int, strategy: AdaptiveStrategy) -> None:
        self.n_actions = n_actions
        self._strategy = strategy

    def get_loss(
        self, t: int, history: History | None, rng: np.random.Generator
    ) -> np.ndarray:
        return np.asarray(self._strategy(t, history, rng), dtype=np.float64)
