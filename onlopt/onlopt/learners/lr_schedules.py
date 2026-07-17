"""学習率スケジューラ。

``eta(t, state)`` の ``state`` は学習者(FTRL / OMD)が公開する
内部累積量の辞書であり、データ依存(適応型)スケジュールが参照する。
BOBW の学習率設計を差し替え実験できるよう、キーは学習者側で拡張してよい。

FTRL / OMD が提供する標準キー:
    - "cum_sq_est_norm": 損失推定ベクトルの二乗ノルムの累積 sum_s ||g_s||^2
    - "cum_est_sq":      成分ごとの二乗の累積の総和(同値だが将来拡張用)
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Mapping


class LRSchedule(ABC):
    @abstractmethod
    def eta(self, t: int, state: Mapping[str, float]) -> float:
        """ラウンド t(0 始まり)の学習率を返す。"""

    def config(self) -> dict[str, object]:
        return {"class": type(self).__name__}


class FixedLR(LRSchedule):
    """固定学習率 eta(t) = eta0。"""

    def __init__(self, eta0: float) -> None:
        if eta0 <= 0:
            raise ValueError("eta0 must be positive")
        self.eta0 = eta0

    def eta(self, t: int, state: Mapping[str, float]) -> float:
        return self.eta0

    def config(self) -> dict[str, object]:
        return {"class": type(self).__name__, "eta0": self.eta0}


class InverseSqrtLR(LRSchedule):
    """eta(t) = c / sqrt(t + 1)。Tsallis-INF の標準スケジュール。"""

    def __init__(self, c: float = 1.0) -> None:
        if c <= 0:
            raise ValueError("c must be positive")
        self.c = c

    def eta(self, t: int, state: Mapping[str, float]) -> float:
        return self.c / math.sqrt(t + 1.0)

    def config(self) -> dict[str, object]:
        return {"class": type(self).__name__, "c": self.c}


class DataDependentLR(LRSchedule):
    """観測に基づく適応型学習率 eta(t) = c / sqrt(1 + state[key])。

    既定では損失推定の二乗ノルム累積(AdaGrad / AdaHedge 型)を参照する。
    """

    def __init__(self, c: float = 1.0, key: str = "cum_sq_est_norm") -> None:
        if c <= 0:
            raise ValueError("c must be positive")
        self.c = c
        self.key = key

    def eta(self, t: int, state: Mapping[str, float]) -> float:
        return self.c / math.sqrt(1.0 + state.get(self.key, 0.0))

    def config(self) -> dict[str, object]:
        return {"class": type(self).__name__, "c": self.c, "key": self.key}
