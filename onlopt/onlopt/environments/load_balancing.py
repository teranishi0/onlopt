"""負荷分散・スケジューリング環境。

各ラウンドでジョブ(サイズ s_t)が到着し、学習者がマシンを選んで
割り当てる。損失ベクトルは「マシン i に割り当てた場合の結果負荷」を
最大値で正規化したもの:

    l_{t,i} = (load_i + s_t) / (max_j load_j + s_t)  in (0, 1]

学習者の割当履歴に依存するため適応的環境の一種である。
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from onlopt.core.environment import Environment, History

# ジョブサイズ分布: (t, rng) -> サイズ(0 なら到着なし)
JobSizeFn = Callable[[int, np.random.Generator], float]


def constant_jobs(size: float = 1.0) -> JobSizeFn:
    """毎ラウンド固定サイズのジョブが到着する。"""
    return lambda t, rng: size


def uniform_jobs(low: float = 0.1, high: float = 1.0) -> JobSizeFn:
    """一様分布サイズのジョブが到着する。"""
    return lambda t, rng: float(rng.uniform(low, high))


def bernoulli_arrivals(p: float, size_fn: JobSizeFn) -> JobSizeFn:
    """確率 p でジョブが到着する到着過程。"""

    def fn(t: int, rng: np.random.Generator) -> float:
        return size_fn(t, rng) if rng.random() < p else 0.0

    return fn


class LoadBalancingEnv(Environment):
    def __init__(
        self,
        n_machines: int,
        job_size: JobSizeFn | None = None,
    ) -> None:
        if n_machines < 2:
            raise ValueError("n_machines must be >= 2")
        self.n_actions = n_machines
        self._job_size = job_size if job_size is not None else constant_jobs(1.0)
        self._loads = np.zeros(n_machines, dtype=np.float64)
        self._last_size = 0.0

    def reset(self) -> None:
        self._loads = np.zeros(self.n_actions, dtype=np.float64)
        self._last_size = 0.0

    def get_loss(
        self, t: int, history: History | None, rng: np.random.Generator
    ) -> np.ndarray:
        # 前ラウンドの割当を負荷に反映する
        if history is not None and len(history) > 0 and self._last_size > 0.0:
            prev = history.actions[-1]
            if isinstance(prev, np.ndarray):
                self._loads += self._last_size * prev
            else:
                self._loads[prev] += self._last_size
        size = float(self._job_size(t, rng))
        self._last_size = size
        denom = float(self._loads.max()) + size
        if denom <= 0.0:
            return np.zeros(self.n_actions, dtype=np.float64)
        return (self._loads + size) / denom

    @property
    def loads(self) -> np.ndarray:
        return self._loads.copy()

    def run_metadata(self) -> dict[str, object]:
        return {
            "final_loads": self._loads.tolist(),
            "makespan": float(self._loads.max()),
        }

    def config(self) -> dict[str, object]:
        return {"class": type(self).__name__, "n_machines": self.n_actions}
