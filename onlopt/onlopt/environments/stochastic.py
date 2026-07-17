"""i.i.d. 確率的環境。"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

from onlopt.core.environment import Environment, History

# 腕ごとの損失サンプラー: rng を受け取り [0, 1] の損失を返す
ArmSampler = Callable[[np.random.Generator], float]


class StochasticEnv(Environment):
    """各腕の損失が i.i.d. な確率的環境。

    既定は Bernoulli(means)。任意分布は samplers(callable の列)で与え、
    擬似リグレット計算のために sampler_means を併せて指定できる。
    """

    def __init__(
        self,
        means: Sequence[float] | np.ndarray | None = None,
        samplers: Sequence[ArmSampler] | None = None,
        sampler_means: Sequence[float] | np.ndarray | None = None,
    ) -> None:
        if (means is None) == (samplers is None):
            raise ValueError("specify exactly one of means / samplers")
        self._samplers = list(samplers) if samplers is not None else None
        if means is not None:
            self._means: np.ndarray | None = np.asarray(means, dtype=np.float64)
            if self._means.min() < 0.0 or self._means.max() > 1.0:
                raise ValueError("means must lie in [0, 1]")
            self.n_actions = self._means.size
        else:
            assert self._samplers is not None
            self.n_actions = len(self._samplers)
            self._means = (
                np.asarray(sampler_means, dtype=np.float64)
                if sampler_means is not None
                else None
            )

    @classmethod
    def bernoulli(cls, means: Sequence[float] | np.ndarray) -> StochasticEnv:
        return cls(means=means)

    @classmethod
    def beta(
        cls, alphas: Sequence[float], betas: Sequence[float]
    ) -> StochasticEnv:
        """Beta(alpha_i, beta_i) 損失の環境。"""
        if len(alphas) != len(betas):
            raise ValueError("alphas and betas must have the same length")

        def make(a: float, b: float) -> ArmSampler:
            return lambda rng: float(rng.beta(a, b))

        samplers = [make(a, b) for a, b in zip(alphas, betas)]
        means = [a / (a + b) for a, b in zip(alphas, betas)]
        return cls(samplers=samplers, sampler_means=means)

    @classmethod
    def from_gap(
        cls, n_actions: int, gap: float, best_arm: int = 0, base: float = 0.5
    ) -> StochasticEnv:
        """最良腕の平均が base - gap、それ以外が base の Bernoulli 環境。"""
        if not 0.0 < gap <= base:
            raise ValueError("gap must be in (0, base]")
        means = np.full(n_actions, base, dtype=np.float64)
        means[best_arm] = base - gap
        return cls(means=means)

    def get_loss(
        self, t: int, history: History | None, rng: np.random.Generator
    ) -> np.ndarray:
        if self._samplers is not None:
            return np.array([s(rng) for s in self._samplers], dtype=np.float64)
        assert self._means is not None
        return (rng.random(self.n_actions) < self._means).astype(np.float64)

    @property
    def mean_loss(self) -> np.ndarray | None:
        return self._means

    def best_fixed_action(self, T: int) -> tuple[int, float] | None:
        if self._means is None:
            return None
        best = int(np.argmin(self._means))
        return best, float(self._means[best] * T)

    def config(self) -> dict[str, object]:
        cfg: dict[str, object] = {
            "class": type(self).__name__,
            "n_actions": self.n_actions,
        }
        if self._means is not None:
            cfg["means"] = self._means.tolist()
        return cfg
