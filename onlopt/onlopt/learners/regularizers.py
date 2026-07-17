"""正則化項(負エントロピー、Tsallis エントロピー、L2)。

FTRL と OMD の両方が単一のプリミティブ

    argmin_linear(theta) = argmin_{x in simplex} R(x) - <theta, x>

に帰着するため、各正則化項はこれを提供する。
FTRL のステップは theta = -eta * cum_loss、
OMD のステップは theta = grad(x_t) - eta * loss_estimate に対応する。
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from onlopt.geometry.projections import project_simplex


class Regularizer(ABC):
    """確率単体上の正則化項。"""

    @abstractmethod
    def value(self, x: np.ndarray) -> float:
        """R(x) を返す。"""

    @abstractmethod
    def grad(self, x: np.ndarray) -> np.ndarray:
        """∇R(x) を返す(単体の内部で定義)。"""

    @abstractmethod
    def argmin_linear(self, theta: np.ndarray) -> np.ndarray:
        """argmin_{x in simplex} R(x) - <theta, x> を返す。"""

    def argmin(self, cum_loss: np.ndarray, eta: float) -> np.ndarray:
        """正則化付き線形最適化 argmin_{x} <cum_loss, x> + R(x) / eta。"""
        return self.argmin_linear(-eta * np.asarray(cum_loss, dtype=np.float64))

    def config(self) -> dict[str, object]:
        return {"class": type(self).__name__}


class NegativeEntropy(Regularizer):
    """負エントロピー R(x) = sum_i x_i log x_i。argmin は softmax(閉形式)。"""

    def value(self, x: np.ndarray) -> float:
        x = np.asarray(x, dtype=np.float64)
        pos = x[x > 0]
        return float(np.sum(pos * np.log(pos)))

    def grad(self, x: np.ndarray) -> np.ndarray:
        return np.log(np.asarray(x, dtype=np.float64)) + 1.0

    def argmin_linear(self, theta: np.ndarray) -> np.ndarray:
        theta = np.asarray(theta, dtype=np.float64)
        w = np.exp(theta - theta.max())
        return w / w.sum()


class TsallisEntropy(Regularizer):
    """Tsallis エントロピーに基づく正則化項。

    R(x) = (1 - sum_i x_i^alpha) / (1 - alpha),  alpha in (0, 1)

    argmin は KKT 条件 x_i = kappa * (z - theta_i)^{-1/(1-alpha)} の
    正規化定数 z をニュートン法で解く。alpha = 1/2 が Tsallis-INF に対応。
    """

    def __init__(self, alpha: float = 0.5) -> None:
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha must be in (0, 1)")
        self.alpha = alpha

    def value(self, x: np.ndarray) -> float:
        x = np.asarray(x, dtype=np.float64)
        return float((1.0 - np.sum(x**self.alpha)) / (1.0 - self.alpha))

    def grad(self, x: np.ndarray) -> np.ndarray:
        a = self.alpha
        x = np.asarray(x, dtype=np.float64)
        return -a / (1.0 - a) * x ** (a - 1.0)

    def argmin_linear(
        self, theta: np.ndarray, tol: float = 1e-12, max_iter: int = 100
    ) -> np.ndarray:
        a = self.alpha
        theta = np.asarray(theta, dtype=np.float64)
        p = 1.0 / (1.0 - a)  # x_i = kappa * (z - theta_i)^{-p}
        kappa = ((1.0 - a) / a) ** (-p)
        # z の初期値: 最大座標の x がちょうど 1 になる点。ここで
        # g(z) = sum_i x_i(z) - 1 >= 0 かつ g は凸減少なので、
        # ニュートン法は単調に収束する。
        z = float(theta.max()) + a / (1.0 - a)
        x = np.empty_like(theta)
        for _ in range(max_iter):
            d = z - theta
            x = kappa * d ** (-p)
            g = float(x.sum()) - 1.0
            if abs(g) < tol:
                break
            gp = float(np.sum(-p * x / d))
            z -= g / gp
        x = np.maximum(x, 0.0)
        return x / x.sum()

    def config(self) -> dict[str, object]:
        return {"class": type(self).__name__, "alpha": self.alpha}


class L2(Regularizer):
    """L2 正則化 R(x) = ||x||^2 / 2。argmin は単体への Euclid 射影。"""

    def value(self, x: np.ndarray) -> float:
        x = np.asarray(x, dtype=np.float64)
        return float(0.5 * x @ x)

    def grad(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(x, dtype=np.float64).copy()

    def argmin_linear(self, theta: np.ndarray) -> np.ndarray:
        return project_simplex(np.asarray(theta, dtype=np.float64))
