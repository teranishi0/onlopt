"""リグレット・擬似リグレットの事後計算ユーティリティ。

Simulator は実行中にストリーミングで同等の量を計算するが、
損失行列を保持している小規模実験の事後解析用に純関数も提供する。
"""

from __future__ import annotations

import numpy as np


def cumulative_regret(
    incurred: np.ndarray, loss_matrix: np.ndarray
) -> np.ndarray:
    """累積リグレット系列を返す。

    regret_t = sum_{s<=t} incurred_s - min_i sum_{s<=t} l_{s,i}

    Args:
        incurred: 被った損失 (T,)。
        loss_matrix: 実現損失ベクトルの行列 (T, K)。
    """
    incurred = np.asarray(incurred, dtype=np.float64)
    loss_matrix = np.asarray(loss_matrix, dtype=np.float64)
    cum_arm = np.cumsum(loss_matrix, axis=0)
    best = cum_arm.min(axis=1)
    return np.cumsum(incurred) - best


def cumulative_pseudo_regret(
    mean_loss: np.ndarray,
    distributions: np.ndarray | None = None,
    actions: np.ndarray | None = None,
) -> np.ndarray:
    """擬似リグレット系列を返す。

    分布があれば pseudo_t = sum_{s<=t} (<p_s, mu> - mu_*)、
    なければ行動列から pseudo_t = sum_{s<=t} (mu[a_s] - mu_*)。
    """
    mu = np.asarray(mean_loss, dtype=np.float64)
    mu_star = mu.min()
    if distributions is not None:
        inst = np.asarray(distributions, dtype=np.float64) @ mu - mu_star
    elif actions is not None:
        inst = mu[np.asarray(actions, dtype=np.intp)] - mu_star
    else:
        raise ValueError("specify distributions or actions")
    return np.cumsum(inst)


def loglog_slope(
    cum_regret: np.ndarray, t_start: int, t_end: int | None = None
) -> float:
    """log-log 軸でのリグレット曲線の傾きを最小二乗で推定する。

    O(sqrt(T)) なら約 0.5、O(log T) なら 0 に漸近する。
    レジーム挙動テスト(仕様 8.3)に使用。
    """
    y = np.asarray(cum_regret, dtype=np.float64)
    t_end = t_end if t_end is not None else y.size
    ts = np.arange(t_start, t_end)
    vals = y[t_start:t_end]
    mask = vals > 0
    lx = np.log(ts[mask] + 1.0)
    ly = np.log(vals[mask])
    slope, _ = np.polyfit(lx, ly, 1)
    return float(slope)
