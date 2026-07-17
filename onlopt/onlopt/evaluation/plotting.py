"""リグレット曲線プロット(log-log 対応)。

matplotlib の Figure を返す設計とし、保存はユーザー側の責務とする。
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from onlopt.evaluation.aggregate import AggregateResult

ReferenceSlope = str  # "sqrt" | "log" | "linear"


def _reference_curve(kind: ReferenceSlope, ts: np.ndarray) -> np.ndarray:
    if kind == "sqrt":
        return np.sqrt(ts)
    if kind == "log":
        return np.log(ts + 1.0)
    if kind == "linear":
        return ts.astype(np.float64)
    raise ValueError(f"unknown reference slope: {kind}")


def plot_regret(
    curves: Mapping[str, AggregateResult] | AggregateResult,
    loglog: bool = False,
    references: Iterable[ReferenceSlope] = (),
    title: str | None = None,
    ylabel: str = "cumulative regret",
    ax: Axes | None = None,
) -> Figure:
    """リグレット曲線(平均 ± 帯)を描画して Figure を返す。

    Args:
        curves: ラベル -> AggregateResult の対応(単一でも可)。
        loglog: True なら log-log 軸で描画する。
        references: log-log 軸に重畳する参照傾き("sqrt", "log", "linear")。
            BOBW の両レジーム最適性を視覚的に確認するために使う。
        ax: 既存の Axes に描く場合に指定。
    """
    if isinstance(curves, AggregateResult):
        curves = {"learner": curves}

    if ax is None:
        fig, ax = plt.subplots(figsize=(7.0, 4.5))
    else:
        fig = ax.figure  # type: ignore[assignment]

    ts: np.ndarray | None = None
    for label, agg in curves.items():
        t = np.arange(1, agg.mean.size + 1)
        ts = t if ts is None or t.size > ts.size else ts
        (line,) = ax.plot(t, agg.mean, label=label, linewidth=1.6)
        ax.fill_between(
            t, agg.lower, agg.upper, alpha=0.2, color=line.get_color(), linewidth=0
        )

    if ts is not None:
        anchor_agg = next(iter(curves.values()))
        anchor = float(max(anchor_agg.mean[-1], 1e-12))
        for kind in references:
            ref = _reference_curve(kind, ts.astype(np.float64))
            scale = anchor / float(ref[-1])
            ax.plot(
                ts,
                scale * ref,
                linestyle="--",
                linewidth=1.0,
                color="gray",
                label=f"~{kind}(T)",
            )

    if loglog:
        ax.set_xscale("log")
        ax.set_yscale("log")
    ax.set_xlabel("round $t$")
    ax.set_ylabel(ylabel)
    if title is not None:
        ax.set_title(title)
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    return fig
