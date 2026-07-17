"""BOBW デモ: Tsallis-INF を3レジームで比較する。

確率的・敵対的・汚染付き確率的の各環境で Tsallis-INF を実行し、
log-log 軸のリグレット曲線に参照傾き(sqrt T, log T)を重ねて描画する。

実行:  python examples/bobw_demo.py
"""

import matplotlib

matplotlib.use("Agg")

import numpy as np

from onlopt import (
    CorruptedStochasticEnv,
    ObliviousAdversary,
    StochasticEnv,
    TsallisINF,
    plot_regret,
)
from onlopt.experiments import run_grid

T = 20_000
K = 8
SEEDS = list(range(10))


def adversarial_loss(t: int, rng: np.random.Generator) -> np.ndarray:
    return rng.random(K)


settings = {
    "stochastic": (
        lambda: TsallisINF(K),
        lambda: StochasticEnv.from_gap(K, gap=0.25),
    ),
    "adversarial": (
        lambda: TsallisINF(K),
        lambda: ObliviousAdversary(K, adversarial_loss),
    ),
    "corrupted (C=200)": (
        lambda: TsallisINF(K),
        lambda: CorruptedStochasticEnv(
            StochasticEnv.from_gap(K, gap=0.25), budget=200.0
        ),
    ),
}

if __name__ == "__main__":
    print(f"running Tsallis-INF on 3 regimes (T={T}, {len(SEEDS)} seeds)...")
    out = run_grid(settings, T=T, seeds=SEEDS)

    for name, r in out.items():
        used = r.results[0].config.get("corruption_used")
        extra = f", corruption used: {used:.0f}" if used is not None else ""
        print(f"  {name:>20}: final regret = {r.aggregate.mean[-1]:8.1f}{extra}")

    curves = {name: r.aggregate for name, r in out.items()}
    fig = plot_regret(
        curves,
        loglog=True,
        references=("sqrt", "log"),
        title=f"Tsallis-INF, K={K}, {len(SEEDS)} seeds (mean ± std)",
    )
    fig.savefig("bobw_demo.png", dpi=150)
    print("saved: bobw_demo.png")
