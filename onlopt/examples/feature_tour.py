"""feature_tour.py — onlopt 全機能の使用方法デモ。

仕様書のコンポーネント構成(core / learners / environments / geometry /
evaluation / experiments / utils)に沿って、各機能を小さな実例で一巡する。

実行:  python examples/feature_tour.py
出力:  コンソールに各セクションの結果、カレントディレクトリに PNG 図
"""

import sys

import matplotlib

matplotlib.use("Agg")

import numpy as np

# Windows の cp932 コンソールでも全文字を出力できるようにする
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def section(title: str) -> None:
    print()
    print("=" * 64)
    print(f"  {title}")
    print("=" * 64)


# ----------------------------------------------------------------
section("1. utils/rng — 乱数の一元管理(再現性の基盤)")
# ----------------------------------------------------------------
from onlopt import make_rng, spawn_rngs

rng = make_rng(seed=42)
print("make_rng(42) から3つの乱数 :", np.round(rng.random(3), 4))
print("同じシードなら同じ値       :", np.round(make_rng(42).random(3), 4))

# 複数シード実験用に統計的に独立な生成器を派生できる
rngs = spawn_rngs(seed=42, n=3)
print("spawn_rngs(42, 3) の各先頭 :", [round(g.random(), 4) for g in rngs])


# ----------------------------------------------------------------
section("2. geometry/projections — 決定集合への射影")
# ----------------------------------------------------------------
from onlopt.geometry import (
    project_l2_ball,
    project_product_simplex,
    project_simplex,
)

v = np.array([0.8, 0.6, -0.3])
x = project_simplex(v)
print(f"単体への射影      : {v} -> {np.round(x, 4)} (sum={x.sum():.4f})")

b = project_l2_ball(np.array([3.0, 4.0]), radius=1.0)
print(f"L2球への射影      : [3, 4] -> {np.round(b, 4)} (norm={np.linalg.norm(b):.4f})")

pp = project_product_simplex(np.array([2.0, 0.0, 1.0, 1.0]), block_sizes=[2, 2])
print(f"単体の直積への射影: -> {np.round(pp, 4)} (各ブロック和=1)")


# ----------------------------------------------------------------
section("3. learners/regularizers — 正則化付き線形最適化 argmin")
# ----------------------------------------------------------------
from onlopt import L2, NegativeEntropy, TsallisEntropy

cum_loss = np.array([1.0, 2.0, 3.0, 0.5])
for reg in [NegativeEntropy(), TsallisEntropy(alpha=0.5), L2()]:
    x = reg.argmin(cum_loss, eta=1.0)  # argmin <L, x> + R(x)/eta
    name = reg.config().get("class")
    print(f"{name:16s}: argmin = {np.round(x, 4)}")
print("→ 同じ累積損失でも正則化項で分布の「尖り方」が変わる")
print("  (L2 はスパース、負エントロピーは指数重み、Tsallis は中間)")


# ----------------------------------------------------------------
section("4. learners/lr_schedules — 学習率スケジューラ")
# ----------------------------------------------------------------
from onlopt import DataDependentLR, FixedLR, InverseSqrtLR

state = {"cum_sq_est_norm": 24.0}  # FTRL/OMD が公開する内部累積量
for lr in [FixedLR(0.1), InverseSqrtLR(c=1.0), DataDependentLR(c=1.0)]:
    vals = [round(lr.eta(t, state), 4) for t in (0, 9, 99)]
    print(f"{lr.config().get('class'):16s}: eta(t=0,9,99) = {vals}")
print("→ DataDependentLR は観測量 state を参照する適応型(BOBW の学習率設計用)")


# ----------------------------------------------------------------
section("5. learners/estimators — バンディットの不偏損失推定")
# ----------------------------------------------------------------
from onlopt import Feedback
from onlopt.learners import ImportanceWeighted

est = ImportanceWeighted()
dist = np.array([0.5, 0.3, 0.2])
fb = Feedback(action=1, loss=0.6)  # 腕1を引いて損失0.6を観測
print("観測: 腕1, 損失0.6, 分布", dist)
print("IW 推定ベクトル:", est.estimate(fb, dist, 3), " (0.6/0.3=2.0)")

# 不偏性のモンテカルロ確認: E[推定] = 真の損失
true_loss = np.array([0.2, 0.6, 0.9])
r = make_rng(0)
acc = np.zeros(3)
n = 50_000
for a in r.choice(3, size=n, p=dist):
    acc += est.estimate(Feedback(action=int(a), loss=float(true_loss[a])), dist, 3)
print("モンテカルロ平均:", np.round(acc / n, 3), "≈ 真値", true_loss)


# ----------------------------------------------------------------
section("6. environments — 3レジーム+負荷分散")
# ----------------------------------------------------------------
from onlopt import (
    AdaptiveAdversary,
    CorruptedStochasticEnv,
    LoadBalancingEnv,
    ObliviousAdversary,
    StochasticEnv,
)
from onlopt.core import History
from onlopt.environments import uniform_jobs

r = make_rng(0)

# (a) 確率的環境: Bernoulli / Beta / ギャップ指定
sto = StochasticEnv.from_gap(n_actions=4, gap=0.3)  # 最良腕の平均だけ 0.2
print("StochasticEnv.from_gap  平均損失 :", sto.mean_loss)
print("                        損失例   :", sto.get_loss(0, None, r))
beta_env = StochasticEnv.beta(alphas=[1, 2], betas=[3, 2])
print("StochasticEnv.beta      平均損失 :", beta_env.mean_loss)

# (b) 固定的敵対者: 損失系列の生成規則を callable で与える
adv = ObliviousAdversary(3, lambda t, rng: rng.random(3))
print("ObliviousAdversary      損失例   :", np.round(adv.get_loss(0, None, r), 3))

# (c) 適応的敵対者: 学習者の直前の行動に高い損失を課す例
def punish_last_action(t, history, rng):
    loss = np.full(3, 0.1)
    if history is not None and len(history) > 0:
        loss[history.actions[-1]] = 1.0
    return loss

adaptive = AdaptiveAdversary(3, punish_last_action)
h = History(actions=[2], losses=[0.1])
print("AdaptiveAdversary       直前=腕2 :", adaptive.get_loss(1, h, r))

# (d) 汚染付き確率的環境: 予算 C の範囲で最良腕を偽装
cor = CorruptedStochasticEnv(StochasticEnv.from_gap(4, 0.3), budget=3.0)
for t in range(5):
    cor.get_loss(t, None, r)
print(f"CorruptedStochasticEnv  5R後の消費汚染量: {cor.corruption_used:.1f} / 予算 3.0")

# (e) 負荷分散環境: 割当履歴に依存して損失が決まる
lb = LoadBalancingEnv(n_machines=3, job_size=uniform_jobs(0.5, 1.0))
print("LoadBalancingEnv        初回損失 :", np.round(lb.get_loss(0, History(), r), 3))


# ----------------------------------------------------------------
section("7. core/Simulator — 1回の実験と RunResult")
# ----------------------------------------------------------------
from onlopt import Hedge, Simulator

T = 2000
res = Simulator().run(
    learner=Hedge(n_actions=4, horizon=T),  # eta を理論最適に自動設定
    env=StochasticEnv.from_gap(4, gap=0.3),
    T=T,
    seed=0,
)
print(f"最終リグレット      : {res.cum_regret[-1]:.1f}")
print(f"最終擬似リグレット  : {res.cum_pseudo_regret[-1]:.1f}  (確率的環境のみ)")
print(f"行動の記録          : 先頭10ラウンド {res.actions[:10]}")
print(f"設定スナップショット: {res.config['learner']}")


# ----------------------------------------------------------------
section("8. learners — アルゴリズム動物園(同一環境で比較)")
# ----------------------------------------------------------------
from onlopt import FTRL, OGD, OMD, FeedbackType, TsallisINF
from onlopt.experiments import run_grid

K, T = 8, 5000
env_f = lambda: StochasticEnv.from_gap(K, gap=0.25)

zoo = {
    # full-info 系
    "Hedge": (lambda: Hedge(K, horizon=T), env_f),
    "FTRL (NegEnt)": (
        lambda: FTRL(K, NegativeEntropy(), InverseSqrtLR(c=1.0)),
        env_f,
    ),
    "OMD (Tsallis)": (
        lambda: OMD(K, TsallisEntropy(0.5), InverseSqrtLR(c=1.0)),
        env_f,
    ),
    "OGD": (lambda: OGD(K, InverseSqrtLR(c=0.5)), env_f),
    # bandit 系(観測が少ないぶん不利)
    "Tsallis-INF (bandit)": (lambda: TsallisINF(K), env_f),
}
out = run_grid(zoo, T=T, seeds=range(5))
for name, r_ in out.items():
    print(f"{name:22s}: 最終リグレット {r_.aggregate.mean[-1]:7.1f}"
          f" ± {r_.aggregate.mean[-1] - r_.aggregate.lower[-1]:.1f}")


# ----------------------------------------------------------------
section("9. 新しいアルゴリズムの追加(基底クラス継承のみで完結)")
# ----------------------------------------------------------------
from onlopt import Learner


class EpsilonGreedy(Learner):
    """設計目標のデモ: predict / update の実装だけで Simulator に載る。"""

    feedback_type = FeedbackType.BANDIT

    def __init__(self, n_actions: int, eps: float = 0.1):
        self.n_actions, self.eps = n_actions, eps
        self.reset()

    def reset(self) -> None:
        self._sum = np.zeros(self.n_actions)
        self._cnt = np.zeros(self.n_actions)

    def predict(self, rng) -> int:
        if rng.random() < self.eps or self._cnt.min() == 0:
            return int(rng.integers(self.n_actions))
        return int(np.argmin(self._sum / self._cnt))

    def update(self, feedback) -> None:
        a = feedback.action
        self._sum[a] += feedback.loss
        self._cnt[a] += 1


res_eg = Simulator().run(EpsilonGreedy(4), StochasticEnv.from_gap(4, 0.3), 3000, seed=0)
print(f"EpsilonGreedy 最終擬似リグレット: {res_eg.cum_pseudo_regret[-1]:.1f}")
print("(distribution を持たない学習者は行動列から擬似リグレットを計算)")


# ----------------------------------------------------------------
section("10. 学習率設計の差し替え実験(BOBW 研究の中心操作)")
# ----------------------------------------------------------------
from onlopt.learners import LRSchedule


class PolynomialLR(LRSchedule):
    """自作スケジュール eta(t) = c / (t+1)^p も注入するだけで使える。"""

    def __init__(self, c: float = 1.0, p: float = 0.5):
        self.c, self.p = c, p

    def eta(self, t, state):
        return self.c / (t + 1.0) ** self.p


lr_variants = {
    "c/sqrt(t)": InverseSqrtLR(c=2.0),
    "data-dependent": DataDependentLR(c=2.0),
    "c/t^0.6 (自作)": PolynomialLR(c=2.0, p=0.6),
}
for name, lr in lr_variants.items():
    tsallis = TsallisINF(4, lr_schedule=lr)
    r_ = Simulator().run(tsallis, StochasticEnv.from_gap(4, 0.3), 5000, seed=1)
    print(f"Tsallis-INF + {name:16s}: 擬似リグレット {r_.cum_pseudo_regret[-1]:6.1f}")


# ----------------------------------------------------------------
section("11. evaluation — 集計(std / bootstrap)と傾き推定")
# ----------------------------------------------------------------
from onlopt import aggregate_regret, loglog_slope
from onlopt.experiments import run_many

many = run_many(
    lambda: TsallisINF(4),
    lambda: StochasticEnv.from_gap(4, 0.3),
    T=5000,
    seeds=range(8),
)
agg_std = many.aggregate  # 既定は平均 ± 標準偏差
agg_boot = aggregate_regret(many.results, method="bootstrap", confidence=0.95)
print(f"std 帯      : {agg_std.mean[-1]:.1f} [{agg_std.lower[-1]:.1f}, {agg_std.upper[-1]:.1f}]")
print(f"bootstrap CI: {agg_boot.mean[-1]:.1f} [{agg_boot.lower[-1]:.1f}, {agg_boot.upper[-1]:.1f}]")

slope = loglog_slope(many.pseudo_aggregate.mean, t_start=1000)
print(f"擬似リグレットの log-log 傾き: {slope:.2f}"
      "  (log T スケールなら 0.5 より十分小さい)")


# ----------------------------------------------------------------
section("12. experiments/config — 設定の JSON 保存・読込(再現性)")
# ----------------------------------------------------------------
from onlopt import ExperimentConfig

cfg = ExperimentConfig(
    name="tsallis_stochastic_gap03",
    T=5000,
    seeds=list(range(8)),
    learner={"class": "TsallisINF", "n_actions": 4, "c": 2.0},
    environment={"class": "StochasticEnv", "gap": 0.3},
    aggregation="bootstrap",
    notes="feature tour のデモ設定",
)
cfg.save_json("feature_tour_config.json")
loaded = ExperimentConfig.load_json("feature_tour_config.json")
print("保存 -> 読込の一致:", loaded == cfg)
print("読込した設定      :", loaded.name, f"T={loaded.T}, {len(loaded.seeds)} seeds")


# ----------------------------------------------------------------
section("13. evaluation/plotting — リグレット曲線の描画")
# ----------------------------------------------------------------
from onlopt import plot_regret

# 線形軸: アルゴリズム比較(セクション8の結果を流用)
fig1 = plot_regret(
    {name: r_.aggregate for name, r_ in out.items()},
    title="Algorithm zoo on stochastic env (mean ± std)",
)
fig1.savefig("feature_tour_zoo.png", dpi=140)

# log-log 軸 + 参照傾き線: BOBW 検証の定番プロット
fig2 = plot_regret(
    {"Tsallis-INF (pseudo)": many.pseudo_aggregate},
    loglog=True,
    references=("sqrt", "log"),
    title="Pseudo-regret with reference slopes",
)
fig2.savefig("feature_tour_loglog.png", dpi=140)
print("保存した図: feature_tour_zoo.png, feature_tour_loglog.png")

print()
print("=" * 64)
print("  feature tour 完了")
print("=" * 64)
