# onlopt

オンライン最適化研究用 Python モジュール(v0.1)。
Best-of-Both-Worlds(BOBW)アルゴリズム、リグレット最小化、オンライン線形最適化(OLO)の数値実験基盤。
詳細は [online_opt_module_spec.md](online_opt_module_spec.md) を参照。

## インストール

```bash
pip install -e .          # 必須依存: numpy, matplotlib
pip install -e ".[dev]"   # 開発用: pytest, mypy, ruff
```

## クイックスタート

```python
from onlopt import Hedge, Simulator, StochasticEnv

env = StochasticEnv.from_gap(n_actions=10, gap=0.2)   # Bernoulli, ギャップ 0.2
learner = Hedge(n_actions=10, horizon=10_000)          # eta を理論最適に自動設定
result = Simulator().run(learner, env, T=10_000, seed=0)
print(result.cum_regret[-1], result.cum_pseudo_regret[-1])
```

### 複数シード実行とプロット

```python
from onlopt import StochasticEnv, TsallisINF, plot_regret
from onlopt.experiments import run_many

out = run_many(
    learner_factory=lambda: TsallisINF(8),
    env_factory=lambda: StochasticEnv.from_gap(8, gap=0.25),
    T=20_000,
    seeds=range(10),
    aggregation="std",          # または "bootstrap"
)
fig = plot_regret(
    {"Tsallis-INF": out.pseudo_aggregate},
    loglog=True,
    references=("sqrt", "log"),  # BOBW の両レジーム確認用の参照傾き
)
fig.savefig("regret.png")        # 保存はユーザー側の責務
```

### FTRL フレームワークによるアルゴリズム構成

Tsallis-INF は FTRL の3部品(正則化項・学習率・損失推定)の合成として定義される:

```python
from onlopt import FTRL, FeedbackType, InverseSqrtLR, TsallisEntropy
from onlopt.learners import ImportanceWeighted

tsallis_inf = FTRL(
    n_actions=8,
    regularizer=TsallisEntropy(alpha=0.5),
    lr_schedule=InverseSqrtLR(c=2.0),
    loss_estimator=ImportanceWeighted(),
    feedback_type=FeedbackType.BANDIT,
)
```

学習率設計の差し替え実験には `DataDependentLR`(学習者の内部累積量
`state["cum_sq_est_norm"]` 等を参照)や自作の `LRSchedule` を注入できる。

### 3レジーム(BOBW 検証)

```python
from onlopt import CorruptedStochasticEnv, ObliviousAdversary, StochasticEnv

sto = StochasticEnv.from_gap(8, gap=0.25)                     # 確率的
adv = ObliviousAdversary(8, lambda t, rng: rng.random(8))     # 敵対的
cor = CorruptedStochasticEnv(sto, budget=200.0)               # 汚染付き確率的
# 実際に消費した汚染量は RunResult.config["corruption_used"] に記録される
```

デモ: `python examples/bobw_demo.py`

## 仕様書からの設計上の決定(v0.1)

- **`Environment.get_loss` の第3引数に `rng` を追加**(仕様 3.3 は
  `get_loss(t, history)`)。再現性規約(6.4: 環境は乱数を自前で保持せず
  Simulator から渡された Generator を使用する)を引数渡しで満たすため。
- **FTRL と OMD の統一プリミティブ**: 各 `Regularizer` は
  `argmin_linear(theta) = argmin_{x∈Δ} R(x) − ⟨theta, x⟩` を提供し、
  FTRL は `theta = −η L̂`、OMD は `theta = ∇R(x_t) − η ĝ` で共有する。
- **リグレット系列のストリーミング計算**: T=10^6 で損失行列 (T, K) を
  保持しないよう、座標ごとの累積損失の最小値と比較して逐次計算する
  (単体型の決定集合で有効。一般の決定集合は `best_fixed_action` で拡張)。
- **汚染量の計上**: ラウンドごとの max ノルム `‖corrupted − clean‖∞` で
  予算を消費する(Zimmert & Seldin 2021 の定義に整合)。

## テスト

```bash
python -m pytest                  # 全テスト(レジーム挙動テスト含む、~30秒)
python -m pytest -m "not slow"    # 高速テストのみ(~6秒)
```

- 単体テスト: 射影の KKT 条件、IW 推定の不偏性(モンテカルロ)、学習率の値
- 理論整合性: Hedge のリグレット < √(T ln K / 2)(η 最適時、10シード平均)
- レジーム挙動: Tsallis-INF の log-log 傾きが確率的で小、敵対的で ≈ 0.5
- 回帰テスト: 固定シードの `RunResult` ハッシュ比較

## 性能

K=100、T=10^6、単一シードの Hedge 実行: 約23秒(要件: 1分以内)。

## マイルストーン対応状況

| 版 | 内容 | 状態 |
|---|---|---|
| v0.1 | core + Hedge + 確率的/敵対的環境 + リグレットプロット | ✅ |
| v0.2 | FTRL フレームワーク + Tsallis-INF + 汚染環境 | ✅(前倒し実装) |
| v0.3 | OLO(射影勾配)+ 負荷分散環境 | ✅(前倒し実装) |
| v0.4 | 実験ランナー整備、設定保存、ドキュメント | ✅(前倒し実装) |

組合せ的決定集合(semi-bandit の本格対応)は仕様どおり v0.2 以降の拡張。
