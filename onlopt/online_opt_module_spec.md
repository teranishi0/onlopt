# オンライン最適化研究用 Python モジュール 仕様書

- **モジュール名(仮)**: `onlopt`
- **版**: v0.1(ドラフト)
- **作成日**: 2026-07-17
- **対象**: Best-of-Both-Worlds(BOBW)アルゴリズム、リグレット最小化、オンライン線形最適化(OLO)の数値実験

---

## 1. 目的と背景

本モジュールは、オンライン学習アルゴリズムの理論的性質(リグレット上界、確率的/敵対的レジームでの挙動)を数値実験で検証するための研究基盤である。以下を設計上の目的とする。

1. アルゴリズム・環境・評価を独立したコンポーネントとして分離し、任意の組み合わせで実験できること
2. BOBW の検証に必要な3レジーム(確率的・敵対的・汚染付き確率的)を統一インターフェースで扱えること
3. 論文用実験に耐える再現性(シード管理・設定の保存)を備えること
4. 新しいアルゴリズムの追加が、基底クラスの継承と少数のメソッド実装で完結すること

### 1.1 スコープ外(v0.1)

- 文脈付きバンディット、強化学習
- 分散実行・GPU 対応
- GUI / Web インターフェース

## 2. 全体アーキテクチャ

```
onlopt/
├── core/
│   ├── learner.py        # Learner 基底クラス、フィードバック型定義
│   ├── environment.py    # Environment 基底クラス
│   └── simulator.py      # 実験ループ
├── learners/
│   ├── hedge.py          # Hedge(指数重み法)
│   ├── ftrl.py           # FTRL 汎用フレームワーク
│   ├── omd.py            # OMD 汎用フレームワーク
│   ├── regularizers.py   # 正則化項(負エントロピー、Tsallis、L2)
│   ├── lr_schedules.py   # 学習率スケジューラ
│   └── tsallis_inf.py    # Tsallis-INF(FTRL + Tsallis + 適応学習率)
├── environments/
│   ├── stochastic.py     # i.i.d. 確率的環境
│   ├── adversarial.py    # 固定的/適応的敵対者
│   ├── corrupted.py      # 汚染付き確率的環境
│   └── load_balancing.py # 負荷分散・スケジューリング環境
├── geometry/
│   └── projections.py    # 決定集合への射影(単体、球、組合せ的集合)
├── evaluation/
│   ├── regret.py         # リグレット・擬似リグレット計算
│   ├── aggregate.py      # 複数シード集計(平均・信頼区間)
│   └── plotting.py       # リグレット曲線プロット(log-log 対応)
├── experiments/
│   ├── config.py         # 実験設定 dataclass、保存・読込
│   └── runner.py         # 複数シード・複数設定の一括実行
└── utils/
    └── rng.py            # 乱数生成器の一元管理
```

依存関係は `learners / environments → core → utils` の一方向とし、`evaluation` と `experiments` は `core` のみに依存する。

## 3. コアインターフェース仕様

### 3.1 フィードバック型

```python
class FeedbackType(Enum):
    FULL_INFO   = auto()  # 損失ベクトル全体を観測
    BANDIT      = auto()  # 選択した行動の損失のみ観測
    SEMI_BANDIT = auto()  # 選択した組合せ的行動の成分ごとの損失を観測
```

Learner は自身が要求するフィードバック型を `feedback_type` プロパティで宣言し、Simulator が Environment の返す損失ベクトルを適切にマスクして渡す。

### 3.2 Learner 基底クラス

```python
class Learner(ABC):
    feedback_type: FeedbackType

    @abstractmethod
    def predict(self, rng: np.random.Generator) -> Action:
        """現ラウンドの行動(または行動上の分布からのサンプル)を返す。"""

    @abstractmethod
    def update(self, feedback: Feedback) -> None:
        """観測したフィードバックで内部状態を更新する。"""

    def reset(self) -> None:
        """内部状態を初期化する(複数シード実行用)。"""
```

- `Action` は v0.1 では腕インデックス `int` または決定集合上の点 `np.ndarray`
- `Feedback` は dataclass とし、`loss`(観測値)、`action`(選択した行動)、必要に応じて `full_loss`(full-info 時のみ)を保持する
- 分布を明示的に持つ Learner(Hedge 等)は `distribution` プロパティで現在の行動分布を公開する(擬似リグレット計算に使用)

### 3.3 Environment 基底クラス

```python
class Environment(ABC):
    n_actions: int  # または決定集合の記述

    @abstractmethod
    def get_loss(self, t: int, history: History | None) -> np.ndarray:
        """ラウンド t の損失ベクトルを返す。

        適応的敵対者は history(学習者の過去の行動列)を参照してよい。
        非適応的環境は history を無視する。
        """

    def best_fixed_action(self, T: int) -> tuple[Action, float]:
        """T ラウンドでの事後最良固定行動とその累積損失を返す(リグレット計算用)。"""
```

損失は `[0, 1]` に正規化することを規約とする(範囲外の場合は Simulator が警告を出す)。

### 3.4 Simulator

```python
class Simulator:
    def run(
        self,
        learner: Learner,
        env: Environment,
        T: int,
        seed: int,
    ) -> RunResult:
        ...
```

`RunResult` は以下を保持する dataclass:

| フィールド | 型 | 内容 |
|---|---|---|
| `actions` | `list[Action]` | 各ラウンドの選択行動 |
| `losses` | `np.ndarray (T,)` | 各ラウンドの被った損失 |
| `cum_regret` | `np.ndarray (T,)` | 累積リグレット系列 |
| `cum_pseudo_regret` | `np.ndarray (T,)` | 擬似リグレット系列(計算可能な場合) |
| `config` | `dict` | 実験設定のスナップショット |
| `seed` | `int` | 使用シード |

## 4. アルゴリズム仕様

### 4.1 実装優先順位

| 優先度 | アルゴリズム | フィードバック | 備考 |
|---|---|---|---|
| P0 | Hedge(指数重み法) | full-info | 基準線。理論上界 √(T log K)/η 最適化版 |
| P0 | FTRL 汎用フレームワーク | full-info / bandit | 正則化項・学習率を注入可能 |
| P1 | Tsallis-INF | bandit | FTRL + Tsallis(α=1/2)+ 適応学習率で構成 |
| P1 | OMD 汎用フレームワーク | full-info / bandit | FTRL と鏡像を共有 |
| P2 | OLO 用射影勾配型(OGD 等) | full-info | 負荷分散応用の足がかり |

### 4.2 FTRL フレームワークの構成部品

FTRL は次の3部品の合成として実装する。

```python
FTRL(regularizer: Regularizer, lr_schedule: LRSchedule, loss_estimator: LossEstimator)
```

- **Regularizer**: `value(x)`, `grad(x)`, および正則化付き線形最適化 `argmin(cum_loss, eta)` を提供。実装対象: 負エントロピー、Tsallis エントロピー(α をパラメータ化)、L2
- **LRSchedule**: `eta(t, state)` を返す。実装対象: 固定、`c/√t`、データ依存(観測に基づく適応型)。BOBW の学習率設計を差し替え実験できるよう、内部状態(累積量など)へのアクセスを許す
- **LossEstimator**: バンディットフィードバックからの不偏推定(importance weighting)。full-info では恒等写像

この分解により Tsallis-INF は「FTRL + Tsallis(α=1/2) + 適応学習率 + IW 推定」の組み合わせとして定義ファイル数十行で表現できることを受け入れ条件とする。

### 4.3 射影ユーティリティ

`geometry/projections.py` に以下を実装する。

- 確率単体への射影(ソート法、O(K log K))
- L2 球への射影
- 負荷分散向け決定集合(v0.1 では単体の直積まで。組合せ的集合は v0.2 で拡張)

## 5. 環境仕様

| 環境 | パラメータ | 用途 |
|---|---|---|
| `StochasticEnv` | 各腕の損失分布(Bernoulli / Beta / 任意の callable)、ギャップ Δ | 確率的レジーム。O(log T) 挙動の確認 |
| `ObliviousAdversary` | 損失系列の生成規則(callable) | 敵対的レジーム。O(√T) 挙動の確認 |
| `AdaptiveAdversary` | 学習者の行動履歴を受け取る戦略(callable) | 適応的敵対者の実験 |
| `CorruptedStochasticEnv` | ベースの確率的環境、汚染予算 C、汚染戦略 | BOBW の中間レジーム検証 |
| `LoadBalancingEnv` | マシン数、ジョブサイズ分布、到着過程 | 負荷分散応用 |

`CorruptedStochasticEnv` は実際に消費した汚染量を記録し、`RunResult.config` に含める。

## 6. 評価・記録仕様

### 6.1 リグレット計算

- **リグレット**: 実現損失の累積 −(事後最良固定行動の累積損失)
- **擬似リグレット**: 学習者の行動分布に対する期待損失を用いた版。確率的環境でのみ計算し、環境側が期待損失ベクトルを提供する場合に有効化

### 6.2 集計

`experiments/runner.py` の `run_many(learner_factory, env_factory, T, seeds)` が複数シードを実行し、以下を返す。

- ラウンドごとの平均リグレットと標準偏差(または bootstrap 信頼区間、方式は設定で選択)
- 全 `RunResult` のリスト(生データ保持)

### 6.3 プロット

- リグレット曲線(平均 ± 帯)。線形軸と log-log 軸を切替可能
- log-log 軸には参照傾き線(√T、log T)をオプションで重畳表示し、BOBW の両レジーム最適性を視覚的に確認できること
- 出力は matplotlib の `Figure` を返す設計とし、保存はユーザー側の責務とする

### 6.4 再現性

- 実験設定は `ExperimentConfig`(dataclass)で表現し、JSON へのシリアライズ・デシリアライズをサポート
- 乱数は `utils/rng.py` の `make_rng(seed)` で生成した `numpy.random.Generator` のみを使用する。グローバル乱数状態(`np.random.seed` 等)の使用は禁止
- Learner・Environment は乱数を自前で保持せず、Simulator から渡された Generator を使用する

## 7. 非機能要件

| 項目 | 要件 |
|---|---|
| 言語・依存 | Python 3.11+、必須依存は numpy と matplotlib のみ。scipy はオプション |
| 型 | 全公開 API に型ヒントを付与。mypy(strict 相当)を CI で実行 |
| テスト | pytest。カバレッジ目標: core / learners で 90% 以上 |
| 性能 | K=100、T=10^6、単一シードの Hedge 実行が手元のノート PC で 1 分以内を目安 |
| コード規約 | ruff によるリント・フォーマット |

## 8. テスト方針

1. **単体テスト**: 射影の正当性(KKT 条件の数値確認)、損失推定の不偏性(モンテカルロ)、学習率スケジュールの値
2. **理論整合性テスト**: Hedge のリグレットが理論上界 √(T log K / 2)(η 最適時)を下回ることを、複数シード平均で小規模 T(例: 10^4)にて検証
3. **レジーム挙動テスト**: 確率的環境で Tsallis-INF の擬似リグレットが log T スケール、敵対的環境で √T スケールに漸近することを傾き推定で確認(許容誤差付き)
4. **回帰テスト**: 固定シードでの `RunResult` のハッシュ比較により、意図しない数値変化を検出

## 9. マイルストーン

| 版 | 内容 | 完了条件 |
|---|---|---|
| v0.1 | core + Hedge + 確率的/敵対的環境 + リグレットプロット | Hedge の理論整合性テストが通過 |
| v0.2 | FTRL フレームワーク + Tsallis-INF + 汚染環境 | 3レジームでの BOBW 挙動が再現できる |
| v0.3 | OLO(射影勾配)+ 負荷分散環境 | 負荷分散の基本実験が回る |
| v0.4 | 実験ランナー整備、設定保存、ドキュメント | 論文用実験の一括実行・再現が可能 |

## 10. 未決事項

- 組合せ的決定集合(負荷分散のスケジュール空間)の表現方法: 明示的な頂点集合か、線形最適化オラクル経由か
- 擬似リグレットの定義を環境側・学習者側のどちらに寄せるか(現案は環境が期待損失を提供)
- 適応学習率の内部状態インターフェースの粒度(汎用性と実装の簡潔さのトレードオフ)
